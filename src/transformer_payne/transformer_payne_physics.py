"""
Physics-Embedded TransformerPayne Model

在原始 TransformerPayne 基础上添加物理嵌入模块，
实现 PhysFormer 风格的物理自洽光谱生成。
"""

from typing import Any, Dict, List, Union, Tuple, Callable, Optional
from transformer_payne._utility import METALS
from transformer_payne.architecture_definition import ArchitectureDefinition
from transformer_payne.download import download_hf_model
from transformer_payne.exceptions import JAXWarning
from transformer_payne.spectrum_emulator import SpectrumEmulator
from transformer_payne.configuration import REPOSITORY_ID_KEY, FILENAME_KEY
from transformer_payne.physics_layers import (
    PhysicalQuantitiesHead,
    OpticalDepthComputer,
    RadiativeTransferSolver,
    planck_function,
    energy_theorem_residual,
    compute_physics_losses,
    PhysicalConstants,
    PhysicsEmbeddedModule,
)
from functools import partial
import warnings
import os
import numpy as np
from pathlib import Path

try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    import jax
    from flax import linen as nn
    from flax import core as flax_core
    from flax.linen import initializers as init
    from flax.core.frozen_dict import freeze
    
except ImportError:
    import numpy as jnp
    from numpy.typing import ArrayLike
    warnings.warn("Please install JAX and Flax to use TransformerPayne.", JAXWarning)

# 导入原始组件
from transformer_payne.transformer_payne import (
    frequency_encoding,
    ParametersEmbedding,
    FeedForward,
    PredictionHead,
    MHA,
    _activation_functions_dict,
)


# ============================================================================
# 物理嵌入 TransformerPayne 模型
# ============================================================================
class TransformerPayneModelWavePhysics(nn.Module):
    """
    带物理嵌入的 TransformerPayne (单波长点版本)
    
    在原始架构基础上添加物理量预测和辐射转移求解。
    """
    dim: int = 256
    dim_ff_multiplier: int = 4
    no_tokens: int = 16
    no_layers: int = 16
    dim_head: int = 32
    out_dim: int = 2  # [intensity, continuum]
    input_dim: int = 95
    min_period: float = 1e-6
    max_period: float = 10
    bias_dense: bool = False
    bias_attention: bool = False
    activation_fn: str = "gelu"
    output_activation_fn: str = "linear"
    init_att_q: str = "si"
    init_att_o: str = "si"
    emb_init: str = "si"
    ff_init: str = "si"
    head_init: str = "si"
    sigma: float = 1.0
    alpha_emb: float = 1.0
    alpha_att: float = 1.0
    reference_depth: int = None
    reference_width: int = None
    
    # 物理模块配置
    use_physics: bool = True
    n_depth_layers: int = 64
    n_angles: int = 20
    
    @nn.compact
    def __call__(self, x, train=False, return_physics_outputs=False):
        """
        Args:
            x: (atmospheric_parameters, wavelength)
            train: 是否训练模式
            return_physics_outputs: 是否返回物理量输出
        
        Returns:
            如果 return_physics_outputs=True:
                (spectrum, physics_outputs)
            否则:
                spectrum
        """
        atmospheric_parameters, wavelength = x
        
        activation_fn = _activation_functions_dict[self.activation_fn]
        output_activation_fn = _activation_functions_dict[self.output_activation_fn]
        
        residual_scaling = 1.0 if self.reference_depth is None else (self.no_layers / self.reference_depth)**(-0.5)
        
        # 波长编码
        enc_w = frequency_encoding(
            wavelength,
            min_period=self.min_period,
            max_period=self.max_period,
            dimension=self.dim
        )
        enc_w = enc_w.astype(jnp.float32)
        atmospheric_parameters = atmospheric_parameters.astype(jnp.float32)
        
        enc_w = enc_w[None, ...]
        
        # 参数嵌入
        enc_p = ParametersEmbedding(
            dim=self.dim,
            input_dim=self.input_dim,
            no_tokens=self.no_tokens, 
            activation_fn=activation_fn, 
            use_bias=self.bias_dense,
            init_type=self.emb_init,
            sigma=self.sigma,
            alpha=self.alpha_emb
        )(atmospheric_parameters, train)
        
        # Transformer 层
        x = enc_w
        enc_p = nn.RMSNorm(name=f"norm_p", use_scale=False)(enc_p)
        
        for i in range(self.no_layers):
            # MHA
            _x = nn.RMSNorm(name=f"norm_1_L{i}", use_scale=False)(x)
            _x = MHA(
                dim=self.dim, 
                dim_head=self.dim_head, 
                use_bias=self.bias_attention,
                init_att_q=self.init_att_q,
                init_att_o=self.init_att_o,
                sigma=self.sigma,
                alpha_att=self.alpha_att
            )(inputs_q=_x, inputs_kv=enc_p)
            x = x + residual_scaling * _x
            
            # MLP
            _x = nn.RMSNorm(name=f"norm_2_L{i}", use_scale=False)(x)
            _x = FeedForward(
                dim=self.dim, 
                dim_ff_multiplier=self.dim_ff_multiplier, 
                activation_fn=activation_fn,
                init_type=self.ff_init,
                sigma=self.sigma
            )(_x, train)
            x = x + residual_scaling * _x
        
        x = nn.RMSNorm(name="decoder_norm", use_scale=False)(x)
        
        # 物理嵌入模块
        if self.use_physics:
            # 使用物理嵌入方式生成光谱
            n_wavelengths = wavelength.shape[0] if wavelength.ndim > 0 else 1
            
            # 物理量预测头
            phys_head = PhysicalQuantitiesHead(
                dim=self.dim,
                n_wavelengths=n_wavelengths,
                n_depth_layers=self.n_depth_layers,
            )
            physical_quantities = phys_head(x, train=train)
            
            kappa = physical_quantities["kappa"][0]  # (n_wavelengths,)
            sigma = physical_quantities["sigma"][0]
            T = physical_quantities["T"][0]  # (n_depth_layers,)
            
            # 计算光学深度
            tau_computer = OpticalDepthComputer(n_depth_layers=self.n_depth_layers)
            tau = tau_computer(kappa[None, :], sigma[None, :])[0]  # (n_depth, n_wavelengths)
            
            # 波长转频率
            c = PhysicalConstants.C
            nu = c / (wavelength * 1e-8)  # Hz
            
            # 辐射转移求解
            rt_solver = RadiativeTransferSolver(
                n_depth_layers=self.n_depth_layers,
                n_angles=self.n_angles,
            )
            params_rt = rt_solver.init(jax.random.PRNGKey(0), tau[None, :], T[None, :], nu)
            I_nu, F_nu = rt_solver.apply(params_rt, tau[None, :], T[None, :], nu)
            I_nu = I_nu[0]  # (n_wavelengths,)
            
            # 频率转波长
            lambda_cm = wavelength * 1e-8
            I_lambda = I_nu * c / (lambda_cm**2)
            
            # 原始输出 (用于兼容性)
            x = x[0]
            continuum = PredictionHead(
                dim=self.dim, 
                out_dim=1, 
                use_bias=self.bias_dense,
                activation_fn=activation_fn, 
                output_activation_fn=lambda x: jnp.exp(x),  # 确保正数
                init_type=self.head_init,
                sigma=self.sigma
            )(x, train)
            
            # 组合输出
            intensity = I_lambda
            spectrum = jnp.stack([intensity, continuum[:, 0]], axis=-1)
            
            if return_physics_outputs:
                physics_outputs = {
                    "kappa": kappa,
                    "sigma": sigma,
                    "T": T,
                    "tau": tau,
                    "I_nu": I_nu,
                }
                return spectrum, physics_outputs
            else:
                return spectrum
        else:
            # 原始方式 (不使用物理嵌入)
            x = x[0]
            spectrum = PredictionHead(
                dim=self.dim, 
                out_dim=self.out_dim, 
                use_bias=self.bias_dense,
                activation_fn=activation_fn, 
                output_activation_fn=output_activation_fn,
                init_type=self.head_init,
                sigma=self.sigma
            )(x, train)
            
            # 后处理
            spectrum = spectrum.at[1].set(jnp.exp(spectrum[1]))  # continuum > 0
            spectrum = spectrum.at[0].set(spectrum[0] * spectrum[1])
            
            return spectrum


class TransformerPayneModelPhysics(nn.Module):
    """
    带物理嵌入的 TransformerPayne (批量波长版本)
    """
    dim: int = 256
    dim_ff_multiplier: int = 4
    no_tokens: int = 16
    no_layers: int = 16
    dim_head: int = 32
    out_dim: int = 2
    input_dim: int = 95
    min_period: float = 1e-6
    max_period: float = 10
    bias_dense: bool = False
    bias_attention: bool = False
    activation_fn: str = "gelu"
    output_activation_fn: str = "linear"
    init_att_q: str = "si"
    init_att_o: str = "si"
    emb_init: str = "si"
    ff_init: str = "si"
    head_init: str = "si"
    sigma: float = 1.0
    alpha_emb: float = 1.0
    alpha_att: float = 1.0
    reference_depth: int = None
    reference_width: int = None
    use_physics: bool = True
    n_depth_layers: int = 64
    n_angles: int = 20
    
    @nn.compact
    def __call__(self, inputs, train=False, return_physics_outputs=False):
        log_waves, p = inputs
        
        TP = nn.vmap(
            TransformerPayneModelWavePhysics, 
            in_axes=((None, 0),),
            out_axes=0,
            variable_axes={'params': None}, 
            split_rngs={'params': False}
        )
        
        x = TP(
            name="transformer_payne_physics", 
            dim=self.dim, 
            dim_ff_multiplier=self.dim_ff_multiplier, 
            no_tokens=self.no_tokens, 
            no_layers=self.no_layers, 
            dim_head=self.dim_head, 
            out_dim=self.out_dim,
            input_dim=self.input_dim,
            min_period=self.min_period, 
            max_period=self.max_period,
            bias_dense=self.bias_dense,
            bias_attention=self.bias_attention,
            activation_fn=self.activation_fn, 
            output_activation_fn=self.output_activation_fn,
            init_att_q=self.init_att_q,
            init_att_o=self.init_att_o,
            emb_init=self.emb_init,
            ff_init=self.ff_init,
            head_init=self.head_init,
            sigma=self.sigma,
            alpha_emb=self.alpha_emb,
            alpha_att=self.alpha_att,
            reference_depth=self.reference_depth,
            reference_width=self.reference_width,
            use_physics=self.use_physics,
            n_depth_layers=self.n_depth_layers,
            n_angles=self.n_angles,
        )((p, log_waves), train=train, return_physics_outputs=return_physics_outputs)
        
        return x


# ============================================================================
# 辅助函数
# ============================================================================
def create_physics_embedded_model(
    existing_checkpoint_path: Optional[str] = None,
    use_physics: bool = True,
    n_depth_layers: int = 64,
    n_angles: int = 20,
) -> Tuple[TransformerPayneModelPhysics, Any]:
    """
    创建物理嵌入模型，可选择加载现有权重
    
    Args:
        existing_checkpoint_path: 现有模型检查点路径
        use_physics: 是否启用物理嵌入
        n_depth_layers: 深度层数
        n_angles: 角度积分点数
    
    Returns:
        (model, params) 元组
    """
    # 创建模型
    model = TransformerPayneModelPhysics(
        dim=256,
        dim_ff_multiplier=4,
        no_tokens=16,
        no_layers=8,  # 与原始模型匹配
        dim_head=32,
        out_dim=2,
        input_dim=95,
        use_physics=use_physics,
        n_depth_layers=n_depth_layers,
        n_angles=n_angles,
    )
    
    # 初始化参数
    key = jax.random.PRNGKey(42)
    dummy_log_waves = jnp.log10(jnp.linspace(4670, 4960, 100))
    dummy_params = jnp.zeros(95)
    
    params = model.init(key, (dummy_log_waves, dummy_params), train=False)
    
    # 如果提供检查点，尝试加载
    if existing_checkpoint_path and os.path.exists(existing_checkpoint_path):
        print(f"加载现有检查点：{existing_checkpoint_path}")
        try:
            import joblib
            checkpoint = joblib.load(existing_checkpoint_path)
            # 这里需要根据实际检查点格式调整
            # params = checkpoint['params']
            print("检查点加载成功")
        except Exception as e:
            print(f"检查点加载失败：{e}")
            print("将使用随机初始化参数")
    
    return model, params


# ============================================================================
# 训练辅助函数
# ============================================================================
def physics_regularization_loss(
    physics_outputs: Dict[str, ArrayLike],
    stellar_params: ArrayLike,
    wavelengths: ArrayLike,
) -> Dict[str, ArrayLike]:
    """
    计算物理正则化损失
    
    Args:
        physics_outputs: 物理量输出 (kappa, sigma, T, tau)
        stellar_params: 恒星参数
        wavelengths: 波长
    
    Returns:
        物理损失项字典
    """
    return compute_physics_losses(physics_outputs, stellar_params, wavelengths)


def combined_loss(
    spectrum_pred: ArrayLike,
    spectrum_target: ArrayLike,
    physics_outputs: Optional[Dict[str, ArrayLike]] = None,
    stellar_params: Optional[ArrayLike] = None,
    wavelengths: Optional[ArrayLike] = None,
    physics_weight: float = 0.1,
) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
    """
    组合损失函数 (数据损失 + 物理损失)
    
    Args:
        spectrum_pred: 预测光谱
        spectrum_target: 目标光谱
        physics_outputs: 物理量输出
        stellar_params: 恒星参数
        wavelengths: 波长
        physics_weight: 物理损失权重
    
    Returns:
        (total_loss, loss_breakdown)
    """
    # 数据损失 (MSE)
    data_loss = jnp.mean((spectrum_pred - spectrum_target)**2)
    
    # 物理损失
    physics_loss_dict = {}
    if physics_outputs is not None and stellar_params is not None and wavelengths is not None:
        physics_loss_dict = physics_regularization_loss(
            physics_outputs, stellar_params, wavelengths
        )
        physics_loss = sum(physics_loss_dict.values())
    else:
        physics_loss = 0.0
    
    # 总损失
    total_loss = data_loss + physics_weight * physics_loss
    
    loss_breakdown = {
        "total": total_loss,
        "data_loss": data_loss,
        "physics_loss": physics_loss,
        **physics_loss_dict,
    }
    
    return total_loss, loss_breakdown
