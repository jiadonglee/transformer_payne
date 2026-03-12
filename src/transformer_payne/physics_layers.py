"""
Physics Layers for TransformerPayne

物理嵌入层模块，实现：
1. 物理量预测头 (吸收系数、散射系数、温度)
2. 光学深度计算
3. 辐射转移方程求解
4. 角度积分

所有函数都是 JAX 可微分的。
"""

from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
from jax import nn
from flax import linen as flax_nn


# ============================================================================
# 物理常数 (CGS 单位制)
# ============================================================================
class PhysicalConstants:
    """物理常数"""
    H = 6.62607015e-27      # Planck 常数 [erg·s]
    K = 1.380649e-16        # Boltzmann 常数 [erg/K]
    C = 2.99792458e10       # 光速 [cm/s]
    SIGMA_SB = 5.670374419e-5  # Stefan-Boltzmann 常数 [erg/cm²/s/K⁴]
    ME = 9.10938356e-28     # 电子质量 [g]
    PI = jnp.pi


# ============================================================================
# 物理量预测头
# ============================================================================
class PhysicalQuantitiesHead(flax_nn.Module):
    """
    从 Transformer 隐藏状态预测物理量
    
    输出:
    - kappa: 吸收系数 [cm⁻¹] (波长相关)
    - sigma: 散射系数 [cm⁻¹] (波长相关)
    - T: 温度 [K] (深度相关)
    """
    dim: int = 256
    n_wavelengths: int = 9875
    n_depth_layers: int = 64
    use_bias: bool = False
    activation_fn: str = "gelu"
    
    @flax_nn.compact
    def __call__(self, x, train=False):
        """
        Args:
            x: Transformer 输出 (batch, n_wavelengths, dim)
            train: 是否训练模式
        
        Returns:
            Dict containing:
                - kappa: (batch, n_wavelengths) 吸收系数
                - sigma: (batch, n_wavelengths) 散射系数  
                - T: (batch, n_depth_layers) 温度分布
        """
        activation_fn = {
            "gelu": flax_nn.gelu,
            "relu": flax_nn.relu,
            "sigmoid": flax_nn.sigmoid,
        }.get(self.activation_fn, flax_nn.gelu)
        
        batch_size = x.shape[0] if x.ndim > 2 else 1
        
        # 全局池化 (如果输入是序列)
        if x.ndim == 3:
            # (batch, seq_len, dim) → (batch, dim)
            x_global = jnp.mean(x, axis=1)
        else:
            x_global = x
        
        # 共享 MLP 骨干
        x_shared = flax_nn.Dense(512, use_bias=self.use_bias)(x_global)
        x_shared = activation_fn(x_shared)
        x_shared = flax_nn.Dense(256, use_bias=self.use_bias)(x_shared)
        x_shared = activation_fn(x_shared)
        
        # 预测吸收系数 (波长相关)
        kappa = flax_nn.Dense(self.n_wavelengths, use_bias=self.use_bias)(x_shared)
        kappa = nn.softplus(kappa)  # 确保非负
        kappa = kappa * 1e-20  # 缩放到合理范围 [cm⁻¹]
        
        # 预测散射系数 (波长相关)
        sigma = flax_nn.Dense(self.n_wavelengths, use_bias=self.use_bias)(x_shared)
        sigma = nn.softplus(sigma)
        sigma = sigma * 1e-22  # 缩放到合理范围 [cm⁻¹]
        
        # 预测温度分布 (深度相关)
        T = flax_nn.Dense(self.n_depth_layers, use_bias=self.use_bias)(x_shared)
        T = nn.sigmoid(T)  # [0, 1]
        T = T * 10000 + 2000  # 映射到 [2000, 12000] K
        
        return {
            "kappa": kappa,
            "sigma": sigma,
            "T": T,
        }


# ============================================================================
# 光学深度计算
# ============================================================================
class OpticalDepthComputer(flax_nn.Module):
    """
    计算波长和深度相关的光学深度
    
    τ_ν(z) = ∫₀ᶻ (κ_ν + σ_ν) ρ dz'
    """
    n_depth_layers: int = 64
    
    @flax_nn.compact
    def __call__(
        self, 
        kappa: jnp.ndarray, 
        sigma: jnp.ndarray,
        depth_spacing: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Args:
            kappa: 吸收系数 (batch, n_wavelengths)
            sigma: 散射系数 (batch, n_wavelengths)
            depth_spacing: 深度间隔 (n_depth_layers,) 或 None
        
        Returns:
            tau: 光学深度 (batch, n_depth_layers, n_wavelengths)
        """
        # 总不透明度
        chi = kappa + sigma  # (batch, n_wavelengths)
        
        # 确保非负
        chi = nn.softplus(chi)
        
        # 默认深度间隔 (对数间隔)
        if depth_spacing is None:
            # 从光球层到大气上层
            depth_spacing = jnp.logspace(-8, 0, self.n_depth_layers)
        
        # 扩展到深度维度
        # chi: (batch, n_wavelengths) → (batch, 1, n_wavelengths)
        chi_expanded = chi[:, jnp.newaxis, :]
        
        # depth_spacing: (n_depth_layers,) → (1, n_depth_layers, 1)
        dz = depth_spacing[jnp.newaxis, :, jnp.newaxis]
        
        # 累积光学深度 (可微分累积和)
        # tau = cumsum(chi * dz)
        tau = jnp.cumsum(chi_expanded * dz, axis=1)
        
        return tau  # (batch, n_depth_layers, n_wavelengths)


# ============================================================================
# 源函数 (Planck 函数)
# ============================================================================
def planck_function(
    nu: jnp.ndarray, 
    T: jnp.ndarray
) -> jnp.ndarray:
    """
    计算 Planck 函数 (黑体辐射强度)
    
    B_ν(T) = (2hν³/c²) / [exp(hν/kT) - 1]
    
    Args:
        nu: 频率 [Hz] (n_freq,)
        T: 温度 [K] (batch, n_depth)
    
    Returns:
        B_nu: Planck 函数 (batch, n_depth, n_freq)
    """
    h = PhysicalConstants.H
    k = PhysicalConstants.K
    c = PhysicalConstants.C
    
    # 频率转换为角频率
    nu_expanded = nu[jnp.newaxis, jnp.newaxis, :]  # (1, 1, n_freq)
    T_expanded = T[:, :, jnp.newaxis]  # (batch, n_depth, 1)
    
    # 指数项 x = hν/kT
    # 对于典型值：h=6.6e-27, ν=1e14, k=1.4e-16, T=5000
    # x ≈ 6.6e-27 * 1e14 / (1.4e-16 * 5000) ≈ 0.94
    x = h * nu_expanded / (k * T_expanded + 1e-10)  # 避免除零
    
    # 数值稳定计算 exp(x) - 1
    # 使用 JAX 的 lax.expm1 对于小 x 更稳定
    expm1_x = jnp.expm1(x)
    
    # 对于非常小的 x，使用 Taylor 展开：exp(x) - 1 ≈ x + x²/2
    x_very_small = x < 1e-8
    expm1_very_small = x + x**2 / 2
    expm1_x = jnp.where(x_very_small, expm1_very_small, expm1_x)
    
    # Planck 函数
    # 使用对数空间计算避免溢出
    # log(B_ν) = log(2h/c²) + 3*log(ν) - log(exp(hν/kT) - 1)
    log_nu = jnp.log(jnp.abs(nu_expanded) + 1e-300)
    log_prefactor = jnp.log(2 * h / c**2) + 3 * log_nu
    
    # 计算 log(denominator)
    log_denom = jnp.log(jnp.abs(expm1_x) + 1e-300)
    
    # B_ν = exp(log(B_ν))
    log_B_nu = log_prefactor - log_denom
    
    # 裁剪到合理范围
    log_B_nu = jnp.clip(log_B_nu, -700, 700)
    B_nu = jnp.exp(log_B_nu)
    
    return B_nu  # (batch, n_depth, n_freq)


# ============================================================================
# 辐射转移方程求解器
# ============================================================================
class RadiativeTransferSolver(flax_nn.Module):
    """
    求解一维平面平行大气辐射转移方程
    
    dI_ν/dτ_ν = I_ν - S_ν
    
    使用 Eddington-Barbier 近似或数值积分
    """
    n_depth_layers: int = 64
    n_angles: int = 20
    
    def setup(self):
        """初始化 Gauss-Legendre 求积点"""
        # Gauss-Legendre 求积点和权重 (0 到 1 区间)
        import numpy as np
        mu, weights = np.polynomial.legendre.leggauss(self.n_angles)
        # 转换到 [0, 1] 区间
        mu = (mu + 1) / 2
        weights = weights / 2
        self.mu = jnp.array(mu)  # (n_angles,)
        self.weights = jnp.array(weights)  # (n_angles,)
    
    @flax_nn.compact
    def __call__(
        self,
        tau: jnp.ndarray,
        T: jnp.ndarray,
        nu: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        求解辐射转移方程
        
        Args:
            tau: 光学深度 (batch, n_depth, n_wavelengths)
            T: 温度分布 (batch, n_depth)
            nu: 频率 (n_wavelengths,)
        
        Returns:
            I_nu: 辐射强度 (batch, n_wavelengths)
            F_nu: 辐射通量 (batch, n_wavelengths)
        """
        batch_size = tau.shape[0]
        
        # 计算源函数 (Planck 函数)
        S_nu = planck_function(nu, T)  # (batch, n_depth, n_freq)
        
        # 辐射强度计算 (简化形式)
        # I_ν(τ_ν, μ) ≈ S_ν(τ_ν=μ) (Eddington-Barbier 近似)
        # 或使用数值积分
        
        # 方法 1: Eddington-Barbier 近似
        # I_nu ≈ S_nu(τ_ν = 1)
        # 找到 τ ≈ 1 的深度层
        tau_ref = 1.0
        tau_diff = jnp.abs(tau - tau_ref)
        idx_closest = jnp.argmin(tau_diff, axis=1)  # (batch, n_freq)
        
        # 收集对应深度的源函数值
        I_nu_eb = jnp.take_along_axis(
            S_nu, 
            idx_closest[:, jnp.newaxis, :], 
            axis=1
        )[:, 0, :]  # (batch, n_freq)
        
        # 方法 2: 数值积分 (更精确)
        # I_ν(0, μ) = ∫₀^∞ S_ν(τ_ν) e^{-τ_ν/μ} d(τ_ν/μ)
        I_nu_integral = self._formal_integral(tau, S_nu)
        
        # 使用数值积分结果
        I_nu = I_nu_integral
        
        # 角度积分计算通量
        # F_ν = 2π ∫ I_ν μ dμ
        F_nu = self._angular_integration(I_nu)
        
        return I_nu, F_nu
    
    def _formal_integral(
        self,
        tau: jnp.ndarray,
        S_nu: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        形式积分求解辐射转移方程
        
        I_ν(0) = ∫₀^∞ S_ν(τ_ν) e^{-τ_ν} dτ_ν
        
        Args:
            tau: 光学深度 (batch, n_depth, n_freq)
            S_nu: 源函数 (batch, n_depth, n_freq)
        
        Returns:
            I_nu: 表面辐射强度 (batch, n_freq)
        """
        # 计算权重 e^{-τ}
        exp_neg_tau = jnp.exp(-tau)
        
        # 计算深度间隔
        dtau = jnp.diff(tau, axis=1, prepend=0)
        
        # 数值积分 (梯形法则)
        # ∫ S_ν e^{-τ} dτ ≈ Σ S_ν(τ_i) e^{-τ_i} Δτ_i
        integrand = S_nu * exp_neg_tau * dtau
        I_nu = jnp.sum(integrand, axis=1)  # (batch, n_freq)
        
        return I_nu
    
    def _angular_integration(
        self,
        I_nu: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        角度积分计算通量
        
        F_ν = 2π ∫₀¹ I_ν(μ) μ dμ
        
        Args:
            I_nu: 辐射强度 (batch, n_freq)
        
        Returns:
            F_nu: 辐射通量 (batch, n_freq)
        """
        # 简化：假设各向同性辐射
        # F_ν = π I_ν
        F_nu = jnp.pi * I_nu
        
        # 更精确：使用 Gauss-Legendre 求积
        # F_nu = 2π Σ_i w_i I_ν(μ_i) μ_i
        # (需要 I_nu 依赖于 μ，这里简化处理)
        
        return F_nu


# ============================================================================
# 能量定理约束
# ============================================================================
def energy_theorem_residual(
    F_nu: jnp.ndarray,
    nu: jnp.ndarray,
    T_eff: jnp.ndarray,
) -> jnp.ndarray:
    """
    计算能量定理残差
    
    ∫ F_ν dν = σ T_eff⁴
    
    Args:
        F_nu: 辐射通量 (batch, n_freq)
        nu: 频率 (n_freq,)
        T_eff: 有效温度 (batch,)
    
    Returns:
        residual: 能量定理残差 (batch,)
    """
    sigma = PhysicalConstants.SIGMA_SB
    
    # 数值积分 ∫ F_ν dν
    # 使用梯形法则
    dnu = jnp.diff(nu, prepend=nu[0])
    integrated_flux = jnp.sum(F_nu * dnu, axis=1)  # (batch,)
    
    # Stefan-Boltzmann 定律
    stefan_boltzmann = sigma * T_eff**4
    
    # 残差
    residual = integrated_flux - stefan_boltzmann
    
    return residual  # (batch,)


# ============================================================================
# 完整的物理嵌入模块
# ============================================================================
class PhysicsEmbeddedModule(flax_nn.Module):
    """
    完整的物理嵌入模块
    
    将 Transformer 输出 → 物理量 → RTE 求解 → 光谱
    """
    dim: int = 256
    n_wavelengths: int = 9875
    n_depth_layers: int = 64
    n_angles: int = 20
    
    @flax_nn.compact
    def __call__(
        self,
        transformer_output: jnp.ndarray,
        wavelengths: jnp.ndarray,
        stellar_params: jnp.ndarray,
        train: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        前向传播
        
        Args:
            transformer_output: Transformer 输出 (batch, seq_len, dim)
            wavelengths: 波长 [Å] (n_wavelengths,)
            stellar_params: 恒星参数 (batch, 4) [Teff, logg, [Fe/H], [α/Fe]]
            train: 是否训练模式
        
        Returns:
            Dict containing:
                - intensity: 辐射强度 (batch, n_wavelengths)
                - flux: 辐射通量 (batch, n_wavelengths)
                - kappa: 吸收系数
                - sigma: 散射系数
                - T: 温度分布
                - tau: 光学深度
        """
        # 1. 预测物理量
        phys_head = PhysicalQuantitiesHead(
            dim=self.dim,
            n_wavelengths=self.n_wavelengths,
            n_depth_layers=self.n_depth_layers,
        )
        physical_quantities = phys_head(transformer_output, train=train)
        
        kappa = physical_quantities["kappa"]
        sigma = physical_quantities["sigma"]
        T = physical_quantities["T"]
        
        # 2. 计算光学深度
        tau_computer = OpticalDepthComputer(n_depth_layers=self.n_depth_layers)
        tau = tau_computer(kappa, sigma)
        
        # 3. 波长转频率
        c = PhysicalConstants.C
        nu = c / (wavelengths * 1e-8)  # Å → Hz
        
        # 4. 求解辐射转移方程
        rt_solver = RadiativeTransferSolver(
            n_depth_layers=self.n_depth_layers,
            n_angles=self.n_angles,
        )
        I_nu, F_nu = rt_solver(tau, T, nu)
        
        # 5. 频率转回波长
        # F_λ = F_ν * |dν/dλ| = F_ν * c / λ²
        lambda_cm = wavelengths * 1e-8
        F_lambda = F_nu * c / (lambda_cm**2)
        I_lambda = I_nu * c / (lambda_cm**2)
        
        return {
            "intensity": I_lambda,
            "flux": F_lambda,
            "kappa": kappa,
            "sigma": sigma,
            "T": T,
            "tau": tau,
        }


# ============================================================================
# 工具函数
# ============================================================================
def compute_physics_losses(
    outputs: Dict[str, jnp.ndarray],
    stellar_params: jnp.ndarray,
    wavelengths: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """
    计算物理约束损失
    
    Args:
        outputs: 模型输出 (包含 kappa, sigma, T, flux 等)
        stellar_params: 恒星参数 (batch, 4)
        wavelengths: 波长
    
    Returns:
        Dict containing loss terms
    """
    # 1. 非负约束 (吸收系数、散射系数)
    kappa_nonneg = jnp.mean(nn.relu(-outputs["kappa"]))
    sigma_nonneg = jnp.mean(nn.relu(-outputs["sigma"]))
    
    # 2. 能量定理残差
    nu = PhysicalConstants.C / (wavelengths * 1e-8)
    T_eff = stellar_params[:, 0]  # 有效温度
    et_residual = energy_theorem_residual(outputs["flux"], nu, T_eff)
    et_loss = jnp.mean(et_residual**2)
    
    # 3. 温度平滑性约束
    T = outputs["T"]  # (batch, n_depth)
    T_smooth = jnp.mean(jnp.diff(T, axis=1)**2)
    
    return {
        "kappa_nonneg": kappa_nonneg,
        "sigma_nonneg": sigma_nonneg,
        "et_loss": et_loss,
        "T_smooth": T_smooth,
    }
