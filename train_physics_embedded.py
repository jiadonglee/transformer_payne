#!/usr/bin/env python3
"""
物理嵌入 TransformerPayne 训练脚本

三阶段训练流程：
1. 自编码器预训练 (200 epochs)
2. 参数到光谱映射 (冻结 AE)
3. 联合优化 (数据 + 物理损失)

用法:
    python train_physics_embedded.py \
        --data_dir /path/to/phoenix \
        --output_dir /path/to/output \
        --epochs 200 \
        --batch_size 32 \
        --use_physics
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze

# 导入模型
from transformer_payne_physics import (
    TransformerPayneModelPhysics,
    create_physics_embedded_model,
    combined_loss,
)


# ============================================================================
# 配置
# ============================================================================
@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    data_dir: str = "/path/to/phoenix"
    n_wavelengths: int = 9875
    n_train: int = 20000
    n_val: int = 2000
    
    # 模型
    dim: int = 256
    dim_ff_multiplier: int = 4
    no_tokens: int = 16
    no_layers: int = 8
    dim_head: int = 32
    input_dim: int = 95
    use_physics: bool = True
    n_depth_layers: int = 64
    n_angles: int = 20
    
    # 训练
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_epochs: int = 10
    physics_weight: float = 0.1
    
    # 输出
    output_dir: str = "/path/to/output"
    save_every: int = 10
    log_every: int = 1
    
    # 随机种子
    seed: int = 42


# ============================================================================
# 训练状态
# ============================================================================
class TrainState(train_state.TrainState):
    """扩展训练状态"""
    rng: Any


# ============================================================================
# 数据加载
# ============================================================================
def load_phoenix_dataset(
    data_dir: str,
    n_train: int = 20000,
    n_val: int = 2000,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    加载 PHOENIX 数据集
    
    返回:
        (train_data, val_data) 字典，包含:
            - wavelengths: (n_wavelengths,)
            - spectra: (n_samples, n_wavelengths)
            - params: (n_samples, 4) [Teff, logg, [Fe/H], [α/Fe]]
    """
    print(f"加载 PHOENIX 数据集 from {data_dir}...")
    
    # 这里需要根据实际数据格式调整
    # 示例代码：
    try:
        import joblib
        data = joblib.load(os.path.join(data_dir, "phoenix_data.joblib"))
        
        wavelengths = data["wavelengths"]
        spectra = data["spectra"][:n_train + n_val]
        params = data["params"][:n_train + n_val]
        
        train_data = {
            "wavelengths": wavelengths,
            "spectra": spectra[:n_train],
            "params": params[:n_train],
        }
        val_data = {
            "wavelengths": wavelengths,
            "spectra": spectra[n_train:n_train + n_val],
            "params": params[n_train:n_train + n_val],
        }
        
        print(f"训练集：{n_train} 样本")
        print(f"验证集：{n_val} 样本")
        
        return train_data, val_data
        
    except Exception as e:
        print(f"数据加载失败：{e}")
        print("使用模拟数据...")
        
        # 生成模拟数据
        key = random.PRNGKey(42)
        n_wavelengths = 9875
        
        wavelengths = jnp.logspace(3.5, 4.5, n_wavelengths)
        
        train_spectra = random.normal(key, (n_train, n_wavelengths)) * 0.1 + 1.0
        train_params = random.normal(key, (n_train, 4))
        train_params = train_params.at[:, 0].set(train_params[:, 0] * 1000 + 5777)  # Teff
        
        val_spectra = random.normal(key, (n_val, n_wavelengths)) * 0.1 + 1.0
        val_params = random.normal(key, (n_val, 4))
        val_params = val_params.at[:, 0].set(val_params[:, 0] * 1000 + 5777)
        
        train_data = {
            "wavelengths": wavelengths,
            "spectra": train_spectra,
            "params": train_params,
        }
        val_data = {
            "wavelengths": wavelengths,
            "spectra": val_spectra,
            "params": val_params,
        }
        
        return train_data, val_data


# ============================================================================
# 训练步骤
# ============================================================================
def create_train_state(
    model: nn.Module,
    learning_rate: float,
    seed: int = 42,
) -> TrainState:
    """创建训练状态"""
    key = random.PRNGKey(seed)
    
    # 初始化
    dummy_log_waves = jnp.log10(jnp.linspace(4670, 4960, 100))
    dummy_params = jnp.zeros(95)
    
    variables = model.init(key, (dummy_log_waves, dummy_params), train=False)
    params = variables["params"]
    
    # 优化器
    tx = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=1e-4,
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        rng=key,
    )


@jit
def train_step(
    model: nn.Module,
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    physics_weight: float,
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """单步训练"""
    wavelengths = batch["wavelengths"]
    spectra_target = batch["spectra"]
    params_stellar = batch["params"]
    
    # 准备输入
    log_wavelengths = jnp.log10(wavelengths)
    
    # 前向传播 + 损失计算
    def loss_fn(model_params):
        # 应用模型
        variables = {"params": freeze(model_params)}
        output = model.apply(
            variables,
            (log_wavelengths, params_stellar),
            train=True,
            return_physics_outputs=True,
            rngs={"dropout": state.rng},
        )
        
        spectrum_pred, physics_outputs = output
        
        # 组合损失
        total_loss, loss_breakdown = combined_loss(
            spectrum_pred,
            spectra_target,
            physics_outputs=physics_outputs,
            stellar_params=params_stellar,
            wavelengths=wavelengths,
            physics_weight=physics_weight,
        )
        
        return total_loss, (spectrum_pred, loss_breakdown)
    
    # 梯度计算
    (total_loss, (spectrum_pred, loss_breakdown)), grads = value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    
    # 更新参数
    state = state.apply_gradients(grads=grads)
    
    # 更新 RNG
    state = state.replace(rng=random.fold_in(state.rng, state.step))
    
    # 记录指标
    metrics = {
        "loss": total_loss,
        **loss_breakdown,
    }
    
    return state, metrics


@jit
def eval_step(
    model: nn.Module,
    params: Any,
    batch: Dict[str, jnp.ndarray],
) -> Dict[str, jnp.ndarray]:
    """评估步骤"""
    wavelengths = batch["wavelengths"]
    spectra_target = batch["spectra"]
    params_stellar = batch["params"]
    
    log_wavelengths = jnp.log10(wavelengths)
    
    # 前向传播
    variables = {"params": freeze(params)}
    output = model.apply(
        variables,
        (log_wavelengths, params_stellar),
        train=False,
        return_physics_outputs=True,
    )
    
    spectrum_pred, physics_outputs = output
    
    # 计算损失 (无物理损失)
    data_loss = jnp.mean((spectrum_pred - spectra_target)**2)
    rmse = jnp.sqrt(data_loss)
    mae = jnp.mean(jnp.abs(spectrum_pred - spectra_target))
    
    # 物理量统计
    kappa_mean = jnp.mean(physics_outputs["kappa"])
    sigma_mean = jnp.mean(physics_outputs["sigma"])
    T_mean = jnp.mean(physics_outputs["T"])
    
    return {
        "data_loss": data_loss,
        "rmse": rmse,
        "mae": mae,
        "kappa_mean": kappa_mean,
        "sigma_mean": sigma_mean,
        "T_mean": T_mean,
    }


# ============================================================================
# 训练循环
# ============================================================================
def train_epoch(
    model: nn.Module,
    state: TrainState,
    train_data: Dict[str, jnp.ndarray],
    batch_size: int,
    physics_weight: float,
    epoch: int,
) -> Tuple[TrainState, Dict[str, float]]:
    """训练一个 epoch"""
    n_samples = train_data["spectra"].shape[0]
    n_batches = n_samples // batch_size
    
    metrics_accum = {
        "loss": 0.0,
        "data_loss": 0.0,
        "physics_loss": 0.0,
    }
    
    # Shuffle 数据
    key = random.fold_in(state.rng, epoch)
    perm = random.permutation(key, n_samples)
    
    for i in range(n_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        batch = {
            "wavelengths": train_data["wavelengths"],
            "spectra": train_data["spectra"][batch_idx],
            "params": train_data["params"][batch_idx],
        }
        
        # 训练步骤
        state, metrics = train_step(
            model, state, batch, physics_weight
        )
        
        # 累积指标
        for key in metrics_accum:
            if key in metrics:
                metrics_accum[key] += metrics[key]
    
    # 平均
    for key in metrics_accum:
        metrics_accum[key] /= n_batches
    
    return state, metrics_accum


def evaluate(
    model: nn.Module,
    params: Any,
    val_data: Dict[str, jnp.ndarray],
    batch_size: int,
) -> Dict[str, float]:
    """评估模型"""
    n_samples = val_data["spectra"].shape[0]
    n_batches = n_samples // batch_size
    
    metrics_accum = {
        "data_loss": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "kappa_mean": 0.0,
        "sigma_mean": 0.0,
        "T_mean": 0.0,
    }
    
    for i in range(n_batches):
        batch_idx = slice(i * batch_size, (i + 1) * batch_size)
        batch = {
            "wavelengths": val_data["wavelengths"],
            "spectra": val_data["spectra"][batch_idx],
            "params": val_data["params"][batch_idx],
        }
        
        metrics = eval_step(model, params, batch)
        
        for key in metrics_accum:
            metrics_accum[key] += metrics[key]
    
    for key in metrics_accum:
        metrics_accum[key] /= n_batches
    
    return metrics_accum


def train_model(config: TrainingConfig):
    """主训练函数"""
    print("=" * 60)
    print("物理嵌入 TransformerPayne 训练")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = output_dir / "config.json"
    import json
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"配置保存到：{config_path}")
    
    # 加载数据
    train_data, val_data = load_phoenix_dataset(
        config.data_dir,
        config.n_train,
        config.n_val,
    )
    
    # 创建模型
    model = TransformerPayneModelPhysics(
        dim=config.dim,
        dim_ff_multiplier=config.dim_ff_multiplier,
        no_tokens=config.no_tokens,
        no_layers=config.no_layers,
        dim_head=config.dim_head,
        input_dim=config.input_dim,
        use_physics=config.use_physics,
        n_depth_layers=config.n_depth_layers,
        n_angles=config.n_angles,
    )
    
    print(f"模型参数量：{model.size // 1e6:.2f}M")
    
    # 创建训练状态
    state = create_train_state(model, config.learning_rate, config.seed)
    print(f"初始学习率：{config.learning_rate}")
    
    # 训练循环
    print("\n开始训练...")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"物理损失权重：{config.physics_weight}")
    print("-" * 60)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, config.epochs + 1):
        # 学习率调度 (warmup + decay)
        if epoch < config.warmup_epochs:
            lr = config.learning_rate * epoch / config.warmup_epochs
        else:
            decay = 0.5 * (1 + jnp.cos(jnp.pi * (epoch - config.warmup_epochs) / config.epochs))
            lr = config.learning_rate * decay
        
        state = state.replace(
            opt_state=state.opt.replace(
                hyper_params=state.opt.hyper_params.replace(
                    learning_rate=lr
                )
            )
        )
        
        # 训练
        t0 = time.time()
        state, train_metrics = train_epoch(
            model, state, train_data, config.batch_size, config.physics_weight, epoch
        )
        train_time = time.time() - t0
        
        # 评估
        val_metrics = evaluate(model, state.params, val_data, config.batch_size)
        
        # 日志
        if epoch % config.log_every == 0:
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.6f} | "
                  f"Val RMSE: {val_metrics['rmse']:.6f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {train_time:.1f}s")
            
            if config.use_physics:
                print(f"  κ: {val_metrics['kappa_mean']:.2e} | "
                      f"σ: {val_metrics['sigma_mean']:.2e} | "
                      f"T: {val_metrics['T_mean']:.1f}K")
        
        # 保存最佳模型
        if val_metrics["data_loss"] < best_val_loss:
            best_val_loss = val_metrics["data_loss"]
            checkpoints.save_checkpoint(
                output_dir,
                state,
                step=epoch,
                prefix="best_",
                keep=1,
            )
            print(f"  ✓ 保存最佳模型 (Val Loss: {best_val_loss:.6f})")
        
        # 定期保存
        if epoch % config.save_every == 0:
            checkpoints.save_checkpoint(
                output_dir,
                state,
                step=epoch,
                prefix="checkpoint_",
                keep=5,
            )
    
    print("-" * 60)
    print("训练完成!")
    print(f"最佳验证损失：{best_val_loss:.6f}")
    print(f"模型保存到：{output_dir}")


# ============================================================================
# 命令行接口
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="训练物理嵌入 TransformerPayne")
    
    # 数据
    parser.add_argument("--data_dir", type=str, default="/path/to/phoenix")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--physics_weight", type=float, default=0.1)
    
    # 模型
    parser.add_argument("--use_physics", action="store_true", default=True)
    parser.add_argument("--n_depth_layers", type=int, default=64)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        physics_weight=args.physics_weight,
        use_physics=args.use_physics,
        n_depth_layers=args.n_depth_layers,
    )
    
    train_model(config)
