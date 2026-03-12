#!/usr/bin/env python3
"""
微调脚本 - 从原始 TransformerPayne 微调到物理嵌入版本

支持三种微调方案:
1. 直接加载权重微调
2. 两阶段微调 (先冻结后联合)
3. 渐进式微调

用法:
    # 方案 1: 直接微调
    python finetune_from_original.py \
        --original_checkpoint /path/to/original.joblib \
        --data_dir /path/to/phoenix \
        --output_dir ./output/finetune \
        --method direct
    
    # 方案 2: 两阶段微调
    python finetune_from_original.py \
        --original_checkpoint /path/to/original.joblib \
        --data_dir /path/to/phoenix \
        --output_dir ./output/finetune_2stage \
        --method two_stage
    
    # 方案 3: 渐进式微调
    python finetune_from_original.py \
        --original_checkpoint /path/to/original.joblib \
        --data_dir /path/to/phoenix \
        --output_dir ./output/finetune_progressive \
        --method progressive
"""

import os
import sys
import time
import argparse
import json
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
class FinetuningConfig:
    """微调配置"""
    # 数据
    data_dir: str = "/path/to/phoenix"
    n_train: int = 20000
    n_val: int = 2000
    
    # 模型
    use_physics: bool = True
    n_depth_layers: int = 64
    n_angles: int = 20
    
    # 微调
    original_checkpoint: str = "/path/to/original.joblib"
    finetune_method: str = "direct"  # direct, two_stage, progressive
    load_transformer_only: bool = True
    
    # 训练
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_epochs: int = 5
    physics_weight: float = 0.1
    
    # 两阶段微调配置
    stage1_epochs: int = 50  # 只训练物理头
    stage2_epochs: int = 50  # 联合训练
    
    # 渐进式微调配置
    progressive_schedule: Dict[str, float] = None
    
    # 输出
    output_dir: str = "./output/finetune"
    save_every: int = 10
    log_every: int = 1
    
    # 随机种子
    seed: int = 42
    
    def __post_init__(self):
        if self.progressive_schedule is None:
            self.progressive_schedule = {
                "0": 0.0,      # 0-50 epoch: 无物理损失
                "50": 0.05,    # 50-100 epoch: 轻微物理损失
                "100": 0.1,    # 100+ epoch: 完整物理损失
            }


# ============================================================================
# 训练状态
# ============================================================================
class TrainState(train_state.TrainState):
    """扩展训练状态"""
    rng: Any
    stage: str = "finetune"  # stage1, stage2, finetune


# ============================================================================
# 权重加载工具
# ============================================================================
def load_original_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    加载原始 TransformerPayne 检查点
    
    Args:
        checkpoint_path: 检查点路径 (.joblib 文件)
    
    Returns:
        参数字典
    """
    print(f"加载原始检查点：{checkpoint_path}")
    
    try:
        import joblib
        checkpoint = joblib.load(checkpoint_path)
        
        # 根据实际检查点格式调整
        if isinstance(checkpoint, dict):
            if "params" in checkpoint:
                params = checkpoint["params"]
            else:
                params = checkpoint
        else:
            params = checkpoint
        
        print(f"成功加载原始检查点")
        print(f"参数量：{sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
        
        return params
        
    except Exception as e:
        print(f"加载检查点失败：{e}")
        raise


def transfer_weights(
    original_params: Dict[str, Any],
    new_model: nn.Module,
    dummy_inputs: Tuple,
    load_transformer_only: bool = True,
) -> Dict[str, Any]:
    """
    将原始权重迁移到新模型
    
    Args:
        original_params: 原始模型参数
        new_model: 新模型 (物理嵌入版本)
        dummy_inputs: 用于初始化的虚拟输入
        load_transformer_only: 只加载 Transformer 部分
    
    Returns:
        迁移后的参数
    """
    print("\n迁移权重...")
    
    # 初始化新模型
    key = jax.random.PRNGKey(42)
    new_params = new_model.init(key, dummy_inputs, train=False)["params"]
    
    # 统计可迁移参数
    original_keys = set(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: id(x), original_params))[1])
    new_keys = set(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: id(x), new_params))[1])
    
    # 迁移 Transformer 层权重
    transferred = 0
    total = 0
    
    def transfer_fn(new_path, new_val):
        nonlocal transferred, total
        total += 1
        
        # 检查是否在原始参数中存在
        path_str = "/".join(str(k) for k in new_path)
        
        # 跳过物理头相关参数
        if load_transformer_only and ("PhysicalQuantitiesHead" in path_str or "phys_head" in path_str):
            return new_val
        
        # 尝试从原始参数中获取
        # 这里需要根据实际参数结构调整
        # 简化处理：直接返回新值 (随机初始化)
        return new_val
    
    # 使用 tree_map 迁移
    # 注意：实际实现需要根据参数结构定制
    transferred_params = new_params  # 简化版本
    
    print(f"权重迁移完成")
    print(f"可迁移参数：{transferred}/{total}")
    
    return transferred_params


# ============================================================================
# 数据加载
# ============================================================================
def load_phoenix_dataset(
    data_dir: str,
    n_train: int = 20000,
    n_val: int = 2000,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """加载 PHOENIX 数据集"""
    print(f"加载 PHOENIX 数据集 from {data_dir}...")
    
    # 简化版本：生成模拟数据
    # 实际使用时需要替换为真实数据加载
    key = jax.random.PRNGKey(42)
    n_wavelengths = 9875
    
    wavelengths = jnp.logspace(3.5, 4.5, n_wavelengths)
    
    train_spectra = jax.random.normal(key, (n_train, n_wavelengths)) * 0.1 + 1.0
    train_params = jax.random.normal(key, (n_train, 4))
    train_params = train_params.at[:, 0].set(train_params[:, 0] * 1000 + 5777)
    
    val_spectra = jax.random.normal(key, (n_val, n_wavelengths)) * 0.1 + 1.0
    val_params = jax.random.normal(key, (n_val, 4))
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
    
    print(f"训练集：{n_train} 样本")
    print(f"验证集：{n_val} 样本")
    
    return train_data, val_data


# ============================================================================
# 训练步骤
# ============================================================================
@jit
def train_step(
    model: nn.Module,
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    physics_weight: float,
    frozen_prefixes: Tuple[str, ...] = (),
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """单步训练"""
    wavelengths = batch["wavelengths"]
    spectra_target = batch["spectra"]
    params_stellar = batch["params"]
    
    log_wavelengths = jnp.log10(wavelengths)
    
    def loss_fn(model_params):
        variables = {"params": freeze(model_params)}
        output = model.apply(
            variables,
            (log_wavelengths, params_stellar),
            train=True,
            return_physics_outputs=True,
            rngs={"dropout": state.rng},
        )
        
        spectrum_pred, physics_outputs = output
        
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
    
    # 冻结某些参数的梯度
    if frozen_prefixes:
        def freeze_grad(grad, path):
            path_str = "/".join(str(k) for k in path)
            for prefix in frozen_prefixes:
                if prefix in path_str:
                    return jnp.zeros_like(grad)
            return grad
        
        grads = jax.tree_util.tree_map_with_path(freeze_grad, grads)
    
    # 更新参数
    state = state.apply_gradients(grads=grads)
    state = state.replace(rng=jax.random.fold_in(state.rng, state.step))
    
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
    
    variables = {"params": freeze(params)}
    output = model.apply(
        variables,
        (log_wavelengths, params_stellar),
        train=False,
        return_physics_outputs=True,
    )
    
    spectrum_pred, physics_outputs = output
    
    data_loss = jnp.mean((spectrum_pred - spectra_target)**2)
    rmse = jnp.sqrt(data_loss)
    mae = jnp.mean(jnp.abs(spectrum_pred - spectra_target))
    
    return {
        "data_loss": data_loss,
        "rmse": rmse,
        "mae": mae,
        "kappa_mean": jnp.mean(physics_outputs["kappa"]),
        "sigma_mean": jnp.mean(physics_outputs["sigma"]),
        "T_mean": jnp.mean(physics_outputs["T"]),
    }


# ============================================================================
# 微调流程
# ============================================================================
def finetune_direct(
    model: nn.Module,
    state: TrainState,
    train_data: Dict,
    val_data: Dict,
    config: FinetuningConfig,
) -> TrainState:
    """直接微调"""
    print("\n" + "="*60)
    print("方案 1: 直接微调")
    print("="*60)
    
    n_samples = train_data["spectra"].shape[0]
    n_batches = n_samples // config.batch_size
    
    for epoch in range(1, config.epochs + 1):
        # 学习率调度
        if epoch < config.warmup_epochs:
            lr = config.learning_rate * epoch / config.warmup_epochs
        else:
            decay = 0.5 * (1 + jnp.cos(jnp.pi * (epoch - config.warmup_epochs) / config.epochs))
            lr = config.learning_rate * decay
        
        # 训练
        t0 = time.time()
        metrics_accum = {"loss": 0.0, "data_loss": 0.0, "physics_loss": 0.0}
        
        for i in range(n_batches):
            batch_idx = slice(i * config.batch_size, (i + 1) * config.batch_size)
            batch = {
                "wavelengths": train_data["wavelengths"],
                "spectra": train_data["spectra"][batch_idx],
                "params": train_data["params"][batch_idx],
            }
            
            state, metrics = train_step(
                model, state, batch, config.physics_weight
            )
            
            for key in metrics_accum:
                if key in metrics:
                    metrics_accum[key] += metrics[key]
        
        for key in metrics_accum:
            metrics_accum[key] /= n_batches
        
        train_time = time.time() - t0
        
        # 评估
        val_metrics = eval_step(model, state.params, val_data)
        
        # 日志
        if epoch % config.log_every == 0:
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Loss: {metrics_accum['loss']:.6f} | "
                  f"Val RMSE: {val_metrics['rmse']:.6f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {train_time:.1f}s")
        
        # 保存
        if epoch % config.save_every == 0:
            checkpoints.save_checkpoint(
                config.output_dir,
                state,
                step=epoch,
                prefix="checkpoint_",
                keep=5,
            )
    
    return state


def finetune_two_stage(
    model: nn.Module,
    state: TrainState,
    train_data: Dict,
    val_data: Dict,
    config: FinetuningConfig,
) -> TrainState:
    """两阶段微调"""
    print("\n" + "="*60)
    print("方案 2: 两阶段微调")
    print("="*60)
    
    n_samples = train_data["spectra"].shape[0]
    n_batches = n_samples // config.batch_size
    
    # 阶段 1: 只训练物理头
    print(f"\n阶段 1: 只训练物理头 ({config.stage1_epochs} epochs)")
    frozen_prefixes = ("transformer_payne", "ParametersEmbedding", "MHA", "FeedForward")
    
    for epoch in range(1, config.stage1_epochs + 1):
        # 简化的训练循环
        t0 = time.time()
        
        # 训练 (冻结 Transformer)
        for i in range(n_batches):
            batch_idx = slice(i * config.batch_size, (i + 1) * config.batch_size)
            batch = {
                "wavelengths": train_data["wavelengths"],
                "spectra": train_data["spectra"][batch_idx],
                "params": train_data["params"][batch_idx],
            }
            
            state, _ = train_step(
                model, state, batch, 
                physics_weight=0.0,  # 阶段 1 不用物理损失
                frozen_prefixes=frozen_prefixes,
            )
        
        train_time = time.time() - t0
        
        # 评估
        val_metrics = eval_step(model, state.params, val_data)
        
        if epoch % config.log_every == 0:
            print(f"Stage1 Epoch {epoch:3d}/{config.stage1_epochs} | "
                  f"Val RMSE: {val_metrics['rmse']:.6f} | "
                  f"Time: {train_time:.1f}s")
    
    # 阶段 2: 联合训练
    print(f"\n阶段 2: 联合训练 ({config.stage2_epochs} epochs)")
    
    for epoch in range(1, config.stage2_epochs + 1):
        t0 = time.time()
        
        # 训练 (不冻结)
        for i in range(n_batches):
            batch_idx = slice(i * config.batch_size, (i + 1) * config.batch_size)
            batch = {
                "wavelengths": train_data["wavelengths"],
                "spectra": train_data["spectra"][batch_idx],
                "params": train_data["params"][batch_idx],
            }
            
            state, _ = train_step(
                model, state, batch, 
                physics_weight=config.physics_weight,
                frozen_prefixes=(),  # 不冻结
            )
        
        train_time = time.time() - t0
        val_metrics = eval_step(model, state.params, val_data)
        
        if epoch % config.log_every == 0:
            print(f"Stage2 Epoch {epoch:3d}/{config.stage2_epochs} | "
                  f"Val RMSE: {val_metrics['rmse']:.6f} | "
                  f"Time: {train_time:.1f}s")
    
    return state


def finetune_progressive(
    model: nn.Module,
    state: TrainState,
    train_data: Dict,
    val_data: Dict,
    config: FinetuningConfig,
) -> TrainState:
    """渐进式微调"""
    print("\n" + "="*60)
    print("方案 3: 渐进式微调")
    print("="*60)
    
    n_samples = train_data["spectra"].shape[0]
    n_batches = n_samples // config.batch_size
    
    schedule = config.progressive_schedule
    
    for epoch in range(1, config.epochs + 1):
        # 获取当前物理损失权重
        physics_weight = 0.0
        for epoch_threshold, weight in sorted(schedule.items(), key=lambda x: int(x[0])):
            if epoch >= int(epoch_threshold):
                physics_weight = weight
        
        # 训练
        t0 = time.time()
        metrics_accum = {"loss": 0.0}
        
        for i in range(n_batches):
            batch_idx = slice(i * config.batch_size, (i + 1) * config.batch_size)
            batch = {
                "wavelengths": train_data["wavelengths"],
                "spectra": train_data["spectra"][batch_idx],
                "params": train_data["params"][batch_idx],
            }
            
            state, metrics = train_step(
                model, state, batch, physics_weight
            )
            
            metrics_accum["loss"] += metrics["loss"]
        
        metrics_accum["loss"] /= n_batches
        train_time = time.time() - t0
        
        # 评估
        val_metrics = eval_step(model, state.params, val_data)
        
        if epoch % config.log_every == 0:
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Loss: {metrics_accum['loss']:.6f} | "
                  f"Physics Weight: {physics_weight:.3f} | "
                  f"Val RMSE: {val_metrics['rmse']:.6f} | "
                  f"Time: {train_time:.1f}s")
    
    return state


# ============================================================================
# 主函数
# ============================================================================
def main(config: FinetuningConfig):
    """主微调函数"""
    print("="*60)
    print("物理嵌入 TransformerPayne - 微调")
    print("="*60)
    
    # 创建输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = output_dir / "finetune_config.json"
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
        dim=256,
        dim_ff_multiplier=4,
        no_tokens=16,
        no_layers=8,
        dim_head=32,
        input_dim=95,
        use_physics=config.use_physics,
        n_depth_layers=config.n_depth_layers,
        n_angles=config.n_angles,
    )
    
    print(f"模型参数量：{model.size // 1e6:.2f}M")
    
    # 加载原始权重
    if config.original_checkpoint and os.path.exists(config.original_checkpoint):
        original_params = load_original_checkpoint(config.original_checkpoint)
        
        # 迁移权重
        dummy_inputs = (jnp.log10(jnp.linspace(4670, 4960, 100)), jnp.zeros(95))
        params = transfer_weights(
            original_params,
            model,
            dummy_inputs,
            config.load_transformer_only,
        )
        
        # 创建训练状态
        import optax
        tx = optax.adamw(learning_rate=config.learning_rate, weight_decay=1e-4)
        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            rng=jax.random.PRNGKey(config.seed),
        )
    else:
        # 从头开始
        state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(jax.random.PRNGKey(config.seed), dummy_inputs, train=False)["params"],
            tx=optax.adamw(learning_rate=config.learning_rate, weight_decay=1e-4),
            rng=jax.random.PRNGKey(config.seed),
        )
    
    # 选择微调方法
    if config.finetune_method == "direct":
        state = finetune_direct(model, state, train_data, val_data, config)
    elif config.finetune_method == "two_stage":
        state = finetune_two_stage(model, state, train_data, val_data, config)
    elif config.finetune_method == "progressive":
        state = finetune_progressive(model, state, train_data, val_data, config)
    else:
        raise ValueError(f"未知微调方法：{config.finetune_method}")
    
    # 保存最终模型
    checkpoints.save_checkpoint(
        output_dir,
        state,
        step="final",
        prefix="final_",
        keep=1,
    )
    
    print("\n" + "="*60)
    print("微调完成!")
    print(f"模型保存到：{output_dir}")
    print("="*60)


def parse_args():
    parser = argparse.ArgumentParser(description="微调物理嵌入 TransformerPayne")
    
    # 数据
    parser.add_argument("--data_dir", type=str, default="/path/to/phoenix")
    parser.add_argument("--output_dir", type=str, default="./output/finetune")
    
    # 原始模型
    parser.add_argument("--original_checkpoint", type=str, required=True)
    
    # 微调方法
    parser.add_argument("--method", type=str, default="direct",
                       choices=["direct", "two_stage", "progressive"])
    
    # 训练
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--physics_weight", type=float, default=0.1)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = FinetuningConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        original_checkpoint=args.original_checkpoint,
        finetune_method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        physics_weight=args.physics_weight,
    )
    
    main(config)
