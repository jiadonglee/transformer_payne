# 物理嵌入 TransformerPayne - 使用指南

> **版本**: 0.1.0  
> **创建日期**: 2026-03-12  
> **基于**: PhysFormer (arXiv:2603.01459)

---

## 📋 目录

1. [快速开始](#快速开始)
2. [模型架构](#模型架构)
3. [训练流程](#训练流程)
4. [推理使用](#推理使用)
5. [API 参考](#api 参考)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 安装依赖

```bash
cd /Users/jordan/.openclaw/workspace/projects/transformer_payne
source .venv/bin/activate

# 确保已安装 JAX 和 Flax
pip install jax jaxlib flax optax
```

### 基础使用

```python
import jax.numpy as jnp
from transformer_payne_physics import create_physics_embedded_model

# 创建模型
model, params = create_physics_embedded_model(
    use_physics=True,
    n_depth_layers=64,
)

# 准备输入
log_wavelengths = jnp.log10(jnp.linspace(4670, 4960, 100))
stellar_params = jnp.array([5777, 4.44, 0.0, 0.0])  # 太阳参数

# 前向传播 (带物理量输出)
from flax.core.frozen_dict import freeze
variables = {"params": freeze(params)}

output = model.apply(
    variables,
    (log_wavelengths, stellar_params),
    train=False,
    return_physics_outputs=True,
)

spectrum, physics_outputs = output

print(f"光谱形状：{spectrum.shape}")
print(f"吸收系数范围：[{physics_outputs['kappa'].min():.2e}, {physics_outputs['kappa'].max():.2e}]")
print(f"温度范围：[{physics_outputs['T'].min():.1f}, {physics_outputs['T'].max():.1f}] K")
```

---

## 🏗️ 模型架构

### 原始 vs 物理嵌入

| 组件 | 原始 TransformerPayne | 物理嵌入版本 |
|------|---------------------|-------------|
| 输入 | (logλ, 恒星参数) | 相同 |
| Transformer 层 | 8 层 | 相同 |
| 输出头 | 直接预测光谱 | 预测物理量 (κ, σ, T) |
| 后处理 | 简单变换 | RTE 求解 + 角度积分 |
| 输出 | (I, continuum) | (I, F_λ) + 物理量 |

### 物理量说明

| 物理量 | 符号 | 单位 | 范围 |
|--------|------|------|------|
| 吸收系数 | κ | cm⁻¹ | ~1e-20 |
| 散射系数 | σ | cm⁻¹ | ~1e-22 |
| 温度 | T | K | 2000-12000 |
| 光学深度 | τ | 无量纲 | 0-10 |

---

## 📊 训练流程

### 三阶段训练

#### 阶段 1: 自编码器预训练 (可选)

```bash
python train_physics_embedded.py \
    --data_dir /path/to/phoenix \
    --output_dir ./output/ae_pretrain \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-4
```

#### 阶段 2: 参数到光谱映射

```bash
python train_physics_embedded.py \
    --data_dir /path/to/phoenix \
    --output_dir ./output/p2f \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --physics_weight 0.05
```

#### 阶段 3: 联合优化

```bash
python train_physics_embedded.py \
    --data_dir /path/to/phoenix \
    --output_dir ./output/joint \
    --epochs 200 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --physics_weight 0.1
```

### 训练配置选项

```bash
# 完整选项
python train_physics_embedded.py \
    --data_dir /path/to/phoenix \
    --output_dir ./output \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --physics_weight 0.1 \
    --n_depth_layers 64 \
    --use_physics
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | PHOENIX 数据集路径 | - |
| `--output_dir` | 输出目录 | ./output |
| `--epochs` | 训练轮数 | 200 |
| `--batch_size` | 批次大小 | 32 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--physics_weight` | 物理损失权重 | 0.1 |
| `--n_depth_layers` | 深度层数 | 64 |

---

## 🔍 推理使用

### 加载训练好的模型

```python
from flax.training import checkpoints
from transformer_payne_physics import create_physics_embedded_model

# 创建模型
model, _ = create_physics_embedded_model(use_physics=True)

# 加载检查点
state = checkpoints.restore_checkpoint("./output", target=None)
params = state["params"]

# 推理
import jax.numpy as jnp
from flax.core.frozen_dict import freeze

log_wavelengths = jnp.log10(jnp.linspace(4670, 4960, 100))
stellar_params = jnp.array([5777, 4.44, 0.0, 0.0])

variables = {"params": freeze(params)}
spectrum, physics = model.apply(
    variables,
    (log_wavelengths, stellar_params),
    train=False,
    return_physics_outputs=True,
)
```

### 批量推理

```python
from jax import vmap

# 批量参数
stellar_params_batch = jnp.array([
    [5777, 4.44, 0.0, 0.0],   # 太阳
    [6000, 4.5, -0.5, 0.2],  # 另一颗星
    [4500, 4.0, -1.0, 0.4],  # 第三颗星
])

# 使用 vmap 批量处理
batch_apply = vmap(
    lambda p: model.apply(variables, (log_wavelengths, p), train=False),
    in_axes=0,
    out_axes=0,
)

spectra_batch = batch_apply(stellar_params_batch)
print(f"批量输出形状：{spectra_batch.shape}")  # (3, 100, 2)
```

---

## 📖 API 参考

### `TransformerPayneModelPhysics`

主模型类。

```python
model = TransformerPayneModelPhysics(
    dim=256,              # 隐藏层维度
    no_layers=8,          # Transformer 层数
    use_physics=True,     # 是否使用物理嵌入
    n_depth_layers=64,    # 深度层数
    n_angles=20,          # 角度积分点数
)
```

### `PhysicalQuantitiesHead`

物理量预测头。

```python
phys_head = PhysicalQuantitiesHead(
    dim=256,
    n_wavelengths=9875,
    n_depth_layers=64,
)

outputs = phys_head(transformer_output, train=False)
# outputs: {"kappa": ..., "sigma": ..., "T": ...}
```

### `RadiativeTransferSolver`

辐射转移求解器。

```python
solver = RadiativeTransferSolver(
    n_depth_layers=64,
    n_angles=20,
)

I_nu, F_nu = solver(tau, T, nu)
```

### `combined_loss`

组合损失函数。

```python
total_loss, breakdown = combined_loss(
    spectrum_pred,
    spectrum_target,
    physics_outputs=physics,
    stellar_params=params,
    wavelengths=wavelengths,
    physics_weight=0.1,
)
```

---

## ❓ 常见问题

### Q: 物理嵌入模式比原始模式慢多少？

A: 大约慢 20-30%，主要来自 RTE 求解和角度积分。但可以通过 JAX JIT 编译优化。

### Q: 如何禁用物理嵌入，使用原始模式？

```python
model = TransformerPayneModelPhysics(
    use_physics=False,  # 禁用物理嵌入
)
```

### Q: 物理损失权重应该设多少？

A: 建议从 0.01 开始，逐渐增加到 0.1。太大可能影响数据拟合。

### Q: 训练时出现 NaN 怎么办？

A: 检查以下几点：
1. 学习率是否太大
2. 物理量是否超出合理范围
3. 尝试 gradient clipping

### Q: 如何查看物理量的中间输出？

```python
output = model.apply(
    variables,
    inputs,
    train=False,
    return_physics_outputs=True,  # 设为 True
)
spectrum, physics = output
print(physics.keys())  # dict_keys(['kappa', 'sigma', 'T', 'tau', 'I_nu'])
```

---

## 📚 相关资源

- **PhysFormer 论文**: arXiv:2603.01459
- **TransformerPayne 原论文**: arXiv:2306.15703
- **JAX 文档**: https://jax.readthedocs.io
- **Flax 文档**: https://flax.readthedocs.io

---

## 📞 支持

如有问题，请提交 Issue 或联系开发团队。

---

*文档更新日期：2026-03-12*
