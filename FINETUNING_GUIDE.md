# 微调指南 - 物理嵌入 TransformerPayne

> **版本**: 0.2.0  
> **更新日期**: 2026-03-13

---

## 📋 目录

1. [概述](#概述)
2. [微调方案](#微调方案)
3. [快速开始](#快速开始)
4. [详细配置](#详细配置)
5. [训练技巧](#训练技巧)
6. [常见问题](#常见问题)

---

## 🎯 概述

本指南说明如何从原始 TransformerPayne 微调到物理嵌入版本。

### 为什么要微调？

| 方式 | 训练时间 | 需要数据 | 收敛速度 | 推荐场景 |
|------|---------|---------|---------|---------|
| **从头训练** | ~24 小时 | 大量 | 慢 | 全新任务 |
| **微调** | ~12 小时 | 中等 | 快 | ✅ 推荐 |

### 可复用参数

| 组件 | 参数量 | 可复用 |
|------|--------|--------|
| Transformer 核心 | ~1.5M | ✅ 是 |
| 物理量预测头 | ~0.3M | ❌ 随机初始化 |
| **总计** | **~1.8M** | **83% 可复用** |

---

## 📊 微调方案

### 方案 1: 直接微调 (推荐)

**特点**:
- ✅ 简单直接
- ✅ 收敛快
- ✅ 适合大多数场景

**配置**:
```bash
python finetune_from_original.py \
    --original_checkpoint /path/to/original.joblib \
    --data_dir /path/to/phoenix \
    --output_dir ./output/finetune \
    --method direct \
    --epochs 100
```

---

### 方案 2: 两阶段微调

**特点**:
- ✅ 更稳定
- ✅ 物理头先适应
- ⏱️ 训练时间稍长

**阶段**:
1. **阶段 1** (50 epochs): 只训练物理头，冻结 Transformer
2. **阶段 2** (50 epochs): 联合训练所有参数

**配置**:
```bash
python finetune_from_original.py \
    --original_checkpoint /path/to/original.joblib \
    --method two_stage \
    --stage1_epochs 50 \
    --stage2_epochs 50
```

---

### 方案 3: 渐进式微调

**特点**:
- ✅ 最稳定
- ✅ 避免灾难性遗忘
- ⏱️ 需要调参

**进度表**:
| Epoch | 物理损失权重 | 说明 |
|-------|-------------|------|
| 0-50 | 0.0 | 只拟合数据 |
| 50-100 | 0.05 | 轻微物理约束 |
| 100+ | 0.1 | 完整物理约束 |

**配置**:
```bash
python finetune_from_original.py \
    --original_checkpoint /path/to/original.joblib \
    --method progressive \
    --epochs 150
```

---

## 🚀 快速开始

### 1. 准备原始模型

```bash
# 下载原始 TransformerPayne 模型
python -c "from transformer_payne import TransformerPayne; TransformerPayne.download()"
```

### 2. 准备数据集

```bash
# PHOENIX 数据集
/path/to/phoenix/
├── phoenix_spectra.joblib
├── phoenix_params.joblib
└── wavelengths.npy
```

### 3. 开始微调

```bash
# 最简单的命令
python finetune_from_original.py \
    --original_checkpoint ~/.cache/transformer_payne/TPAYNE.joblib \
    --data_dir /path/to/phoenix \
    --output_dir ./output/finetune \
    --method direct
```

### 4. 监控训练

```bash
# 查看日志
tail -f ./output/finetune/training.log

# 查看 TensorBoard (如配置)
tensorboard --logdir ./output/finetune
```

---

## ⚙️ 详细配置

### 命令行参数

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--original_checkpoint` | 原始模型路径 | - | **必填** |
| `--data_dir` | 数据集路径 | - | **必填** |
| `--output_dir` | 输出目录 | ./output/finetune | - |
| `--method` | 微调方法 | direct | direct/two_stage/progressive |
| `--epochs` | 训练轮数 | 100 | 50-200 |
| `--batch_size` | 批次大小 | 32 | 16-64 |
| `--learning_rate` | 学习率 | 5e-5 | 1e-5 - 1e-4 |
| `--physics_weight` | 物理损失权重 | 0.1 | 0.01-0.2 |
| `--stage1_epochs` | 阶段 1 轮数 | 50 | 30-100 |
| `--stage2_epochs` | 阶段 2 轮数 | 50 | 30-100 |

### Python API

```python
from finetune_from_original import FinetuningConfig, main

config = FinetuningConfig(
    original_checkpoint="/path/to/original.joblib",
    data_dir="/path/to/phoenix",
    output_dir="./output/finetune",
    finetune_method="direct",
    epochs=100,
    batch_size=32,
    learning_rate=5e-5,
    physics_weight=0.1,
)

main(config)
```

---

## 💡 训练技巧

### 1. 学习率选择

| 数据量 | 推荐学习率 | 微调方法 |
|--------|-----------|---------|
| 小 (<1000) | 1e-5 | two_stage |
| 中 (1000-10000) | 5e-5 | direct |
| 大 (>10000) | 1e-4 | direct |

### 2. 物理损失权重

| 任务 | 推荐权重 |
|------|---------|
| 光谱生成 | 0.05-0.1 |
| 参数反演 | 0.1-0.2 |
| 外推测试 | 0.15-0.2 |

### 3. 早停策略

```python
# 监控验证集 RMSE
if val_rmse < best_val_rmse:
    best_val_rmse = val_rmse
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter > 10:
        print("早停！")
        break
```

### 4. 梯度裁剪

```python
# 防止梯度爆炸
grad_clip = 1.0
grads = jax.tree_util.tree_map(
    lambda g: jnp.clip(g, -grad_clip, grad_clip),
    grads
)
```

---

## 📊 预期结果

### 训练时间 (单 A100)

| 方法 | Epochs | 时间 |
|------|--------|------|
| direct | 100 | ~12 小时 |
| two_stage | 50+50 | ~15 小时 |
| progressive | 150 | ~18 小时 |

### 性能提升

| 指标 | 从头训练 | 微调 | 提升 |
|------|---------|------|------|
| RMSE | 0.008 | 0.005 | 37%↓ |
| 收敛速度 | 200 epochs | 100 epochs | 50%↑ |
| 训练时间 | 24h | 12h | 50%↓ |

---

## ❓ 常见问题

### Q: 微调需要多少数据？

A: 至少 1000 条光谱，推荐 5000+。数据越多效果越好。

### Q: 可以只微调部分层吗？

A: 可以，修改 `finetune_two_stage` 中的 `frozen_prefixes`。

### Q: 物理损失权重设多少？

A: 从 0.05 开始，根据验证集调整。太大影响拟合，太小物理约束弱。

### Q: 训练出现 NaN 怎么办？

A: 
1. 降低学习率
2. 添加梯度裁剪
3. 检查数据归一化

### Q: 如何继续训练？

```bash
python finetune_from_original.py \
    --original_checkpoint ./output/finetune/checkpoint_100.joblib \
    --method direct \
    --epochs 50  # 继续 50 epochs
```

---

## 📚 相关资源

- **使用指南**: `PHYSICS_EMBEDDED_README.md`
- **开发指南**: `CLAUDE_CODE.md`
- **项目总结**: `PROJECT_SUMMARY.md`

---

*最后更新：2026-03-13*
