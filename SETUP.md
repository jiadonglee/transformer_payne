# TransformerPayne 设置指南

## ✅ 安装完成

项目已成功安装并测试通过！

## 📁 项目位置

```
/Users/jordan/.openclaw/workspace/projects/transformer_payne
```

## 🚀 快速开始

### 1. 激活虚拟环境
```bash
cd /Users/jordan/.openclaw/workspace/projects/transformer_payne
source .venv/bin/activate
```

### 2. 运行测试
```bash
python -m pytest src/tests/ -v
```

### 3. 运行示例
```bash
python test_run.py
```

### 4. 使用 Python 代码
```python
import jax
jax.config.update("jax_enable_x64", True)

import transformer_payne as tp
import numpy as np

# 下载模型
emulator = tp.TransformerPayne.download()

# 生成光谱
wave = np.linspace(4670, 4960, 2000)
mu = 1.0
parameters = emulator.solar_parameters
spectrum = emulator(np.log10(wave), mu, parameters)
```

## 📊 项目信息

**TransformerPayne** 是一个基于 Transformer 的恒星光谱模拟器：

- **用途**: 模拟恒星光谱 across broad parameter space
- **参数范围**:
  - Teff: 4,000 - 8,000 K
  - logg: 2.0 - 5.0
  - [Fe/H]: -2.5 to 1.0
  - 分辨率：~300,000
  - 波长：1,500 - 20,000 Å

## 📚 教程 Notebooks

在 `tutorial/` 目录下有详细的教程：

- `transformer_payne.ipynb` - 基础使用
- `blackbody_flux.ipynb` - 黑体辐射
- `defining_abundances.ipynb` - 定义丰度
- `fitting_metallicity_and_alpha_elements.ipynb` - 拟合金属丰度
- `intensity_flux_luminosity.ipynb` - 强度、流量、光度

## 🧪 测试结果

```
11 passed in 14.15s
```

所有测试通过！✅

## 📦 已安装的依赖

- jax 0.7.1
- jaxlib 0.7.1
- numpy 1.26.4
- flax 0.10.6
- matplotlib 3.10.8
- huggingface-hub 1.6.0
- transformer-payne 0.10 (本地安装)

## 🔗 相关资源

- GitHub: https://github.com/jiadonglee/transformer_payne
- 论文：https://arxiv.org/abs/2306.15703

---

设置日期：2026-03-12
