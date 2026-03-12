# 物理公式融合到 TransformerPayne 计划

> 基于论文：**PhysFormer: A Physics-Embedded Generative Model for Physically Self-Consistent Spectral Synthesis**  
> arXiv:2603.01459 | 2026 年 3 月

---

## 📋 执行摘要

本计划旨在将物理约束和物理公式嵌入到 TransformerPayne 恒星光谱模拟模型中，使其生成**物理自洽**的光谱预测。当前 TransformerPayne 是纯数据驱动模型，通过融合物理知识可提高外推能力、物理一致性和可解释性。

---

## 🎯 目标

1. **物理一致性**：确保预测光谱满足基本物理定律（如辐射转移方程、能量守恒）
2. **提高外推能力**：在训练数据范围外仍能生成合理光谱
3. **保持精度**：在训练分布内保持或提高现有精度
4. **可解释性**：模型决策过程可追溯到物理原理

---

## 📐 可融合的物理公式

### 1. 辐射转移方程 (Radiative Transfer Equation)

$$\frac{dI_\nu}{ds} = -\kappa_\nu \rho I_\nu + j_\nu \rho$$

或光学深度形式：

$$\frac{dI_\nu}{d\tau_\nu} = I_\nu - S_\nu$$

其中：
- $I_\nu$ = 比强度
- $\kappa_\nu$ = 吸收系数
- $j_\nu$ = 发射系数
- $S_\nu = j_\nu/\kappa_\nu$ = 源函数
- $\tau_\nu$ = 光学深度

**融合方式**：
- 作为软约束加入损失函数
- 作为网络架构的物理层

---

### 2. 局部热动平衡 (LTE) 条件

$$\frac{n_u}{n_l} = \frac{g_u}{g_l} \exp\left(-\frac{h\nu}{k_B T}\right)$$

**融合方式**：
- 约束能级布居数预测
- 作为源函数计算的物理基础

---

### 3. Saha 电离方程

$$\frac{N_{i+1}N_e}{N_i} = \frac{2}{\Lambda^3}\frac{g_{i+1}}{g_i}\exp\left(-\frac{\chi_i}{k_B T}\right)$$

其中 $\Lambda = \sqrt{\frac{h^2}{2\pi m_e k_B T}}$ 是热德布罗意波长

**融合方式**：
- 约束电离态预测
- 作为输入参数的物理变换层

---

### 4. 玻尔兹曼激发方程

$$\frac{n_j}{n} = \frac{g_j}{U(T)}\exp\left(-\frac{E_j}{k_B T}\right)$$

**融合方式**：
- 约束激发态布居
- 作为谱线强度计算的中间层

---

### 5. 谱线轮廓函数

**多普勒展宽**：
$$\phi_D(\nu) = \frac{1}{\Delta\nu_D\sqrt{\pi}}\exp\left[-\left(\frac{\nu-\nu_0}{\Delta\nu_D}\right)^2\right]$$

**洛伦兹展宽**：
$$\phi_L(\nu) = \frac{1}{\pi}\frac{\gamma/2}{(\nu-\nu_0)^2 + (\gamma/2)^2}$$

**Voigt 轮廓**（卷积）：
$$\phi_V(\nu) = \int_{-\infty}^{\infty} \phi_D(\nu')\phi_L(\nu-\nu')d\nu'$$

**融合方式**：
- 作为谱线合成的可微分层
- 约束谱线形状预测

---

### 6. 连续谱不透明度

**束缚 - 自由跃迁**：
$$\kappa_{bf} \propto \frac{1}{\nu^3} n_i \sigma_{bf}(\nu)$$

**自由 - 自由跃迁**：
$$\kappa_{ff} \propto \frac{1}{\nu^2} n_e n_i Z^2 g_{ff}$$

**融合方式**：
- 作为连续谱预测的物理约束

---

### 7. 能量守恒约束

$$\int_0^\infty I_\nu d\nu = \sigma T_{eff}^4 / \pi$$

**融合方式**：
- 作为全局损失约束
- 确保积分流量与有效温度一致

---

## 🏗️ 融合架构设计

### 方案 A: 物理嵌入损失 (Physics-Informed Loss)

```
┌─────────────────────────────────────────────────────┐
│                    TransformerPayne                  │
│  输入 → [Embedding] → [Transformer Layers] → 输出    │
└─────────────────────────────────────────────────────┘
                            ↓
         ┌──────────────────┴──────────────────┐
         ↓                                     ↓
┌─────────────────┐                 ┌─────────────────────┐
│  数据驱动损失    │                 │   物理约束损失       │
│  L_data         │                 │   L_physics         │
│  (MSE/MAE)      │                 │   (PDE 残差)         │
└─────────────────┘                 └─────────────────────┘
         ↓                                     ↓
         └──────────────────┬──────────────────┘
                            ↓
                   L_total = L_data + λ·L_physics
```

**物理损失项**：
$$\mathcal{L}_{physics} = \mathcal{L}_{RTE} + \mathcal{L}_{LTE} + \mathcal{L}_{energy} + \mathcal{L}_{smooth}$$

---

### 方案 B: 物理层嵌入 (Physics-Embedded Layers)

```
输入参数 (Teff, logg, [X/H], ...)
        ↓
┌───────────────────┐
│ 物理参数变换层     │ ← Saha/Boltzmann 方程
│ (可微分)          │
└───────────────────┘
        ↓
┌───────────────────┐
│ 谱线强度计算层     │ ← 振子强度、展宽公式
│ (基于物理公式)     │
└───────────────────┘
        ↓
┌───────────────────┐
│ Transformer 核心   │ ← 学习残差/修正
└───────────────────┘
        ↓
┌───────────────────┐
│ 辐射转移层        │ ← 数值积分 RTE
│ (可微分 RT 求解器)  │
└───────────────────┘
        ↓
输出光谱 (物理自洽)
```

---

### 方案 C: 混合方法 (推荐)

结合方案 A 和 B：
1. **前处理**：使用物理公式计算初始估计
2. **核心网络**：Transformer 学习物理模型与真实数据的残差
3. **后处理**：物理约束层确保输出自洽
4. **损失函数**：联合优化数据拟合和物理一致性

---

## 📝 实现步骤

### 阶段 1: 基础准备 (2-3 周)

- [ ] **1.1** 分析当前 TransformerPayne 架构
  - 位置：`src/transformer_payne/transformer_payne.py`
  - 理解输入/输出格式、网络结构

- [ ] **1.2** 实现物理公式的可微分版本
  - 创建 `physics_layers.py` 模块
  - 实现 Saha、Boltzmann、Voigt 等函数（使用 JAX）

- [ ] **1.3** 建立物理一致性评估指标
  - 能量守恒误差
  - 辐射转移残差
  - LTE 偏离度

### 阶段 2: 物理层实现 (3-4 周)

- [ ] **2.1** 实现 Saha-Boltzmann 层
  ```python
  class SahaBoltzmannLayer(nn.Module):
      """计算电离态和激发态布居数"""
      def __call__(self, Teff, logg, abundances):
          # 实现 Saha 方程
          # 实现 Boltzmann 方程
          return level_populations
  ```

- [ ] **2.2** 实现 Voigt 轮廓层
  ```python
  class VoigtProfileLayer(nn.Module):
      """计算谱线 Voigt 轮廓"""
      def __call__(self, wavelength, line_params):
          # 多普勒 + 洛伦兹卷积
          return line_profile
  ```

- [ ] **2.3** 实现简化辐射转移层
  ```python
  class RadiativeTransferLayer(nn.Module):
      """数值求解辐射转移方程"""
      def __call__(self, source_function, optical_depth):
          # 数值积分 RTE
          return emergent_intensity
  ```

### 阶段 3: 损失函数设计 (2-3 周)

- [ ] **3.1** 实现物理约束损失
  ```python
  def physics_loss(params, spectrum, inputs):
      # RTE 残差
      l_rte = compute_rte_residual(spectrum, params)
      # 能量守恒
      l_energy = energy_conservation_loss(spectrum, inputs['Teff'])
      # 平滑性约束
      l_smooth = spectral_smoothness_loss(spectrum)
      return l_rte + l_energy + l_smooth
  ```

- [ ] **3.2** 设计自适应权重λ
  - 动态调整物理损失权重
  - 基于训练阶段的课程学习

### 阶段 4: 模型集成与训练 (4-6 周)

- [ ] **4.1** 修改 TransformerPayne 架构
  - 插入物理层
  - 保持端到端可微分

- [ ] **4.2** 准备训练数据
  - 现有 MARCS+Korg 合成光谱
  - 可选：加入观测光谱（如太阳、标准星）

- [ ] **4.3** 训练策略
  - 阶段 1：仅数据驱动损失（预热）
  - 阶段 2：逐渐增加物理损失权重
  - 阶段 3：联合优化

### 阶段 5: 评估与验证 (2-3 周)

- [ ] **5.1** 在测试集上评估
  - 精度指标（RMSE、MAE）
  - 物理一致性指标

- [ ] **5.2** 外推能力测试
  - 在参数空间边界测试
  - 与纯数据驱动模型对比

- [ ] **5.3** 消融实验
  - 移除各物理组件的影响
  - 分析各物理约束的贡献

---

## 🛠️ 技术要点

### JAX 可微分实现

所有物理公式必须用 JAX 实现以保持可微分性：

```python
import jax
import jax.numpy as jnp
from jax import vmap, grad

@jax.jit
def saha_equation(T, n_e, chi, g_ratio):
    """可微分 Saha 方程"""
    h = 6.626e-27
    k = 1.38e-16
    m_e = 9.109e-28
    
    Lambda = h / jnp.sqrt(2 * jnp.pi * m_e * k * T)
    ratio = (2 / Lambda**3) * (g_ratio / n_e) * jnp.exp(-chi / (k * T))
    return ratio
```

### 数值稳定性

- 使用 `jax.lax.stop_gradient` 防止梯度爆炸
- 对指数函数使用 `jax.nn.softplus` 或裁剪
- 光学深度计算使用对数空间

### 性能优化

- 使用 `jax.vmap` 批量处理
- 使用 `jax.lax.scan` 替代循环
- 预计算不变的物理常数

---

## 📊 预期结果

| 指标 | 当前模型 | 目标 (物理融合) |
|------|----------|-----------------|
| 训练集 RMSE | ~1% | <1% |
| 外推 RMSE | ~10-20% | <5% |
| 能量守恒误差 | 无约束 | <0.1% |
| 物理一致性 | 无保证 | 保证 |
| 可解释性 | 低 | 中 - 高 |

---

## 📚 参考文献

1. **PhysFormer 论文**: arXiv:2603.01459
2. **TransformerPayne 原论文**: arXiv:2306.15703
3. **Korg.jl**: https://github.com/ajwheeler/Korg.jl
4. **Physics-Informed Neural Networks**: Raissi et al. (2019)
5. **Radiative Transfer in Stellar Atmospheres**: Hubeny & Mihalas (2014)

---

## 📅 时间线

```
Week 1-3:  基础准备、物理公式实现
Week 4-7:  物理层开发
Week 8-10: 损失函数设计
Week 11-16: 模型集成与训练
Week 17-19: 评估与论文撰写
```

**总预计时间**: 4-5 个月

---

## 👥 所需资源

- **计算资源**: GPU/TPU 用于训练（至少 1×A100 或同等）
- **数据**: MARCS 网格光谱、可选观测数据
- **人员**: 1-2 名研究者（天体物理 + 机器学习背景）

---

## 🚀 下一步行动

1. **立即可做**：
   - 详细阅读 PhysFormer 论文全文
   - 分析 TransformerPayne 源码结构
   - 创建 `physics_layers.py` 框架

2. **本周完成**：
   - 实现 Saha-Boltzmann 可微分版本
   - 设计物理损失函数原型
   - 建立评估基准

3. **本月完成**：
   - 完成所有物理层实现
   - 初步集成到 TransformerPayne
   - 开始小规模训练实验

---

*文档创建日期：2026-03-12*  
*项目位置：`/Users/jordan/.openclaw/workspace/projects/transformer_payne/`*
