#!/usr/bin/env python3
"""
Test script for TransformerPayne
下载模型并生成一个简单的光谱
"""

import jax
jax.config.update("jax_enable_x64", True)

import transformer_payne as tp
import numpy as np

print("🚀 TransformerPayne 测试脚本")
print("=" * 50)

# 下载预训练模型
print("\n📥 下载模型权重...")
emulator = tp.TransformerPayne.download()
print("✅ 模型下载完成!")

# 打印模型信息
print(f"\n📊 模型信息:")
print(f"   太阳参数：{emulator.solar_parameters}")

# 生成光谱
print("\n🔬 生成光谱...")
wave = np.linspace(4670, 4960, 2000)  # 波长范围
mu = 1.0  # 光线垂直于太阳表面
parameters = emulator.solar_parameters  # 使用太阳参数

spectrum = emulator(np.log10(wave), mu, parameters)
print(f"✅ 光谱生成完成!")
print(f"   波长范围：{wave[0]:.1f} - {wave[-1]:.1f} Å")
print(f"   数据点数：{len(spectrum)}")

# 输出结果
intensity = spectrum[:, 0]
continuum = spectrum[:, 1]
normalized_intensity = intensity / continuum

print(f"\n📈 光谱统计:")
print(f"   强度范围：{intensity.min():.2e} - {intensity.max():.2e}")
print(f"   归一化强度范围：{normalized_intensity.min():.3f} - {normalized_intensity.max():.3f}")

print("\n" + "=" * 50)
print("✅ TransformerPayne 运行成功!")
