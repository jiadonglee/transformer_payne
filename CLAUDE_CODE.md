# Claude Code 开发指南

> 项目：物理嵌入 TransformerPayne  
> 创建日期：2026-03-13

---

## 🎯 开发流程

### 1. 使用 Claude Code 开发

所有新功能和优化都应通过 Claude Code 完成：

```bash
# 基本用法
claude -p "实现 XXX 功能，要求..."

# 在项目中开发
cd /Users/jordan/.openclaw/workspace/projects/transformer_payne
claude -p "优化 physics_layers.py 的数值稳定性"

# 使用计划模式
claude -p "分析当前架构并提出改进建议" --plan
```

### 2. Git 工作流

```bash
# 开发前
git pull origin main

# 开发后
git add -A
git commit -m "type: description"
git push origin main
```

### 3. 提交信息规范

```
feat: 新功能
fix: 修复 bug
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试
chore: 构建/工具
```

---

## 📝 Claude Code 常用提示词

### 代码实现

```
实现 [功能描述]，要求：
1. 使用 JAX/JAX 实现
2. 保持可微分
3. 添加单元测试
4. 更新文档
```

### 代码审查

```
审查 [文件名] 代码：
1. 检查数值稳定性
2. 检查性能瓶颈
3. 检查代码规范
4. 提出改进建议
```

### Bug 修复

```
修复 [问题描述]：
1. 分析根本原因
2. 实现修复方案
3. 添加回归测试
4. 更新相关文档
```

### 性能优化

```
优化 [模块名] 性能：
1. 使用 JAX JIT 编译
2. 使用 vmap 批量处理
3. 减少内存分配
4. 基准测试对比
```

---

## 🧪 测试流程

### 运行测试

```bash
cd /Users/jordan/.openclaw/workspace/projects/transformer_payne
source .venv/bin/activate

# 运行所有测试
python -m pytest src/transformer_payne/test_physics_forward.py -v

# 运行特定测试
python -m pytest src/transformer_payne/test_physics_forward.py::test_physical_quantities_head -v
```

### 添加测试

```python
def test_new_feature():
    """测试新功能"""
    # Arrange
    ...
    
    # Act
    ...
    
    # Assert
    assert ...
```

---

## 📊 性能基准

### 推理性能

```bash
python benchmarks/benchmark_inference.py
```

### 训练性能

```bash
python benchmarks/benchmark_training.py
```

---

## 🔧 开发环境

### 设置

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e ".[dev]"

# 安装开发工具
pip install pytest black flake8 mypy
```

### 代码质量

```bash
# 格式化
black src/transformer_payne/

# Lint
flake8 src/transformer_payne/

# 类型检查
mypy src/transformer_payne/
```

---

## 📚 文档更新

### 代码注释

```python
def function_name(param1, param2):
    """简短描述
    
    详细描述（如需要）
    
    Args:
        param1: 参数 1 说明
        param2: 参数 2 说明
    
    Returns:
        返回值说明
    
    Raises:
        ExceptionType: 异常说明
    """
```

### API 文档

更新 `PHYSICS_EMBEDDED_README.md` 中的 API 参考部分。

---

## 🚀 部署流程

### 发布新版本

```bash
# 更新版本号
# pyproject.toml: version = "0.2.0"

# 提交
git commit -am "chore: release version 0.2.0"
git tag v0.2.0
git push origin main --tags

# 构建
python -m build

# 发布（如需要）
twine upload dist/*
```

---

## 📞 协作指南

### 代码审查

1. 所有 PR 需要至少 1 人审查
2. 运行测试确保通过
3. 检查代码质量
4. 更新文档

### 问题追踪

使用 GitHub Issues 追踪：
- Bug 报告
- 功能请求
- 性能问题
- 文档改进

---

## 🎯 当前任务

### 待完成

- [ ] 修复 Planck 函数数值稳定性
- [ ] 添加更多单元测试
- [ ] GPU 训练验证
- [ ] 性能基准测试

### 进行中

- [x] 核心功能实现
- [x] 基础测试
- [x] 文档编写

---

*最后更新：2026-03-13*
