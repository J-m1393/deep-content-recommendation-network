# 测试文档（Test Report）

## 1. 测试环境
- OS：Windows 10/11（本项目兼容 Linux/macOS）
- Python：3.9+
- 依赖：见 `requirements.txt`

## 2. 测试范围
1) 模型前向（shape、数值范围）
2) 特征工程（输出列、维度、标准化可用）
3) 推荐逻辑（Top-K 数量、过滤已交互物品）

## 3. 测试用例
| 编号 | 用例 | 输入 | 预期 | 结果 |
|---|---|---|---|---|
| T1 | 模型前向输出shape | 随机 user/item 特征 batch | out=(batch,), weights=(batch,) | 通过 |
| T2 | 特征工程列完整 | 构造小型 ratings DataFrame | user/item 特征包含 6 列 | 通过 |
| T3 | 推荐过滤已交互 | user_id 已交互 item_id 集合 | 推荐结果不包含已交互 item | 通过 |

## 4. 执行方式
```bash
pytest -q
```

## 5. 训练回归验证（手工/集成）
- 运行训练脚本，观察 loss 下降与 test accuracy 稳定（示例日志可贴在此处）
- 运行推理脚本，输出 Top-K 推荐列表
