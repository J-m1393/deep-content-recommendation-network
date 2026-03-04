# 使用文档（User Manual）

## 1. 安装
```bash
pip install -r requirements.txt
```

## 2. 训练
```bash
python -m src.recommender.train --epochs 30 --output_dir outputs
```

常用参数：
- `--rating_threshold`：喜欢阈值（默认 4.0）
- `--epochs`：训练轮数
- `--batch_size`：批大小
- `--device`：cpu/cuda

## 3. 推理推荐
确保训练后有：
- `outputs/model.pt`
- `outputs/processor.pkl`

执行：
```bash
python -m src.recommender.infer --user_id 6 --top_k 5
```

输出解释：
- “预测喜欢概率”：模型输出的 Sigmoid 概率
- “平均用户/物品特征注意力权重”：越大表示模型越关注该侧特征（注意：当前实现是整体缩放权重）
