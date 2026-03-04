# 开发者说明文档（Developer Guide）

## 1. 模块说明
- `src/recommender/data.py`
  - `ML100KProcessor`：数据加载、ID 编码、特征工程、训练样本构造
- `src/recommender/model.py`
  - `FeatureAttention`：注意力模块
  - `ContentBasedRecommender`：主模型（attention + MLP）
- `src/recommender/train.py`
  - `train()`：训练入口，保存模型与处理器
- `src/recommender/infer.py`
  - `load_artifacts()`：加载 checkpoint 与 processor
  - `recommend_for_user()`：对用户生成 Top-K 推荐

## 2. 关键数据约定
- 用户特征列：`USER_FEATURE_COLS`（6维）
- 物品特征列：`ITEM_FEATURE_COLS`（6维）
- 标签：`rating >= rating_threshold` => 1，否则 0

## 3. 如何新增特征
1) 在 `data.py` 的 `create_user_features` / `create_item_features` 中加入新列
2) 更新 `USER_FEATURE_COLS` / `ITEM_FEATURE_COLS`
3) 注意：新增特征后需要重新训练，并确保 scaler 与 checkpoint 保存/加载保持一致

## 4. 如何替换模型结构
- 在 `model.py` 中修改 `ContentBasedRecommender`
- checkpoint 保存结构参数：`hidden_dims/dropout/user_feature_dim/item_feature_dim`
- 推理时会从 checkpoint 重建模型，请保证字段一致或提供默认值

## 5. 训练指标扩展建议
- accuracy 受类别不平衡影响；可在 `train.py` 中增加：
  - AUC：`sklearn.metrics.roc_auc_score`
  - F1：`sklearn.metrics.f1_score`
  - PR-AUC：`average_precision_score`

## 6. 常见问题
- Surprise 数据下载失败：检查网络或代理；数据缓存目录在用户目录 `.surprise_data`
- Windows 下安装 torch：建议用官方命令安装 CPU 版本或对应 CUDA 版本
