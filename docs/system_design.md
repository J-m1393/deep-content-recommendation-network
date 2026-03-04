# 系统设计文档：基于内容特征 + 注意力机制的深度推荐系统（MovieLens 100K）

## 1. 目标与范围
- 输入：MovieLens 100K 评分数据（user, item, rating, timestamp）
- 输出：
  1) 二分类预测：用户对物品“喜欢概率”（rating >= 阈值 => like=1）
  2) 推荐：对指定用户生成 Top-K 未交互物品列表
  3) 解释：输出用户/物品特征注意力权重（平均值）

## 2. 总体架构
1) 数据层：加载与清洗（Surprise 内置 ml-100k）、ID 编码（LabelEncoder）
2) 特征层：用户/物品交互行为统计特征（6维 + 6维）+ 标准化（StandardScaler）
3) 模型层：
   - FeatureAttention：对用户特征向量、物品特征向量分别进行“特征级注意力”加权
   - Interaction MLP：拼接加权后的 user/item 向量，预测喜欢概率（Sigmoid）
4) 训练评估层：train/test split + BCE Loss + Accuracy
5) 推理层：对用户未交互物品逐个打分排序，生成 Top-K，并统计平均注意力权重

## 3. 数据与特征设计

### 3.1 用户特征（6维）
- rating_mean：用户平均评分
- rating_std：评分标准差（反映口味稳定性）
- interaction_count：评分次数
- unique_items：评分过的物品数
- activity_level：基于 interaction_count 分桶（0/1/2/3）
- rating_pattern：基于 rating_mean 分桶（0/1/2/3）

### 3.2 物品特征（6维）
- rating_mean：物品平均评分
- rating_std：评分标准差
- interaction_count：被评分次数
- unique_users：评分过该物品的用户数
- popularity：基于 interaction_count 分桶（0/1/2/3）
- rating_level：基于 rating_mean 分桶（0/1/2/3）

### 3.3 标签定义
- like = 1 if rating >= rating_threshold else 0（默认阈值 4.0）

## 4. 模型结构

### 4.1 FeatureAttention
输入：x ∈ R^{d}
网络：Linear(d→d/2) + ReLU + Linear(d/2→1) + Sigmoid
输出：
- attention_weights ∈ (0,1)（每个样本一个标量权重）
- weighted_features = x * attention_weights

> 注意：该实现是“样本级别的统一缩放权重”，而不是对每一维分配独立权重；因此它更像 “gating” 而非传统 feature-wise attention。

### 4.2 Interaction MLP
输入：concat(user_attended, item_attended) ∈ R^{d_u + d_i}
隐藏层：[64, 32, 16]（默认） + ReLU + Dropout(0.3)
输出层：Linear → Sigmoid

## 5. 训练与评估
- 数据划分：test_size=0.2，stratify=y
- 损失函数：Binary Cross Entropy
- 指标：Accuracy（可扩展：AUC、F1、PR-AUC 用于不均衡数据更稳健）

## 6. 推理与推荐
- 对用户已交互物品集合进行过滤
- 对未交互物品逐个计算喜欢概率并排序
- 输出 Top-K 与平均注意力权重（用户、物品）

## 7. 可扩展性与优化方向
- 推理可向量化/批量化，减少 Python 循环开销
- 注意力可升级为真正 feature-wise attention（输出维度 d 的权重向量）
- 引入负采样、Pairwise loss（BPR）或多任务学习以提升排序效果
