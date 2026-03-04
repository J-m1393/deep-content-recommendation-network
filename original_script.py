#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from surprise import Dataset
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# In[3]:


class FeatureAttention(nn.Module):
    """特征级别的注意力机制"""
    def __init__(self, feature_dim):
        super(FeatureAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, feature_dim)
        attention_weights = self.attention_net(x)  # (batch_size, 1)
        weighted_features = x * attention_weights  # 特征加权
        return weighted_features, attention_weights.squeeze()


# In[4]:


class ContentBasedRecommender(nn.Module):
    """基于内容的推荐模型（使用注意力机制）"""

    def __init__(self, user_feature_dim, item_feature_dim, hidden_dims=[64, 32, 16]):
        super(ContentBasedRecommender, self).__init__()

        # 用户特征注意力
        self.user_attention = FeatureAttention(user_feature_dim)

        # 物品特征注意力  
        self.item_attention = FeatureAttention(item_feature_dim)

        # 交互网络
        layers = []
        input_dim = user_feature_dim + item_feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())

        self.interaction_net = nn.Sequential(*layers)

    def forward(self, user_features, item_features):
        # 应用注意力机制
        user_attended, user_weights = self.user_attention(user_features)
        item_attended, item_weights = self.item_attention(item_features)

        # 拼接特征
        combined = torch.cat([user_attended, item_attended], dim=1)

        # 预测
        output = self.interaction_net(combined)

        return output.squeeze(), user_weights, item_weights


# In[5]:


class ML100KProcessor:
    """处理MovieLens 100K数据集"""

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()

    def load_and_process_data(self, rating_threshold=4.0):

        # 加载数据
        data = Dataset.load_builtin('ml-100k')
        ratings = pd.DataFrame(data.raw_ratings, columns=['user', 'item', 'rating', 'timestamp'])

        print(f"数据统计:")
        print(f"- 用户数: {ratings['user'].nunique()}")
        print(f"- 物品数: {ratings['item'].nunique()}")
        print(f"- 交互数: {len(ratings)}")

        # 编码用户和物品ID
        ratings['user_id'] = self.user_encoder.fit_transform(ratings['user'])
        ratings['item_id'] = self.item_encoder.fit_transform(ratings['item'])

        # 创建用户特征（基于交互行为）
        print("创建用户特征...")
        user_features = self._create_user_features(ratings)

        # 创建物品特征（基于交互行为）  
        print("创建物品特征...")
        item_features = self._create_item_features(ratings)

        # 准备训练数据
        X_user, X_item, y = self._prepare_training_data(ratings, user_features, item_features, rating_threshold)

        return X_user, X_item, y, ratings

    def _create_user_features(self, ratings):
        """基于用户交互行为创建特征   用户特征工程"""
        user_stats = ratings.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count'],
            'item_id': 'nunique'
        }).fillna(0)

        user_stats.columns = ['rating_mean', 'rating_std', 'interaction_count', 'unique_items']

        # 添加用户活跃度特征
        user_stats['activity_level'] = pd.cut(
            user_stats['interaction_count'], 
            bins=[0, 10, 50, 100, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # 添加评分模式特征
        user_stats['rating_pattern'] = pd.cut(
            user_stats['rating_mean'],
            bins=[0, 2, 3, 4, 5],
            labels=[0, 1, 2, 3]
        ).astype(int)

        return user_stats

    def _create_item_features(self, ratings):
        """基于物品交互行为创建特征"""
        item_stats = ratings.groupby('item_id').agg({
            'rating': ['mean', 'std', 'count'],
            'user_id': 'nunique'
        }).fillna(0)

        item_stats.columns = ['rating_mean', 'rating_std', 'interaction_count', 'unique_users']

        # 添加物品流行度特征
        item_stats['popularity'] = pd.cut(
            item_stats['interaction_count'],
            bins=[0, 5, 20, 50, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # 添加物品评分特征
        item_stats['rating_level'] = pd.cut(
            item_stats['rating_mean'],
            bins=[0, 2, 3, 4, 5],
            labels=[0, 1, 2, 3]
        ).astype(int)

        return item_stats

    def _prepare_training_data(self, ratings, user_features, item_features, rating_threshold):
        """准备训练数据"""
        X_user, X_item, y = [], [], []

        feature_cols = ['rating_mean', 'rating_std', 'interaction_count', 'unique_items', 
                       'activity_level', 'rating_pattern']

        for _, row in ratings.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']

            if user_id in user_features.index and item_id in item_features.index:
                # 用户特征
                user_feat = user_features.loc[user_id][feature_cols].values
                X_user.append(user_feat)

                # 物品特征
                item_feat_cols = ['rating_mean', 'rating_std', 'interaction_count', 'unique_users', 
                                 'popularity', 'rating_level']
                item_feat = item_features.loc[item_id][item_feat_cols].values
                X_item.append(item_feat)

                # 标签：是否喜欢（评分高于阈值）
                y.append(1 if row['rating'] >= rating_threshold else 0)

        # 标准化特征
        X_user = self.user_scaler.fit_transform(X_user)
        X_item = self.item_scaler.fit_transform(X_item)

        return (np.array(X_user, dtype=np.float32), 
                np.array(X_item, dtype=np.float32), 
                np.array(y, dtype=np.float32))


# In[6]:


def train_model():
    """训练模型"""

    # 初始化处理器
    processor = ML100KProcessor()

    # 加载和处理数据
    X_user, X_item, y, ratings = processor.load_and_process_data(rating_threshold=4.0)

    print(f"\n特征维度:")
    print(f"- 用户特征: {X_user.shape[1]}维")
    print(f"- 物品特征: {X_item.shape[1]}维")
    print(f"- 样本数: {len(y)}")


    # 划分训练测试集
    X_user_train, X_user_test, X_item_train, X_item_test, y_train, y_test = train_test_split(
        X_user, X_item, y, test_size=0.2, random_state=42, stratify=y
    )

    # 转换为PyTorch张量
    X_user_train = torch.FloatTensor(X_user_train)
    X_user_test = torch.FloatTensor(X_user_test)
    X_item_train = torch.FloatTensor(X_item_train)
    X_item_test = torch.FloatTensor(X_item_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_user_train, X_item_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 模型参数
    user_feature_dim = X_user.shape[1]
    item_feature_dim = X_item.shape[1]

    # 初始化模型
    model = ContentBasedRecommender(user_feature_dim, item_feature_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    epochs = 30
    train_losses = []
    test_accuracies = []

    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_user, batch_item, batch_y in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs, _, _ = model(batch_user, batch_item)
            loss = criterion(outputs, batch_y)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 计算测试集准确率
        model.eval()
        with torch.no_grad():
            test_outputs, _, _ = model(X_user_test, X_item_test)
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test).float().mean()
            test_accuracies.append(accuracy.item())

        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

    return model, processor, ratings




# In[7]:


def recommend_for_user(model, processor, user_id, ratings, top_k=10):
    """为用户生成推荐"""

    model.eval()

    # 获取用户特征
    user_features = processor._create_user_features(ratings)
    feature_cols = ['rating_mean', 'rating_std', 'interaction_count', 'unique_items', 
                   'activity_level', 'rating_pattern']

    if user_id not in user_features.index:
        print(f"用户 {user_id} 不存在")
        return []

    user_feat = user_features.loc[user_id][feature_cols].values
    user_feat = processor.user_scaler.transform([user_feat])
    user_tensor = torch.FloatTensor(user_feat)

    # 获取所有物品特征
    item_features = processor._create_item_features(ratings)
    item_feat_cols = ['rating_mean', 'rating_std', 'interaction_count', 'unique_users', 
                     'popularity', 'rating_level']

    # 用户已经交互过的物品
    user_interacted = ratings[ratings['user_id'] == user_id]['item_id'].unique()

    scores = []
    user_weights = []
    item_weights = []

    for item_id in item_features.index:
        # 排除已经交互过的物品
        if item_id in user_interacted:
            continue

        item_feat = item_features.loc[item_id][item_feat_cols].values
        item_feat = processor.item_scaler.transform([item_feat])
        item_tensor = torch.FloatTensor(item_feat)

        with torch.no_grad():
            score, u_weight, i_weight = model(user_tensor, item_tensor)
            scores.append((item_id, score.item()))
            user_weights.append(u_weight.item())
            item_weights.append(i_weight.item())

    # 返回Top-K推荐
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k], np.mean(user_weights), np.mean(item_weights)


# In[8]:


def analyze_user_behavior(ratings, user_id):
    """分析用户行为"""
    user_ratings = ratings[ratings['user_id'] == user_id]

    print(f"\n用户 {user_id} 的行为分析:")
    print(f"- 评分数量: {len(user_ratings)}")
    print(f"- 平均评分: {user_ratings['rating'].mean():.2f}")
    print(f"- 评分分布:")
    rating_counts = user_ratings['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        print(f"  {rating}星: {count}次")


# In[ ]:


# 主程序
if __name__ == "__main__":
    print("基于内容的深度推荐系统（MovieLens 100K）")
    print("=" * 60)

    # 训练模型
    model, processor, ratings = train_model()

    print("\n" + "=" * 60)
    print("推荐示例")
    print("=" * 60)

    # 为用户0生成推荐
    user_id = 6
    analyze_user_behavior(ratings, user_id)

    recommendations, avg_user_weight, avg_item_weight = recommend_for_user(
        model, processor, user_id, ratings, top_k=5
    )

    print(f"\n为用户 {user_id} 的Top-5推荐:")
    print(f"平均用户特征注意力权重: {avg_user_weight:.4f}")
    print(f"平均物品特征注意力权重: {avg_item_weight:.4f}")
    print("-" * 50)

    for i, (item_id, score) in enumerate(recommendations, 1):
        # 获取物品的原始ID
        original_item_id = processor.item_encoder.inverse_transform([item_id])[0]
        item_ratings = ratings[ratings['item_id'] == item_id]

        print(f"{i}. 物品ID: {original_item_id} (内部ID: {item_id})")
        print(f"   预测喜欢概率: {score:.4f}")
        print(f"   平均评分: {item_ratings['rating'].mean():.2f}")
        print(f"   被评分次数: {len(item_ratings)}")
        print()

    # 模型解释

    #权重越高表示该特征对推荐决策的影响越大
    print(f"\n当前模型关注度:")
    print(f"- 用户特征: {avg_user_weight:.3f}")#用户特征注意力: 模型对用户行为特征的关注程度
    print(f"- 物品特征: {avg_item_weight:.3f}")#物品特征注意力: 模型对物品属性特征的关注程度


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




