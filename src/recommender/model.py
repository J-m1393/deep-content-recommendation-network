import torch
import torch.nn as nn


class FeatureAttention(nn.Module):
    """Feature-level attention: returns (weighted_features, weights)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        hidden = max(1, feature_dim // 2)
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, feature_dim)
        weights = self.attention_net(x)          # (batch, 1)
        weighted = x * weights                   # broadcast to (batch, feature_dim)
        return weighted, weights.squeeze(-1)      # (batch, feature_dim), (batch,)


class ContentBasedRecommender(nn.Module):
    """Content-based recommender with user/item feature attention and MLP interaction net."""

    def __init__(self, user_feature_dim: int, item_feature_dim: int, hidden_dims=(64, 32, 16), dropout=0.3):
        super().__init__()
        self.user_attention = FeatureAttention(user_feature_dim)
        self.item_attention = FeatureAttention(item_feature_dim)

        layers = []
        input_dim = user_feature_dim + item_feature_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.interaction_net = nn.Sequential(*layers)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor):
        user_attended, user_w = self.user_attention(user_features)
        item_attended, item_w = self.item_attention(item_features)
        combined = torch.cat([user_attended, item_attended], dim=1)
        out = self.interaction_net(combined).squeeze(-1)
        return out, user_w, item_w
