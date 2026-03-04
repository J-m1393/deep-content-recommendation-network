from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from surprise import Dataset  # type: ignore
except Exception:
    Dataset = None


USER_FEATURE_COLS: List[str] = [
    "rating_mean",
    "rating_std",
    "interaction_count",
    "unique_items",
    "activity_level",
    "rating_pattern",
]

ITEM_FEATURE_COLS: List[str] = [
    "rating_mean",
    "rating_std",
    "interaction_count",
    "unique_users",
    "popularity",
    "rating_level",
]


@dataclass
class ProcessedData:
    X_user: np.ndarray
    X_item: np.ndarray
    y: np.ndarray
    ratings: pd.DataFrame


class ML100KProcessor:
    """Process MovieLens 100K into user/item behavior features (6+6 dims) and binary labels."""

    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()

    def load_builtin_ratings(self) -> pd.DataFrame:
        """Load ratings from Surprise built-in 'ml-100k'. Requires internet the first time."""
        if Dataset is None:
            raise RuntimeError("scikit-surprise is required to load ml-100k.")
        data = Dataset.load_builtin("ml-100k")
        ratings = pd.DataFrame(data.raw_ratings, columns=["user", "item", "rating", "timestamp"])
        return ratings

    def encode_ids(self, ratings: pd.DataFrame) -> pd.DataFrame:
        ratings = ratings.copy()
        ratings["user_id"] = self.user_encoder.fit_transform(ratings["user"])
        ratings["item_id"] = self.item_encoder.fit_transform(ratings["item"])
        return ratings

    def create_user_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        user_stats = ratings.groupby("user_id").agg(
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
            interaction_count=("rating", "count"),
            unique_items=("item_id", "nunique"),
        ).fillna(0)

        user_stats["activity_level"] = pd.cut(
            user_stats["interaction_count"],
            bins=[0, 10, 50, 100, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(int)

        user_stats["rating_pattern"] = pd.cut(
            user_stats["rating_mean"],
            bins=[0, 2, 3, 4, 5],
            labels=[0, 1, 2, 3],
        ).astype(int)

        return user_stats

    def create_item_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        item_stats = ratings.groupby("item_id").agg(
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
            interaction_count=("rating", "count"),
            unique_users=("user_id", "nunique"),
        ).fillna(0)

        item_stats["popularity"] = pd.cut(
            item_stats["interaction_count"],
            bins=[0, 5, 20, 50, float("inf")],
            labels=[0, 1, 2, 3],
        ).astype(int)

        item_stats["rating_level"] = pd.cut(
            item_stats["rating_mean"],
            bins=[0, 2, 3, 4, 5],
            labels=[0, 1, 2, 3],
        ).astype(int)

        return item_stats

    def prepare_training_data(
        self,
        ratings: pd.DataFrame,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        rating_threshold: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_user, X_item, y = [], [], []

        for _, row in ratings.iterrows():
            uid = row["user_id"]
            iid = row["item_id"]
            if uid in user_features.index and iid in item_features.index:
                X_user.append(user_features.loc[uid][USER_FEATURE_COLS].values)
                X_item.append(item_features.loc[iid][ITEM_FEATURE_COLS].values)
                y.append(1.0 if float(row["rating"]) >= rating_threshold else 0.0)

        X_user = self.user_scaler.fit_transform(X_user)
        X_item = self.item_scaler.fit_transform(X_item)
        return (
            np.asarray(X_user, dtype=np.float32),
            np.asarray(X_item, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )

    def load_and_process_data(self, rating_threshold: float = 4.0) -> ProcessedData:
        ratings = self.load_builtin_ratings()
        ratings = self.encode_ids(ratings)

        # stats
        print("数据统计:")
        print(f"- 用户数: {ratings['user'].nunique()}")
        print(f"- 物品数: {ratings['item'].nunique()}")
        print(f"- 交互数: {len(ratings)}")

        print("创建用户特征...")
        u = self.create_user_features(ratings)
        print("创建物品特征...")
        it = self.create_item_features(ratings)

        X_user, X_item, y = self.prepare_training_data(ratings, u, it, rating_threshold=rating_threshold)
        return ProcessedData(X_user=X_user, X_item=X_item, y=y, ratings=ratings)
