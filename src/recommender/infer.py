from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import pandas as pd

from .data import ML100KProcessor, USER_FEATURE_COLS, ITEM_FEATURE_COLS
from .model import ContentBasedRecommender


def load_artifacts(model_path: str, processor_path: str, device: str = "cpu"):
    ckpt = torch.load(model_path, map_location=device)

    model = ContentBasedRecommender(
        ckpt["user_feature_dim"],
        ckpt["item_feature_dim"],
        hidden_dims=tuple(ckpt.get("hidden_dims", [64, 32, 16])),
        dropout=float(ckpt.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(torch.device(device))
    model.eval()

    with open(processor_path, "rb") as f:
        proc_obj = pickle.load(f)

    processor = ML100KProcessor()
    # restore fitted objects
    processor.user_encoder = proc_obj["user_encoder"]
    processor.item_encoder = proc_obj["item_encoder"]
    processor.user_scaler = proc_obj["user_scaler"]
    processor.item_scaler = proc_obj["item_scaler"]

    return model, processor, ckpt


def analyze_user_behavior(ratings: pd.DataFrame, user_id: int):
    user_ratings = ratings[ratings["user_id"] == user_id]
    print(f"\n用户 {user_id} 的行为分析:")
    print(f"- 评分数量: {len(user_ratings)}")
    if len(user_ratings) > 0:
        print(f"- 平均评分: {user_ratings['rating'].mean():.2f}")
        print("- 评分分布:")
        counts = user_ratings["rating"].value_counts().sort_index()
        for r, c in counts.items():
            print(f"  {r}星: {c}次")


def recommend_for_user(
    model: ContentBasedRecommender,
    processor: ML100KProcessor,
    ratings: pd.DataFrame,
    user_id: int,
    top_k: int = 10,
    device: str = "cpu",
) -> Tuple[List[Tuple[int, float]], float, float]:
    device_t = torch.device(device)
    # features for the given user
    user_features = processor.create_user_features(ratings)
    if user_id not in user_features.index:
        raise ValueError(f"user_id={user_id} not found")

    user_feat = user_features.loc[user_id][USER_FEATURE_COLS].values
    user_feat = processor.user_scaler.transform([user_feat])
    user_tensor = torch.tensor(user_feat, dtype=torch.float32, device=device_t)

    item_features = processor.create_item_features(ratings)
    user_interacted = set(ratings[ratings["user_id"] == user_id]["item_id"].unique().tolist())

    scores = []
    user_ws = []
    item_ws = []

    for item_id in item_features.index:
        if int(item_id) in user_interacted:
            continue
        item_feat = item_features.loc[item_id][ITEM_FEATURE_COLS].values
        item_feat = processor.item_scaler.transform([item_feat])
        item_tensor = torch.tensor(item_feat, dtype=torch.float32, device=device_t)

        with torch.no_grad():
            s, uw, iw = model(user_tensor, item_tensor)
            scores.append((int(item_id), float(s.item())))
            user_ws.append(float(uw.item()))
            item_ws.append(float(iw.item()))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k], float(np.mean(user_ws)) if user_ws else 0.0, float(np.mean(item_ws)) if item_ws else 0.0


def build_parser():
    p = argparse.ArgumentParser(description="Infer top-k recommendations for a user.")
    p.add_argument("--model_path", type=str, default="outputs/model.pt")
    p.add_argument("--processor_path", type=str, default="outputs/processor.pkl")
    p.add_argument("--user_id", type=int, default=6)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p


def main():
    args = build_parser().parse_args()

    # load ratings fresh (so it can be reproduced even if not serialized)
    processor_for_ratings = ML100KProcessor()
    ratings = processor_for_ratings.load_builtin_ratings()
    ratings = processor_for_ratings.encode_ids(ratings)

    model, processor, ckpt = load_artifacts(args.model_path, args.processor_path, device=args.device)

    analyze_user_behavior(ratings, args.user_id)
    recs, avg_uw, avg_iw = recommend_for_user(model, processor, ratings, args.user_id, top_k=args.top_k, device=args.device)

    print(f"\n为用户 {args.user_id} 的Top-{args.top_k}推荐:")
    print(f"平均用户特征注意力权重: {avg_uw:.4f}")
    print(f"平均物品特征注意力权重: {avg_iw:.4f}")
    print("-" * 50)

    for i, (item_id, score) in enumerate(recs, 1):
        original_item_id = processor.item_encoder.inverse_transform([item_id])[0]
        item_ratings = ratings[ratings["item_id"] == item_id]
        print(f"{i}. 物品ID: {original_item_id} (内部ID: {item_id})")
        print(f"   预测喜欢概率: {score:.4f}")
        if len(item_ratings) > 0:
            print(f"   平均评分: {item_ratings['rating'].mean():.2f}")
            print(f"   被评分次数: {len(item_ratings)}")
        print()

    print("\n当前模型关注度:")
    print(f"- 用户特征: {avg_uw:.3f}")
    print(f"- 物品特征: {avg_iw:.3f}")


if __name__ == "__main__":
    main()
