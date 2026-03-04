from __future__ import annotations

import argparse
import os
import json
import random
from dataclasses import asdict
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from .data import ML100KProcessor
from .model import ContentBasedRecommender


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)

    processor = ML100KProcessor()
    processed = processor.load_and_process_data(rating_threshold=args.rating_threshold)

    X_user, X_item, y = processed.X_user, processed.X_item, processed.y
    print(f"\n特征维度:")
    print(f"- 用户特征: {X_user.shape[1]}维")
    print(f"- 物品特征: {X_item.shape[1]}维")
    print(f"- 样本数: {len(y)}")

    X_user_train, X_user_test, X_item_train, X_item_test, y_train, y_test = train_test_split(
        X_user, X_item, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # tensors
    X_user_train = torch.tensor(X_user_train, dtype=torch.float32)
    X_user_test = torch.tensor(X_user_test, dtype=torch.float32)
    X_item_train = torch.tensor(X_item_train, dtype=torch.float32)
    X_item_test = torch.tensor(X_item_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_user_train, X_item_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ContentBasedRecommender(X_user.shape[1], X_item.shape[1], hidden_dims=tuple(args.hidden_dims), dropout=args.dropout)
    device = torch.device(args.device)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    test_accs = []

    print("\n开始训练...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for bu, bi, by in train_loader:
            bu, bi, by = bu.to(device), bi.to(device), by.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(bu, bi)
            loss = criterion(outputs, by)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            out, _, _ = model(X_user_test.to(device), X_item_test.to(device))
            preds = (out > 0.5).float()
            acc = (preds == y_test.to(device)).float().mean().item()
            test_accs.append(acc)

        if epoch % args.log_every == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_feature_dim": X_user.shape[1],
            "item_feature_dim": X_item.shape[1],
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "rating_threshold": args.rating_threshold,
            "seed": args.seed,
        },
        ckpt_path,
    )

    # save processor artifacts (encoders/scalers) + minimal metadata
    import pickle

    proc_path = os.path.join(args.output_dir, "processor.pkl")
    with open(proc_path, "wb") as f:
        pickle.dump(
            {
                "user_encoder": processor.user_encoder,
                "item_encoder": processor.item_encoder,
                "user_scaler": processor.user_scaler,
                "item_scaler": processor.item_scaler,
            },
            f,
        )

    metrics = {"train_losses": train_losses, "test_accuracy": test_accs}
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n已保存:")
    print(f"- 模型: {ckpt_path}")
    print(f"- 处理器: {proc_path}")
    print(f"- 指标: {metrics_path}")

    return {"ckpt": ckpt_path, "processor": proc_path, "metrics": metrics_path}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train content-based attention recommender (MovieLens 100K).")
    p.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and metrics.")
    p.add_argument("--rating_threshold", type=float, default=4.0, help="rating >= threshold => like=1 else 0")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32, 16])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--log_every", type=int, default=5)
    return p


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
