import pandas as pd
import torch
from src.recommender.data import ML100KProcessor
from src.recommender.model import ContentBasedRecommender
from src.recommender.infer import recommend_for_user


def test_recommend_excludes_interacted():
    # synthetic ratings
    df = pd.DataFrame({
        "user": ["u1","u1","u1","u2","u2","u3"],
        "item": ["i1","i2","i3","i1","i2","i3"],
        "rating": [5,3,4,2,5,4],
        "timestamp": [0,1,2,3,4,5],
    })

    p = ML100KProcessor()
    df = p.encode_ids(df)

    # fit scalers via training data prep
    uf = p.create_user_features(df)
    itf = p.create_item_features(df)
    p.prepare_training_data(df, uf, itf, rating_threshold=4.0)

    model = ContentBasedRecommender(user_feature_dim=6, item_feature_dim=6)

    user_id = int(df[df["user"]=="u1"]["user_id"].iloc[0])
    recs, _, _ = recommend_for_user(model, p, df, user_id, top_k=10, device="cpu")

    interacted = set(df[df["user_id"] == user_id]["item_id"].unique().tolist())
    for item_id, _ in recs:
        assert item_id not in interacted
