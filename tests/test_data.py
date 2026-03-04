import pandas as pd
import numpy as np
from src.recommender.data import ML100KProcessor, USER_FEATURE_COLS, ITEM_FEATURE_COLS


def test_feature_engineering_dims():
    # synthetic ratings
    df = pd.DataFrame({
        "user": ["u1","u1","u2","u3","u3","u3"],
        "item": ["i1","i2","i1","i2","i3","i1"],
        "rating": [5,3,4,2,5,4],
        "timestamp": [0,1,2,3,4,5],
    })

    p = ML100KProcessor()
    df = p.encode_ids(df)

    uf = p.create_user_features(df)
    itf = p.create_item_features(df)

    assert all(c in uf.columns for c in USER_FEATURE_COLS)
    assert all(c in itf.columns for c in ITEM_FEATURE_COLS)

    X_user, X_item, y = p.prepare_training_data(df, uf, itf, rating_threshold=4.0)
    assert X_user.shape[1] == len(USER_FEATURE_COLS)
    assert X_item.shape[1] == len(ITEM_FEATURE_COLS)
    assert set(np.unique(y)).issubset({0.0, 1.0})
