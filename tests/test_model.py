import torch
from src.recommender.model import ContentBasedRecommender


def test_forward_shapes_and_range():
    model = ContentBasedRecommender(user_feature_dim=6, item_feature_dim=6)
    user = torch.randn(8, 6)
    item = torch.randn(8, 6)

    out, uw, iw = model(user, item)
    assert out.shape == (8,)
    assert uw.shape == (8,)
    assert iw.shape == (8,)
    assert torch.all(out >= 0) and torch.all(out <= 1)
