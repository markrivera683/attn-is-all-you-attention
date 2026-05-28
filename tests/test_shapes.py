import pytest
import torch
from attention_regression.models import (
    Attention,
    DotProdAttention,
    LearnBilinear,
    LearnSignedBilinear,
    LinearRegression,
    WhitenedDotProdAttention,
)


@pytest.mark.parametrize(
    ("model_name", "model_cls", "kwargs"),
    [
        ("DotProdAttention", DotProdAttention, {}),
        ("WhitenedDotProdAttention", WhitenedDotProdAttention, {"lambda_reg": 1e-5}),
        ("LinearRegression", LinearRegression, {"lambda_reg": 1e-5}),
        ("LearnBilinear", LearnBilinear, {}),
        ("LearnSignedBilinear", LearnSignedBilinear, {}),
        ("Attention", Attention, {}),
    ],
)

def test_attention_shapes(model_name, model_cls, kwargs):

    Q = 7
    K = 11
    d = 5
    dy = 1

    Xq = torch.randn(Q, d)
    Xk = torch.randn(K, d)
    V  = torch.randn(K, dy)

    expected_shape = (Q, dy)
    model = model_cls(dim=d, **kwargs)

    Yhat = model(Xq, Xk, V)

    print(f"\n[Testing Model]: {model_name}")
    print("  Xq   :", Xq.shape)
    print("  Xk   :", Xk.shape)
    print("  V    :", V.shape)
    print("  Yhat :", Yhat.shape)

    assert Yhat.shape == expected_shape, (
        f"{model_name} outputs wrong shape: "
        f"expected {expected_shape}, got {Yhat.shape}"
    )

    assert torch.isfinite(Yhat).all(), (
        f"{model_name} outputs contain invalid values (NaN or Inf)"
    )
