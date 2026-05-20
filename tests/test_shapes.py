import pytest
import torch
from attention_regression.models import HandCraftAttention, AttentionMode


@pytest.mark.parametrize("mode", list(AttentionMode))
def test_attention_shapes(mode):

    Q = 7
    K = 11
    d = 5
    dy = 1

    Xq = torch.randn(Q, d)
    Xk = torch.randn(K, d)
    V  = torch.randn(K, dy)

    expected_shape = (Q, dy)
    model = HandCraftAttention(dim=d, mode=mode, lambda_reg=1e-5)

    Yhat = model(Xq, Xk, V)

    print(f"\n[Testing Mode]: {mode.name}")
    print("  Xq   :", Xq.shape)
    print("  Xk   :", Xk.shape)
    print("  V    :", V.shape)
    print("  Yhat :", Yhat.shape)

    assert Yhat.shape == expected_shape, (
        f"{mode.name} outputs wrong shape: "
        f"expected {expected_shape}, got {Yhat.shape}"
    )

    assert torch.isfinite(Yhat).all(), (
        f"{mode.name} outputs contain invalid values (NaN or Inf)"
    )