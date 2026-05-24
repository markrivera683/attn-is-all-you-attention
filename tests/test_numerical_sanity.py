import math

import torch
from attention_regression.evaluate import (
    prediction_error_sums,
    ridge_references,
    structure_metrics,
)
from attention_regression.models import LinearRegression, WhitenedDotProdAttention


def test_prediction_error_sums_use_query_count():
    pred = torch.tensor([1.0, 3.0, 5.0])
    target = torch.tensor([0.0, 1.0, 2.0])

    squared_error, absolute_error, count = prediction_error_sums(pred, target)

    assert squared_error == 14.0
    assert absolute_error == 6.0
    assert count == 3
    assert squared_error / count == 14.0 / 3.0
    assert absolute_error / count == 2.0


def test_reference_matrix_metrics_are_finite():
    torch.manual_seed(0)
    dim = 3
    lambda_reg = 1e-3
    X_m = torch.randn(5, dim)
    X_q = torch.randn(2, dim)
    Y_m = torch.randn(5)

    model = WhitenedDotProdAttention(dim=dim, lambda_reg=lambda_reg)
    model(X_q, X_m, Y_m)
    references = ridge_references(X_q, X_m, lambda_reg, dim)
    values = structure_metrics(model, references)

    assert references["M_ref"].shape == (dim, dim)
    assert references["G_ref_softmax"].shape == (X_q.shape[0], X_m.shape[0])
    assert references["Attn_ref"].shape == (X_q.shape[0], X_m.shape[0])
    assert "eval/m_cosine" in values
    assert "eval/g_cosine" in values
    assert "eval/attn_cosine" in values
    assert "eval/m_condition_number" in values
    assert all(math.isfinite(value) for value in values.values())



def test_linear_regression_uses_linear_weight_metrics():
    torch.manual_seed(1)
    dim = 4
    lambda_reg = 1e-3
    X_m = torch.randn(6, dim)
    X_q = torch.randn(3, dim)
    Y_m = torch.randn(6)

    model = LinearRegression(dim=dim, lambda_reg=lambda_reg)
    model(X_q, X_m, Y_m)
    values = structure_metrics(model, ridge_references(X_q, X_m, lambda_reg, dim))

    assert "eval/linear_weight_cosine" in values
    assert "eval/attn_cosine" not in values
    assert all(math.isfinite(value) for value in values.values())