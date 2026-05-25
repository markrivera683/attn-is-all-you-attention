import math

import torch

from attention_regression.metrics import (
    attention_distribution_stats,
    band_energy_ratio,
    best_scale,
    check_covariance_matrix,
    effective_keys,
    entropy_rows,
    logit_scale_stats,
    rowwise_cosine,
    scaled_relative_error,
    signed_weight_stats,
    spd_stats,
    symmetry_error,
    topk_overlap,
    whitening_residual,
)


def test_covariance_matrix_centers_features():
    X = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    cov = check_covariance_matrix(X)

    expected_centered = X - X.mean(dim=0, keepdim=True)
    expected = expected_centered.T @ expected_centered / X.shape[0]
    assert torch.allclose(cov, expected)


def test_scaled_relative_error_handles_global_scale():
    reference = torch.eye(3)
    M = 2.5 * reference

    assert math.isclose(best_scale(M, reference), 2.5)
    assert scaled_relative_error(M, reference) < 1e-6


def test_whitening_residual_is_small_for_precision_matrix():
    Sigma = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    precision = torch.linalg.inv(Sigma)

    values = whitening_residual(precision, Sigma)

    assert values["left_whitening_residual"] < 1e-6
    assert values["right_whitening_residual"] < 1e-6
    assert math.isclose(values["left_whitening_scale"], 1.0, rel_tol=1e-6)
    assert math.isclose(values["right_whitening_scale"], 1.0, rel_tol=1e-6)


def test_spd_stats_and_symmetry_error_detect_non_spd_matrix():
    spd = torch.eye(3)
    non_spd = torch.diag(torch.tensor([1.0, -0.5, 2.0]))
    non_symmetric = torch.tensor([[1.0, 2.0], [0.0, 1.0]])

    assert spd_stats(spd)["is_positive_definite"] == 1.0
    assert spd_stats(non_spd)["negative_eig_fraction"] > 0.0
    assert symmetry_error(non_symmetric) > 0.0


def test_band_energy_ratio_identifies_banded_matrix():
    tridiagonal = torch.tensor(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0],
        ]
    )

    assert band_energy_ratio(tridiagonal, bandwidth=1) == 1.0
    assert band_energy_ratio(tridiagonal, bandwidth=0) < 1.0


def test_rowwise_cosine_and_logit_scale_ignore_row_bias():
    G_ref = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    G = 4.0 * G_ref + torch.tensor([[10.0], [-7.0]])

    row_cos = rowwise_cosine(G, G_ref, center=True)
    scale_stats = logit_scale_stats(G, G_ref, center=True)

    assert torch.allclose(row_cos, torch.ones_like(row_cos))
    assert math.isclose(scale_stats["logit_scale_alpha_mean"], 4.0, rel_tol=1e-6)
    assert scale_stats["logit_scaled_error_mean"] < 1e-6


def test_attention_distribution_stats_and_effective_keys():
    uniform = torch.full((2, 4), 0.25)
    peaked = torch.tensor(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
        ]
    )

    values = attention_distribution_stats(uniform, uniform)

    assert values["attn_js_mean"] < 1e-6
    assert values["attn_tv_mean"] < 1e-6
    assert torch.allclose(effective_keys(uniform), torch.full((2,), 4.0))
    assert torch.all(entropy_rows(peaked) < entropy_rows(uniform))


def test_topk_overlap_and_signed_weight_stats():
    A = torch.tensor([[0.5, 0.3, 0.2], [0.05, 0.8, 0.15]])
    B = torch.tensor([[0.4, 0.35, 0.25], [0.7, 0.05, 0.25]])
    W = torch.tensor([[1.0, -2.0, 0.0], [-1.0, 3.0, -4.0]])

    stats = signed_weight_stats(W)

    assert topk_overlap(A, B, k=2) == 0.75
    assert math.isclose(stats["negative_weight_fraction"], 3 / 6)
    assert math.isclose(stats["negative_weight_mass"], 7 / 11, rel_tol=1e-6)
    assert math.isclose(stats["positive_weight_mass"], 4 / 11, rel_tol=1e-6)
