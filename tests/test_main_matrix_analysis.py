import math
from pathlib import Path

import torch

from main import Config, resolve_w_true, run_matrix_analysis


def test_resolve_w_true_is_seeded_and_stable():
    config = Config(seed=123, dim=5, device="cpu")

    first = resolve_w_true(config)
    second = resolve_w_true(config)
    different_seed = resolve_w_true(Config(seed=124, dim=5, device="cpu"))

    assert first.w_true is not None
    assert second.w_true is not None
    assert different_seed.w_true is not None
    assert torch.allclose(first.w_true, second.w_true)
    assert not torch.allclose(first.w_true, different_seed.w_true)
    assert isinstance(first.ckpt_dir, Path)


def test_resolve_w_true_preserves_explicit_w():
    w_true = torch.arange(4, dtype=torch.float32)
    config = Config(seed=123, dim=4, w_true=w_true, device="cpu")

    resolved = resolve_w_true(config)

    assert resolved.w_true is w_true


def test_run_matrix_analysis_smoke_without_checkpoints():
    config = Config(
        seed=0,
        dim=4,
        eval_num_samples=16,
        eval_batch_size=8,
        eval_query_ratio=0.25,
        matrix_analysis_num_batches=1,
        matrix_analysis_top_k=2,
        device="cpu",
    )

    summaries = run_matrix_analysis(
        config,
        bilinear_ckpt=None,
        signed_bilinear_ckpt=None,
        attn_ckpt=None,
    )

    assert set(summaries) == {
        "LinearRegression",
        "DotProdAttention",
        "WhitenedDotProdAttention",
        "LearnBilinear",
        "LearnSignedBilinear",
        "Attention",
    }
    assert "mse" in summaries["LinearRegression"]
    assert "linear_weight/negative_weight_fraction" in summaries["LinearRegression"]
    assert "attn/attn_js_mean" in summaries["DotProdAttention"]
    assert "m_batch_scaled_error" in summaries["LearnBilinear"]
    assert "linear_weight/negative_weight_fraction" in summaries["LearnSignedBilinear"]
    assert all(
        math.isfinite(value)
        for model_summary in summaries.values()
        for value in model_summary.values()
    )
