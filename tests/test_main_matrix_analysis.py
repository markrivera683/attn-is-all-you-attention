import math

from main import Config, run_matrix_analysis


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
