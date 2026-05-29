import csv
import math

import torch

from attention_regression.comparison import episode_ridge
from attention_regression.data import iter_episodes
from main import Config, main


def test_episode_ridge_matches_manual_solution():
    episode = iter_episodes(
        num_episodes=1,
        dim=4,
        memory_size=9,
        query_size=3,
        noise=0.0,
        seed=0,
    )[0]
    lambda_reg = 1e-3

    pred, refs = episode_ridge(
        episode.X_q,
        episode.X_m,
        episode.y_m,
        lambda_reg,
    )

    I = torch.eye(episode.X_m.shape[-1])
    manual_beta = torch.linalg.solve(
        episode.X_m.T @ episode.X_m + lambda_reg * I,
        episode.X_m.T @ episode.y_m,
    )
    manual_pred = episode.X_q @ manual_beta

    assert torch.allclose(pred, manual_pred)
    assert refs["M_ref"].shape == (4, 4)
    assert refs["G_ref_linear"].shape == (3, 9)


def test_episode_ridge_beats_zero_predictor_when_memory_is_sufficient():
    episode = iter_episodes(
        num_episodes=1,
        dim=3,
        memory_size=128,
        query_size=32,
        noise=0.0,
        seed=1,
    )[0]

    pred, _ = episode_ridge(
        episode.X_q,
        episode.X_m,
        episode.y_m,
        lambda_reg=1e-6,
    )

    ridge_mse = torch.mean((pred - episode.y_q) ** 2).item()
    zero_mse = torch.mean(episode.y_q ** 2).item()
    assert ridge_mse < zero_mse


def test_episodic_end_to_end_writes_local_logs(tmp_path):
    config = Config(
        data_mode="episodic_w",
        dim=4,
        memory_size=8,
        query_size=4,
        train_num_epochs=1,
        train_num_episodes=8,
        eval_num_episodes=8,
        device="cpu",
        use_wandb=False,
        train_use_wandb=False,
        run_dir=tmp_path,
        run_id="episodic-test",
        log_interval=1,
    )

    main(config)

    run_path = tmp_path / "episodic-test"
    assert (run_path / "config.json").exists()
    assert (run_path / "metrics.jsonl").exists()
    assert (run_path / "summary.csv").exists()
    assert (run_path / "summary.json").exists()

    rows = list(csv.DictReader((run_path / "summary.csv").open(encoding="utf-8")))
    models = {row["model"] for row in rows}
    assert {
        "adaptive_ridge",
        "identity",
        "population_precision",
        "bilinear",
        "signed_bilinear",
        "attn",
    }.issubset(models)
    for row in rows:
        assert row["data_mode"] == "episodic_w"
        assert row["mse"]
        assert row["mse_ratio_to_ridge"]
        assert math.isfinite(float(row["mse"]))
        assert math.isfinite(float(row["mse_ratio_to_ridge"]))
