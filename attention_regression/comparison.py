from pathlib import Path
from typing import Any

import torch

from . import metrics
from .data import iter_episodes, load_covariance
from .evaluate import load_ckpt, prediction_error_sums, ridge_references
from .logging_utils import ExperimentLogger, make_run_id
from .models import Attention, LearnBilinear, LearnSignedBilinear


def episode_ridge(
    X_q: torch.Tensor,
    X_m: torch.Tensor,
    y_m: torch.Tensor,
    lambda_reg: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    refs = ridge_references(
        X_q=X_q,
        X_m=X_m,
        lambda_reg=lambda_reg,
        dim=X_m.shape[-1],
    )
    pred = refs["G_ref_linear"] @ y_m
    return pred, refs


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    return numerator / max(denominator, eps)


def _prediction_summary(
    pred: torch.Tensor,
    target: torch.Tensor,
    ridge_mse: float,
) -> dict[str, float]:
    squared_error, absolute_error, count = prediction_error_sums(pred, target)
    mse = squared_error / count
    mae = absolute_error / count
    return {
        "mse": mse,
        "mae": mae,
        "mse_ratio_to_ridge": _safe_ratio(mse, ridge_mse),
    }


def _kernel_metrics(
    M: torch.Tensor | None,
    G: torch.Tensor | None,
    weights: torch.Tensor | None,
    references: dict[str, torch.Tensor],
    population_precision: torch.Tensor,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if M is not None:
        values["m_cosine_to_population_precision"] = metrics.matrix_cosine_similarity(
            M,
            population_precision,
        )
        values["m_scaled_error_to_population_precision"] = metrics.scaled_relative_error(
            M,
            population_precision,
        )
    if G is not None:
        values["g_cosine_to_ridge"] = metrics.matrix_cosine_similarity(
            G,
            references["G_ref_linear"],
        )
        values["g_scaled_error_to_ridge"] = metrics.scaled_relative_error(
            G,
            references["G_ref_linear"],
        )
    if weights is not None:
        values["signed_weight_negative_mass"] = metrics.signed_weight_stats(
            weights
        )["negative_weight_mass"]
        if torch.all(weights >= 0):
            values["softmax_vs_ridge_distribution_gap"] = metrics.attention_distribution_stats(
                weights,
                references["Attn_ref"],
            )["attn_js_mean"]
    return values


def _fixed_kernel_episode_metrics(
    M: torch.Tensor,
    X_q: torch.Tensor,
    X_m: torch.Tensor,
    y_m: torch.Tensor,
    y_q: torch.Tensor,
    references: dict[str, torch.Tensor],
    population_precision: torch.Tensor,
    ridge_mse: float,
) -> dict[str, float]:
    G = X_q @ M @ X_m.T
    pred = G @ y_m
    values = _prediction_summary(pred, y_q, ridge_mse)
    values.update(_kernel_metrics(M, G, G, references, population_precision))
    return values


def _learned_model_episode_metrics(
    model: torch.nn.Module,
    X_q: torch.Tensor,
    X_m: torch.Tensor,
    y_m: torch.Tensor,
    y_q: torch.Tensor,
    references: dict[str, torch.Tensor],
    population_precision: torch.Tensor,
    ridge_mse: float,
) -> dict[str, float]:
    pred = model(X_q, X_m, y_m)
    values = _prediction_summary(pred, y_q, ridge_mse)
    values.update(
        _kernel_metrics(
            model.M.detach() if model.M is not None else None,
            model.G.detach() if model.G is not None else None,
            model.Attn.detach() if model.Attn is not None else None,
            references,
            population_precision,
        )
    )
    return values


def _accumulate(
    sums: dict[str, float],
    counts: dict[str, int],
    values: dict[str, float],
) -> None:
    for key, value in values.items():
        if torch.isfinite(torch.tensor(value)):
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1


def _summarize(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {
        key: sums[key] / counts[key]
        for key in sorted(sums)
        if counts[key] > 0
    }


def run_episode_comparison(
    config: Any,
    *,
    bilinear_ckpt: Path | str | None = None,
    signed_bilinear_ckpt: Path | str | None = None,
    attn_ckpt: Path | str | None = None,
    logger: ExperimentLogger | None = None,
) -> dict[str, dict[str, float | str]]:
    device = torch.device(getattr(config, "device", "cpu"))
    dim = getattr(config, "dim")
    lambda_reg = getattr(config, "lambda_reg")
    data_mode = getattr(config, "data_mode", "episodic_w")
    eval_seed = getattr(config, "eval_seed", getattr(config, "seed", 43))
    eval_num_episodes = getattr(config, "eval_num_episodes", 1000)
    memory_size = getattr(config, "memory_size", 128)
    query_size = getattr(config, "query_size", 64)
    sigma = getattr(config, "sigma", "default")
    rho = getattr(config, "rho", 0.9)
    noise = getattr(config, "noise", 0.1)
    use_wandb = getattr(config, "use_wandb", False)
    wandb_project = getattr(config, "wandb_project", "attention mechanism")
    run_dir = Path(getattr(config, "run_dir", Path("runs")))

    Sigma = load_covariance(dim=dim, sigma=sigma, rho=rho).to(device)
    I = torch.eye(dim, device=device, dtype=Sigma.dtype)
    population_precision = torch.linalg.solve(Sigma + lambda_reg * I, I)

    ckpt_bilinear = bilinear_ckpt if bilinear_ckpt is not None else getattr(config, "bilinear_ckpt", None)
    ckpt_signed = (
        signed_bilinear_ckpt
        if signed_bilinear_ckpt is not None
        else getattr(config, "signed_bilinear_ckpt", None)
    )
    ckpt_attn = attn_ckpt if attn_ckpt is not None else getattr(config, "attn_ckpt", None)

    learned_models: list[tuple[str, torch.nn.Module]] = [
        ("bilinear", load_ckpt(LearnBilinear(dim).to(device), ckpt_bilinear, str(device))),
        (
            "signed_bilinear",
            load_ckpt(LearnSignedBilinear(dim).to(device), ckpt_signed, str(device)),
        ),
        ("attn", load_ckpt(Attention(dim).to(device), ckpt_attn, str(device))),
    ]
    for _, model in learned_models:
        model.eval()

    owns_logger = logger is None
    if logger is None:
        logger = ExperimentLogger(
            run_dir / make_run_id("episode-compare"),
            config,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
        )

    metric_sums: dict[str, dict[str, float]] = {}
    metric_counts: dict[str, dict[str, int]] = {}
    model_names = [
        "adaptive_ridge",
        "identity",
        "population_precision",
        *[name for name, _ in learned_models],
    ]
    for model_name in model_names:
        metric_sums[model_name] = {}
        metric_counts[model_name] = {}

    episodes = iter_episodes(
        num_episodes=eval_num_episodes,
        dim=dim,
        memory_size=memory_size,
        query_size=query_size,
        sigma=sigma,
        rho=rho,
        noise=noise,
        seed=eval_seed,
    )

    try:
        with torch.no_grad():
            for step, episode in enumerate(episodes):
                X_m = episode.X_m.to(device)
                y_m = episode.y_m.to(device)
                X_q = episode.X_q.to(device)
                y_q = episode.y_q.to(device)
                ridge_pred, references = episode_ridge(X_q, X_m, y_m, lambda_reg)
                ridge_values = _prediction_summary(ridge_pred, y_q, ridge_mse=1.0)
                ridge_values["mse_ratio_to_ridge"] = 1.0
                ridge_values.update(
                    _kernel_metrics(
                        references["M_ref"],
                        references["G_ref_linear"],
                        references["G_ref_linear"],
                        references,
                        population_precision,
                    )
                )
                ridge_mse = ridge_values["mse"]

                per_model_values: dict[str, dict[str, float]] = {
                    "adaptive_ridge": ridge_values,
                    "identity": _fixed_kernel_episode_metrics(
                        I,
                        X_q,
                        X_m,
                        y_m,
                        y_q,
                        references,
                        population_precision,
                        ridge_mse,
                    ),
                    "population_precision": _fixed_kernel_episode_metrics(
                        population_precision,
                        X_q,
                        X_m,
                        y_m,
                        y_q,
                        references,
                        population_precision,
                        ridge_mse,
                    ),
                }

                for model_name, model in learned_models:
                    per_model_values[model_name] = _learned_model_episode_metrics(
                        model,
                        X_q,
                        X_m,
                        y_m,
                        y_q,
                        references,
                        population_precision,
                        ridge_mse,
                    )

                for model_name, values in per_model_values.items():
                    _accumulate(
                        metric_sums[model_name],
                        metric_counts[model_name],
                        values,
                    )
                    logger.log(
                        {
                            "model": model_name,
                            "data_mode": data_mode,
                            **{
                                f"compare/{model_name}/{key}": value
                                for key, value in values.items()
                            },
                        },
                        step=step,
                        phase="compare",
                    )

        summaries: dict[str, dict[str, float | str]] = {}
        for model_name in model_names:
            summaries[model_name] = {
                "data_mode": data_mode,
                **_summarize(metric_sums[model_name], metric_counts[model_name]),
            }
        logger.write_summary(summaries)
    finally:
        if owns_logger:
            logger.close()

    print("\nPer-episode comparison summary")
    for model_name, summary in summaries.items():
        print(
            f"{model_name}: mse={summary['mse']:.6g}, "
            f"ratio={summary['mse_ratio_to_ridge']:.6g}"
        )
    return summaries
