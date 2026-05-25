import torch
import tyro
import numpy as np

from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader

import attention_regression.metrics as metrics
from attention_regression.data import SynDataset, synthetic_data
from attention_regression.evaluate import (
    EvalConfig,
    load_ckpt,
    prediction_error_sums,
    ridge_references,
    run_evaluation,
)
from attention_regression.models import (
    Attention,
    DotProdAttention,
    LearnBilinear,
    LinearRegression,
    WhitenedDotProdAttention,
)
from attention_regression.train import TrainConfig, run_training, sample_memory_query_batch


@dataclass
class Config:
    """Overall configuration for the attention regression experiment"""

    seed: int = 43

    # Data config
    dim: int = 20
    w_true: torch.Tensor | None = None
    sigma: str = "default"
    rho: float = 0.9
    noise: float = 0.1

    # Model config
    lambda_reg: float = 1e-3

    # Train config
    train_num_epochs: int = 50
    train_num_samples: int = 10000

    lr: float = 1e-3
    train_batch_size: int = 192
    train_query_ratio: float = 0.3333
    
    log_interval: int = 10
    ckpt_dir: Path = Path("ckpt/")

    # Eval Config
    eval_num_samples: int = 192000

    eval_batch_size: int = 192
    eval_query_ratio: float = 0.3333
    
    bilinear_ckpt: Path | str | None = None
    attn_ckpt: Path | str | None = None

    # Matrix analysis config
    matrix_analysis_num_batches: int | None = None
    matrix_analysis_bandwidth: int = 1
    matrix_analysis_top_k: int = 5

    # Miscellaneous
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_use_wandb: bool = False


def parse_config(
    args: list[str] | None = None,
    *,
    defaults: Config | None = None,
    description: str = "Evaluate the Bilinear Model and Attention Model on synthetic data",
) -> Config:
    defaults = defaults or Config()
    return tyro.cli(
        Config,
        args=args,
        default=defaults,
        description=description,
)


def config_to_dict(config: Config) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def to_train_config(config: Config) -> TrainConfig:
    return TrainConfig(
        seed=config.seed,
        num_epochs=config.train_num_epochs,
        num_samples=config.train_num_samples,
        dim=config.dim,
        w_true=config.w_true,
        sigma=config.sigma,
        rho=config.rho,
        batch_size=config.train_batch_size,
        query_ratio=config.train_query_ratio,
        lr=config.lr,
        noise=config.noise,
        log_interval=config.log_interval,
        ckpt_dir=config.ckpt_dir,
        device=config.device,
        use_wandb=config.train_use_wandb,
    )


def to_eval_config(
    config: Config,
    *,
    bilinear_ckpt: Path | str | None,
    attn_ckpt: Path | str | None,
) -> EvalConfig:
    return EvalConfig(
        seed=config.seed,
        num_samples=config.eval_num_samples,
        dim=config.dim,
        w_true=config.w_true,
        sigma=config.sigma,
        rho=config.rho,
        batch_size=config.eval_batch_size,
        query_ratio=config.eval_query_ratio,
        lambda_reg=config.lambda_reg,
        noise=config.noise,
        bilinear_ckpt=bilinear_ckpt,
        attn_ckpt=attn_ckpt,
        device=config.device,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_population_covariance(config: Config) -> torch.Tensor:
    if config.sigma == "default":
        idx = torch.arange(config.dim)
        Sigma = config.rho ** (idx[:, None] - idx[None, :]).abs()
        return Sigma.float()

    sigma_path = Path(config.sigma)
    if not sigma_path.exists():
        raise ValueError(
            f"sigma must be 'default' or a valid file path, but got: {config.sigma}"
        )

    Sigma = torch.load(sigma_path, map_location="cpu")
    if not isinstance(Sigma, torch.Tensor):
        Sigma = torch.tensor(Sigma)
    Sigma = Sigma.float()
    if Sigma.shape != (config.dim, config.dim):
        raise ValueError(
            f"Sigma must have shape ({config.dim}, {config.dim}), but got {Sigma.shape}"
        )
    return Sigma


def prefix_metrics(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in values.items()}


def aggregate_metric(
    metric_sums: dict[str, float],
    metric_counts: dict[str, int],
    values: dict[str, float],
) -> None:
    for key, value in values.items():
        if np.isfinite(value):
            metric_sums[key] = metric_sums.get(key, 0.0) + value
            metric_counts[key] = metric_counts.get(key, 0) + 1


def matrix_level_metrics(
    M: torch.Tensor,
    batch_ref: torch.Tensor,
    population_precision: torch.Tensor,
    Sigma: torch.Tensor,
    bandwidth: int,
) -> dict[str, float]:
    values = {
        "m_batch_cosine": metrics.matrix_cosine_similarity(M, batch_ref),
        "m_batch_scaled_error": metrics.scaled_relative_error(M, batch_ref),
        "m_population_cosine": metrics.matrix_cosine_similarity(M, population_precision),
        "m_population_scaled_error": metrics.scaled_relative_error(M, population_precision),
        "m_band_energy_ratio": metrics.band_energy_ratio(M, bandwidth),
        "m_condition_number": metrics.condition_number(M),
    }
    values.update(prefix_metrics("m_spd", metrics.spd_stats(M)))
    values.update(prefix_metrics("m_whitening", metrics.whitening_residual(M, Sigma)))
    return values


def batch_matrix_analysis_metrics(
    model: torch.nn.Module,
    pred: torch.Tensor,
    Y_q: torch.Tensor,
    references: dict[str, torch.Tensor],
    Sigma: torch.Tensor,
    population_precision: torch.Tensor,
    config: Config,
) -> dict[str, float]:
    values: dict[str, float] = {}
    squared_error, absolute_error, query_count = prediction_error_sums(pred, Y_q)
    values["mse"] = squared_error / query_count
    values["mae"] = absolute_error / query_count

    if model.M is not None:
        values.update(
            matrix_level_metrics(
                model.M.detach().cpu(),
                references["M_ref"].detach().cpu(),
                population_precision,
                Sigma,
                config.matrix_analysis_bandwidth,
            )
        )

    if model.G is not None:
        is_linear_regression = isinstance(model, LinearRegression)
        G_ref = references["G_ref_linear"] if is_linear_regression else references["G_ref_softmax"]
        values["g_cosine"] = metrics.matrix_cosine_similarity(model.G, G_ref)
        values["g_scaled_error"] = metrics.scaled_relative_error(model.G, G_ref)
        values.update(prefix_metrics("g_row", metrics.rowwise_cosine_stats(model.G, G_ref)))
        values.update(prefix_metrics("g_scale", metrics.logit_scale_stats(model.G, G_ref)))

    if model.Attn is not None:
        if isinstance(model, LinearRegression):
            values.update(prefix_metrics("linear_weight", metrics.signed_weight_stats(model.Attn)))
        else:
            values.update(
                prefix_metrics(
                    "attn",
                    metrics.attention_distribution_stats(model.Attn, references["Attn_ref"]),
                )
            )
            top_k = min(config.matrix_analysis_top_k, model.Attn.shape[-1])
            values["attn_top1_overlap"] = metrics.topk_overlap(model.Attn, references["Attn_ref"], 1)
            values[f"attn_top{top_k}_overlap"] = metrics.topk_overlap(
                model.Attn,
                references["Attn_ref"],
                top_k,
            )

    values.update(prefix_metrics("linear_ref", metrics.signed_weight_stats(references["G_ref_linear"])))
    return values


def summarize_metrics(
    model_name: str,
    metric_sums: dict[str, float],
    metric_counts: dict[str, int],
) -> dict[str, float]:
    summary = {
        key: metric_sums[key] / metric_counts[key]
        for key in sorted(metric_sums)
        if metric_counts[key] > 0
    }

    print(f"\n==== Matrix Analysis: {model_name} ====")
    for key, value in summary.items():
        print(f"{key}: {value:.6g}")

    return summary


def run_matrix_analysis(
    config: Config,
    *,
    bilinear_ckpt: Path | str | None,
    attn_ckpt: Path | str | None,
) -> dict[str, dict[str, float]]:
    set_seed(config.seed)
    device = torch.device(config.device)
    Sigma = load_population_covariance(config)
    I = torch.eye(config.dim, dtype=Sigma.dtype)
    population_precision = torch.linalg.solve(Sigma + config.lambda_reg * I, I)

    X, Y = synthetic_data(
        samples=config.eval_num_samples,
        dim=config.dim,
        sigma=config.sigma,
        rho=config.rho,
        noise=config.noise,
        w=config.w_true,
    )
    dataloader = DataLoader(
        SynDataset(X, Y),
        batch_size=config.eval_batch_size,
        shuffle=True,
        drop_last=True,
    )

    models: list[tuple[str, torch.nn.Module]] = [
        ("LinearRegression", LinearRegression(config.dim, config.lambda_reg).to(device)),
        ("DotProdAttention", DotProdAttention(config.dim).to(device)),
        (
            "WhitenedDotProdAttention",
            WhitenedDotProdAttention(config.dim, config.lambda_reg).to(device),
        ),
        (
            "LearnBilinear",
            load_ckpt(LearnBilinear(config.dim).to(device), bilinear_ckpt, config.device),
        ),
        (
            "Attention",
            load_ckpt(Attention(config.dim).to(device), attn_ckpt, config.device),
        ),
    ]

    summaries: dict[str, dict[str, float]] = {}
    print("========== Matrix Analysis ==========")
    for model_name, model in models:
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        model.eval()

        with torch.no_grad():
            for step, (X_batch, Y_batch) in enumerate(dataloader):
                if config.matrix_analysis_num_batches is not None and step >= config.matrix_analysis_num_batches:
                    break

                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                X_m, Y_m, X_q, Y_q = sample_memory_query_batch(
                    X_batch,
                    Y_batch,
                    config.eval_query_ratio,
                )
                pred = model(X_q, X_m, Y_m)
                references = ridge_references(
                    X_q,
                    X_m,
                    config.lambda_reg,
                    config.dim,
                )
                batch_metrics = batch_matrix_analysis_metrics(
                    model,
                    pred,
                    Y_q,
                    references,
                    Sigma,
                    population_precision,
                    config,
                )
                aggregate_metric(metric_sums, metric_counts, batch_metrics)

        summaries[model_name] = summarize_metrics(model_name, metric_sums, metric_counts)

    return summaries


def main(config: Config) -> None:
    print("========== Training ==========")
    train_config = to_train_config(config)
    bilinear_ckpt, attn_ckpt = run_training(train_config)

    print("========== Training Completed ==========")
    print(f"Saved bilinear model checkpoint to: {bilinear_ckpt}")
    print(f"Saved attention model checkpoint to: {attn_ckpt}")
    print(f"Training logs are saved to wandb under the project: {train_config.wandb_project}")

    print("========== Evaluation ==========")
    eval_config = to_eval_config(
        config,
        bilinear_ckpt=bilinear_ckpt,
        attn_ckpt=attn_ckpt,
    )
    run_evaluation(eval_config)
    run_matrix_analysis(config, bilinear_ckpt=bilinear_ckpt, attn_ckpt=attn_ckpt)


if __name__ == "__main__":
    config = parse_config()
    main(config)
