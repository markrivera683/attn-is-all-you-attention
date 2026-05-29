import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SynDataset, synthetic_data
from .logging_utils import ExperimentLogger, make_run_id
from .metrics import condition_number, frobenius_cosine, matrix_cosine_similarity
from .models import (
    Attention,
    DotProdAttention,
    LearnBilinear,
    LearnSignedBilinear,
    LinearRegression,
    WhitenedDotProdAttention,
)
from .train import sample_memory_query_batch


@dataclass
class EvalConfig:
    seed: int = 43

    num_samples: int = 192000
    dim: int = 20
    data_mode: str = "global_w"

    w_true: torch.Tensor | None = None
    sigma: str = "default"
    rho: float = 0.9

    batch_size: int = 192
    query_ratio: float = 0.3333
    lambda_reg: float = 1e-3
    noise: float = 0.1

    bilinear_ckpt: Path | str | None = None
    signed_bilinear_ckpt: Path | str | None = None
    attn_ckpt: Path | str | None = None
    run_dir: Path = Path("runs")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_wandb: bool = False
    wandb_project: str = "attention mechanism"


def config_to_dict(config: EvalConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_config(
    args: list[str] | None = None,
    *,
    defaults: EvalConfig | None = None,
    description: str = "Evaluate the Bilinear Model and Attention Model on synthetic data",
) -> EvalConfig:
    defaults = defaults or EvalConfig()
    return tyro.cli(
        EvalConfig,
        args=args,
        default=defaults,
        description=description,
    )


def prediction_error_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float, int]:
    error = pred - target
    count = target.numel()
    squared_error = error.pow(2).sum().item()
    absolute_error = error.abs().sum().item()
    return squared_error, absolute_error, count


def ridge_references(
    X_q: torch.Tensor,
    X_m: torch.Tensor,
    lambda_reg: float,
    dim: int,
) -> dict[str, torch.Tensor]:
    I = torch.eye(
        X_m.shape[-1],
        device=X_m.device,
        dtype=X_m.dtype,
    )
    M_ref = torch.linalg.solve(X_m.T @ X_m + lambda_reg * I, I)
    G_ref_linear = X_q @ M_ref @ X_m.T
    G_ref_softmax = G_ref_linear / math.sqrt(dim)
    Attn_ref = F.softmax(G_ref_softmax, dim=-1)

    return {
        "M_ref": M_ref,
        "G_ref_linear": G_ref_linear,
        "G_ref_softmax": G_ref_softmax,
        "Attn_ref": Attn_ref,
    }


def structure_metrics(
    model: torch.nn.Module,
    references: dict[str, torch.Tensor],
) -> dict[str, float]:
    values: dict[str, float] = {}
    is_signed_weight_model = isinstance(model, (LinearRegression, LearnSignedBilinear))

    if model.G is not None:
        G_ref = references["G_ref_linear"] if is_signed_weight_model else references["G_ref_softmax"]
        values["eval/g_cosine"] = matrix_cosine_similarity(model.G, G_ref)
        values["eval/g_frobenius_cosine"] = frobenius_cosine(model.G, G_ref)

    if model.Attn is not None:
        Attn_ref = references["G_ref_linear"] if is_signed_weight_model else references["Attn_ref"]
        metric_prefix = "linear_weight" if is_signed_weight_model else "attn"
        values[f"eval/{metric_prefix}_cosine"] = matrix_cosine_similarity(model.Attn, Attn_ref)
        values[f"eval/{metric_prefix}_frobenius_cosine"] = frobenius_cosine(model.Attn, Attn_ref)

    if model.M is not None:
        values["eval/m_cosine"] = matrix_cosine_similarity(model.M, references["M_ref"])
        values["eval/m_frobenius_cosine"] = frobenius_cosine(model.M, references["M_ref"])
        values["eval/m_condition_number"] = condition_number(model.M)

    return values


def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    config: EvalConfig,
    logger: ExperimentLogger,
) -> dict[str, float | str]:
    set_seed(config.seed)
    device = torch.device(config.device)
    print(f"Evaluating {model_name} on {device}")

    X, Y = synthetic_data(
        samples=config.num_samples,
        dim=config.dim,
        sigma=config.sigma,
        rho=config.rho,
        noise=config.noise,
        w=config.w_true,
    )
    dataloader = DataLoader(
        SynDataset(X, Y),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_query_count = 0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    with torch.no_grad():
        for steps, (X_batch, Y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            X_m, Y_m, X_q, Y_q = sample_memory_query_batch(
                X_batch,
                Y_batch,
                config.query_ratio,
            )
            pred = model(X_q, X_m, Y_m)
            squared_error, absolute_error, query_count = prediction_error_sums(pred, Y_q)
            mse = squared_error / query_count
            mae = absolute_error / query_count
            total_loss += squared_error
            total_absolute_error += absolute_error
            total_query_count += query_count

            references = ridge_references(X_q, X_m, config.lambda_reg, config.dim)
            batch_metrics = structure_metrics(model, references)
            for key, value in batch_metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1

            logger.log(
                {
                    f"eval/{model_name}/mse": mse,
                    f"eval/{model_name}/mae": mae,
                    **{
                        f"eval/{model_name}/{key.removeprefix('eval/')}": value
                        for key, value in batch_metrics.items()
                    },
                },
                step=steps,
                phase="eval",
            )

    avg_loss = total_loss / total_query_count
    avg_mae = total_absolute_error / total_query_count
    summary: dict[str, float | str] = {
        "data_mode": config.data_mode,
        "mse": avg_loss,
        "mae": avg_mae,
        "total_query_count": float(total_query_count),
    }
    for key, value in metric_sums.items():
        summary[key] = value / metric_counts[key]
    return summary


def load_ckpt(
        model: nn.Module,
        ckpt_path: Path | str | None,
        device: str
) -> nn.Module:
    if ckpt_path is None:
        print("No checkpoint path provided, using untrained model.")
        return model.to(device)

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")

    loaded = torch.load(ckpt, map_location=device)
    if "bilinear" in loaded:
        state_dict = loaded["bilinear"]
    elif "signed_bilinear" in loaded:
        state_dict = loaded["signed_bilinear"]
    else:
        state_dict = loaded["attn"]

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            k.replace("_orig_mod.", "", 1): v
            for k, v in state_dict.items()
        }

    state_dict = {
        k: v
        for k, v in state_dict.items()
        if k not in {"_M", "_G", "Attn"}
    }

    model.load_state_dict(state_dict)
    model.to(device)
    return model


def run_evaluation(
    config: EvalConfig,
    logger: ExperimentLogger | None = None,
) -> dict[str, dict[str, float | str]]:
    owns_logger = logger is None
    if logger is None:
        logger = ExperimentLogger(
            Path(config.run_dir) / make_run_id("eval"),
            config,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )

    device = torch.device(config.device)
    models: list[tuple[str, torch.nn.Module]] = [
        ("linear_regression", LinearRegression(config.dim, config.lambda_reg).to(device)),
        ("dot_product", DotProdAttention(config.dim).to(device)),
        (
            "whitened_dot_product",
            WhitenedDotProdAttention(config.dim, config.lambda_reg).to(device),
        ),
        (
            "bilinear",
            load_ckpt(LearnBilinear(config.dim).to(device), config.bilinear_ckpt, config.device),
        ),
        (
            "signed_bilinear",
            load_ckpt(
                LearnSignedBilinear(config.dim).to(device),
                config.signed_bilinear_ckpt,
                config.device,
            ),
        ),
        (
            "attn",
            load_ckpt(Attention(config.dim).to(device), config.attn_ckpt, config.device),
        ),
    ]

    try:
        summaries = {
            model_name: evaluate_model(model_name, model, config, logger)
            for model_name, model in models
        }
        logger.write_summary(summaries)
    finally:
        if owns_logger:
            logger.close()

    print("\nEvaluation summary")
    for model_name, summary in summaries.items():
        print(f"{model_name}: mse={summary['mse']:.6g}, mae={summary['mae']:.6g}")
    return summaries


def main() -> None:
    config = parse_config()
    run_evaluation(config)


if __name__ == "__main__":
    main()
