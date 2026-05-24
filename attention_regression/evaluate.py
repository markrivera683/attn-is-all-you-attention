import torch
import tyro
import wandb
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from typing import Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .train import sample_memory_query_batch
from .data import SynDataset, synthetic_data
from .models import (
    LearnBilinear,
    Attention,
    LinearRegression,
    WhitenedDotProdAttention,
    DotProdAttention,
)

@dataclass
class EvalConfig:
    seed: int = 43

    num_samples: int = 192000
    dim: int = 20

    w_true: torch.Tensor | None = None
    sigma: str = "default"
    rho: float = 0.9

    batch_size: int = 192
    query_ratio: float = 0.3333
    lambda_reg: float = 1e-3
    noise: float = 0.1

    bilinear_ckpt: Path | str | None = None
    attn_ckpt: Path | str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

def evaluate_model(model: torch.nn.Module, config: EvalConfig) -> None:
    set_seed(config.seed)
    print(f"Using device: {config.device}")

    exp_name = f"eval--{model.__class__.__name__}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    wandb.init(
        project="attention mechanism",
        name=exp_name,
        config=config_to_dict(config),
        dir=str(Path("wandb") / exp_name),
    )

    X, Y = synthetic_data(
        samples=config.num_samples,
        dim=config.dim,
        sigma=config.sigma,
        rho=config.rho,
        noise=config.noise,
        w=config.w_true,
    )
    data = SynDataset(X, Y)
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=True)

    model.eval()
    steps: int = 0
    total_loss = 0.0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(config.device), Y.to(config.device)
            X_m, Y_m, X_q, Y_q = sample_memory_query_batch(X, Y, config.query_ratio)
            pred = model(X_q, X_m, Y_m)
            loss = torch.nn.functional.mse_loss(pred, Y_q)
            total_loss += loss.item() * X_q.size(0)
            wandb.log({"eval/loss": loss.item()}, step=steps)
            steps += 1

    avg_loss = total_loss / len(data)
    print(f"Average Loss: {avg_loss}")
    wandb.summary["timestamp"] = datetime.now().isoformat()
    wandb.summary["avg_loss"] = avg_loss
    wandb.summary["num_steps"] = steps
    wandb.summary["num_samples"] = config.num_samples
    wandb.summary["w_true"] = config.w_true.tolist() if config.w_true is not None else None
    wandb.summary["sigma"] = config.sigma
    wandb.summary["rho"] = config.rho
    wandb.summary["noise"] = config.noise
    wandb.summary["lambda_reg"] = config.lambda_reg
    wandb.summary["query_ratio"] = config.query_ratio
    wandb.summary["model"] = f"{model.__class__.__name__}"
    wandb.finish()


def load_ckpt(
        model: nn.Module, 
        ckpt_path: Path | str | None,
        device: str
) -> nn.Module:
    if ckpt_path is None:
        print("No checkpoint path provided, using untrained model.")
        return model
    
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt}")
    
    ckpt = torch.load(ckpt)
    state_dict = ckpt["bilinear"] if "bilinear" in ckpt else ckpt["attn"]

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            k.replace("_orig_mod.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    return model


def run_evaluation(config: EvalConfig) -> None:
    print("==== Linear Regression Evaluation ====")
    model = LinearRegression(config.dim, config.lambda_reg).to(config.device)
    evaluate_model(model, config)

    print("\n==== DotProduct Model Evaluation ====")
    model = DotProdAttention(config.dim).to(config.device)
    evaluate_model(model, config)

    print("\n==== WhitenedDotProduct Model Evaluation ====")
    model = WhitenedDotProdAttention(config.dim, config.lambda_reg).to(config.device)
    evaluate_model(model, config)

    print("\n==== Bilinear Model Evaluation ====")
    model = LearnBilinear(config.dim).to(config.device)
    model = load_ckpt(model, config.bilinear_ckpt, config.device)
    evaluate_model(model, config)

    print("\n==== Attention Model Evaluation ====")
    model = Attention(config.dim).to(config.device)
    model = load_ckpt(model, config.attn_ckpt, config.device)
    evaluate_model(model, config)


def main() -> None:
    config = parse_config()
    run_evaluation(config)

if __name__ == "__main__":
    main()