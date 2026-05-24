import torch
import wandb
import tyro
import datetime
import numpy as np

from pathlib import Path
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Any

from models import LearnBilinear, Attention
from data import synthetic_data, SynDataset


@dataclass
class TrainConfig:
    seed: int = 42
    samples: int = 10000
    dim: int = 20
    lambda_reg: float = 1e-3
    lr: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 192
    query_ratio: float = 0.3333
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir: Path = Path("ckpt/")

    use_wandb: bool = False
    wandb_project: str = "attention mechanism"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train the Bilinear Model and Attention Model on synthetic data",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        args=args,
        default=defaults,
        description=description,
    )


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def sample_memory_query_batch(
        X: torch.Tensor,
        Y: torch.Tensor,
        query_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = X.shape[0]
    perm = torch.randperm(N, device=X.device)
    memory_size = int(N * (1 - query_ratio))
    idx_m = perm[:memory_size]
    idx_q = perm[memory_size:]

    Xm, Ym = X[idx_m], Y[idx_m]
    Xq, Yq = X[idx_q], Y[idx_q]

    return Xm, Ym, Xq, Yq


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ckpt_dir = config.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X, Y = synthetic_data(samples=config.samples, dim=config.dim, noise=0.1)
    data = SynDataset(X, Y)
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=True)


    # Training the Bilinear Model
    bilinear_raw = LearnBilinear(config.dim).to(device)

    exp_name = f"bilinear_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = Path("wandb") / exp_name
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=exp_name,
            config=config_to_dict(config),
            dir=str(log_dir),
        )

    bilinearModel = torch.compile(bilinear_raw)
    optimizer = torch.optim.Adam(bilinearModel.parameters(), lr=config.lr)

    bilinearModel.train()
    best_loss: float = float("inf")
    steps: int = 0
    
    for epoch in range(config.num_epochs):
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            X_m, Y_m, X_q, Y_q = sample_memory_query_batch(X, Y, config.query_ratio)
            optimizer.zero_grad()
            pred = bilinearModel(X_q, X_m, Y_m)
            loss = F.mse_loss(pred, Y_q)
            loss.backward()
            optimizer.step()
            best_loss = min(best_loss, loss.item())

            if config.use_wandb and steps % config.log_interval == 0:
                wandb.log({"train/loss": loss.item()}, step=steps)

            steps += 1

    if config.use_wandb:
        wandb.summary["num_steps"] = steps
        wandb.summary["num_epochs"] = config.num_epochs
        wandb.summary["best_loss"] = best_loss
        wandb.summary["model"] = "bilinear"
        wandb.finish()

    torch.save(
        {
            "bilinear": bilinearModel.state_dict(),
            "config": config_to_dict(config),
        },
        ckpt_dir / f"bilinear_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt",
    )


    # ================================
    # Training classic Attention Model
    attn_raw = Attention(config.dim).to(device)

    exp_name = f"attention_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = Path("wandb") / exp_name
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=exp_name,
            config=config_to_dict(config),
            dir=str(log_dir),
    )

    attnModel = torch.compile(attn_raw)
    optimizer = torch.optim.Adam(attnModel.parameters(), lr=config.lr)

    attnModel.train()
    steps: int = 0
    best_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            X_m, Y_m, X_q, Y_q = sample_memory_query_batch(X, Y, config.query_ratio)
            optimizer.zero_grad()
            loss = F.mse_loss(attnModel(X_q, X_m, Y_m), Y_q)
            loss.backward()
            optimizer.step()
            best_loss = min(best_loss, loss.item())

            if config.use_wandb and steps % config.log_interval == 0:
                wandb.log({"train/loss": loss.item()}, step=steps)

            steps += 1

    if config.use_wandb:
        wandb.summary["num_steps"] = steps
        wandb.summary["num_epochs"] = config.num_epochs
        wandb.summary["model"] = "attention"
        wandb.summary["best_loss"] = best_loss
        wandb.finish()

    torch.save(
        {
            "attn": attnModel.state_dict(),
            "config": config_to_dict(config),
        },
        ckpt_dir / f"attention_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt",
    )


def main() -> None:
    config = parse_config()
    run_training(config)


if __name__ == "__main__":
    main()