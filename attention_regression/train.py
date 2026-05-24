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
    num_epochs: int = 5
    eval_episodes: int = 100
    batch_size: int = 128
    query_ratio: float = 0.333
    log_interval: int = 20
    device: str = "cpu"

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
    memory_size = int(N * (1 - query_ratio))
    query_size = int(N * query_ratio)

    idx_m = torch.randint(0, N, (memory_size,))
    idx_q = torch.randint(0, N, (query_size,))

    Xm, Ym = X[idx_m], Y[idx_m]
    Xq, Yq = X[idx_q], Y[idx_q]

    return Xm, Ym, Xq, Yq


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, Y = synthetic_data(samples=config.samples, dim=config.dim, noise=0.1)
    data = SynDataset(X, Y)
    dataloader = DataLoader(data, batch_size=config.batch_size)

    # Training the Bilinear Model
    bilinearModel = LearnBilinear(config.dim).to(device)

    exp_name = f"bilinear_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = Path("logs") / exp_name
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=exp_name,
            config=config_to_dict(config),
            dir=str(log_dir),
        )

    bilinearModel = torch.compile(bilinearModel)
    optimizer = torch.optim.Adam(bilinearModel.parameters(), lr=config.lr)

    bilinearModel.train()
    steps: int = 0
    
    for epoch in range(config.num_epochs):
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            X_m, Y_m, X_q, Y_q = sample_memory_query_batch(X, Y, config.query_ratio)
            optimizer.zero_grad()
            loss = F.mse_loss(bilinearModel(X_q, X_m, Y_m), Y_q)
            loss.backward()
            optimizer.step()

            if steps % config.log_interval == 0:
                wandb.log({"loss": loss.item()}, step=steps)

            steps += 1
    
    if config.use_wandb:
        wandb.finish()

    # Training classic Attention Model
    attnModel = Attention(config.dim).to(device)

    exp_name = f"attention_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = Path("logs") / exp_name
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=exp_name,
            config=config_to_dict(config),
            dir=str(log_dir),
    )

    attnModel = torch.compile(attnModel)
    optimizer = torch.optim.Adam(attnModel.parameters(), lr=config.lr)

    attnModel.train()
    steps: int = 0
    
    for epoch in range(config.num_epochs):
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            X_m, Y_m, X_q, Y_q = sample_memory_query_batch(X, Y, config.query_ratio)
            optimizer.zero_grad()
            loss = F.mse_loss(attnModel(X_q, X_m, Y_m), Y_q)
            loss.backward()
            optimizer.step()

            if steps % config.log_interval == 0:
                wandb.log({"loss": loss.item()}, step=steps)

            steps += 1

    if config.use_wandb:
        wandb.finish()


def main() -> None:
    config = parse_config()
    run_training(config)


if __name__ == "__main__":
    main()