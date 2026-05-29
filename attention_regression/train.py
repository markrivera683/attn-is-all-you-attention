import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SynDataset, iter_episodes, synthetic_data
from .logging_utils import ExperimentLogger, make_run_id
from .models import Attention, LearnBilinear, LearnSignedBilinear


@dataclass
class TrainConfig:
    seed: int = 43

    num_epochs: int = 50
    num_samples: int = 10000
    dim: int = 20

    data_mode: str = "global_w"
    num_episodes: int = 1000
    memory_size: int = 128
    query_size: int = 64

    w_true: torch.Tensor | None = None
    sigma: str = "default"
    rho: float = 0.9

    batch_size: int = 192
    query_ratio: float = 0.3333
    lr: float = 1e-3
    signed_lr: float = 1e-5
    signed_weight_decay: float = 1e-4
    noise: float = 0.1

    log_interval: int = 10
    ckpt_dir: Path = Path("ckpt/")
    run_dir: Path = Path("runs")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_wandb: bool = False
    wandb_project: str = "attention mechanism"
    use_compile: bool = False


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


def _global_batches(
    config: TrainConfig,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        batches.append(sample_memory_query_batch(X_batch, Y_batch, config.query_ratio))
    return batches


def _episodic_batches(
    config: TrainConfig,
    device: torch.device,
    epoch: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    episodes = iter_episodes(
        num_episodes=config.num_episodes,
        dim=config.dim,
        memory_size=config.memory_size,
        query_size=config.query_size,
        sigma=config.sigma,
        rho=config.rho,
        noise=config.noise,
        seed=config.seed + epoch,
    )
    return [
        (
            episode.X_m.to(device),
            episode.y_m.to(device),
            episode.X_q.to(device),
            episode.y_q.to(device),
        )
        for episode in episodes
    ]


def _iter_training_batches(
    config: TrainConfig,
    device: torch.device,
    epoch: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if config.data_mode == "global_w":
        return _global_batches(config, device)
    if config.data_mode == "episodic_w":
        return _episodic_batches(config, device, epoch)
    raise ValueError("data_mode must be 'global_w' or 'episodic_w'")


def _train_one_model(
    model_name: str,
    model: torch.nn.Module,
    config: TrainConfig,
    logger: ExperimentLogger,
    *,
    lr: float,
    weight_decay: float = 0.0,
) -> tuple[Path, float, int]:
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)
    train_model = torch.compile(model) if config.use_compile else model
    optimizer = torch.optim.Adam(
        train_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_model.train()
    best_loss = float("inf")
    steps = 0
    for epoch in range(config.num_epochs):
        for X_m, Y_m, X_q, Y_q in _iter_training_batches(config, device, epoch):
            optimizer.zero_grad()
            pred = train_model(X_q, X_m, Y_m)
            loss = F.mse_loss(pred, Y_q)
            loss.backward()
            optimizer.step()
            best_loss = min(best_loss, loss.item())

            if steps % config.log_interval == 0:
                logger.log(
                    {f"train/{model_name}/mse": loss.item()},
                    step=steps,
                    phase="train",
                )
            steps += 1

    ckpt_name = f"train--{model_name}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    ckpt_path = logger.checkpoint_dir / ckpt_name
    state_model = train_model
    torch.save(
        {
            model_name: state_model.state_dict(),
            "config": config_to_dict(config),
            "best_loss": best_loss,
            "num_steps": steps,
        },
        ckpt_path,
    )
    return ckpt_path, best_loss, steps


def run_training(
    config: TrainConfig,
    logger: ExperimentLogger | None = None,
) -> tuple[Path, Path, Path]:
    device = torch.device(config.device)
    print(f"Using device: {device}")

    owns_logger = logger is None
    if logger is None:
        logger = ExperimentLogger(
            Path(config.run_dir) / make_run_id("train"),
            config,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
        )

    try:
        set_seed(config.seed)
        bilinear_ckpt, bilinear_best, bilinear_steps = _train_one_model(
            "bilinear",
            LearnBilinear(config.dim),
            config,
            logger,
            lr=config.lr,
        )
        set_seed(config.seed)
        signed_bilinear_ckpt, signed_best, signed_steps = _train_one_model(
            "signed_bilinear",
            LearnSignedBilinear(config.dim),
            config,
            logger,
            lr=config.signed_lr,
            weight_decay=config.signed_weight_decay,
        )
        set_seed(config.seed)
        attn_ckpt, attn_best, attn_steps = _train_one_model(
            "attn",
            Attention(config.dim),
            config,
            logger,
            lr=config.lr,
        )
        logger.write_summary(
            {
                "bilinear": {
                    "data_mode": config.data_mode,
                    "best_mse": bilinear_best,
                    "num_steps": bilinear_steps,
                },
                "signed_bilinear": {
                    "data_mode": config.data_mode,
                    "best_mse": signed_best,
                    "num_steps": signed_steps,
                },
                "attn": {
                    "data_mode": config.data_mode,
                    "best_mse": attn_best,
                    "num_steps": attn_steps,
                },
            }
        )
    finally:
        if owns_logger:
            logger.close()

    return bilinear_ckpt, signed_bilinear_ckpt, attn_ckpt


def main() -> None:
    config = parse_config()
    run_training(config)


if __name__ == "__main__":
    main()
