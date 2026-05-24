import torch
import tyro

from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict

import attention_regression.metrics as metrics
from attention_regression.train import TrainConfig, run_training
from attention_regression.evaluate import EvalConfig, run_evaluation


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

    # Miscellaneous
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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


if __name__ == "__main__":
    config = parse_config()
    main(config)
