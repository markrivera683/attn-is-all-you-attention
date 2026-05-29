from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

class SynDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


@dataclass
class Episode:
    X_m: torch.Tensor
    y_m: torch.Tensor
    X_q: torch.Tensor
    y_q: torch.Tensor
    w: torch.Tensor


def load_covariance(
    dim: int,
    sigma: str = "default",
    rho: float = 0.9,
) -> torch.Tensor:
    if sigma == "default":
        if not 0 <= rho < 1:
            raise ValueError("rho must be in the range [0, 1)")
        idx = torch.arange(dim)
        Sigma = rho ** (idx[:, None] - idx[None, :]).abs()
        return Sigma.float()

    sigma_path = Path(sigma)
    if not sigma_path.exists():
        raise ValueError(
            f"sigma must be 'default' or a valid file path, but got: {sigma}"
        )

    Sigma = torch.load(sigma_path, map_location="cpu")
    if not isinstance(Sigma, torch.Tensor):
        Sigma = torch.tensor(Sigma)

    Sigma = Sigma.float()
    if Sigma.shape != (dim, dim):
        raise ValueError(
            f"Sigma must have shape ({dim}, {dim}), but got {Sigma.shape}"
        )
    return Sigma


def sample_correlated_features(
    samples: int,
    covariance_cholesky: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z = torch.randn(
        samples,
        covariance_cholesky.shape[0],
        generator=generator,
        dtype=covariance_cholesky.dtype,
    )
    return z @ covariance_cholesky.T


def sample_episode(
    dim: int,
    memory_size: int,
    query_size: int,
    covariance_cholesky: torch.Tensor,
    noise: float,
    generator: torch.Generator | None = None,
) -> Episode:
    w = torch.randn(dim, generator=generator, dtype=covariance_cholesky.dtype)
    X_m = sample_correlated_features(memory_size, covariance_cholesky, generator)
    X_q = sample_correlated_features(query_size, covariance_cholesky, generator)
    y_m = X_m @ w + noise * torch.randn(
        memory_size,
        generator=generator,
        dtype=covariance_cholesky.dtype,
    )
    y_q = X_q @ w + noise * torch.randn(
        query_size,
        generator=generator,
        dtype=covariance_cholesky.dtype,
    )
    return Episode(X_m=X_m, y_m=y_m, X_q=X_q, y_q=y_q, w=w)


def iter_episodes(
    num_episodes: int,
    dim: int,
    memory_size: int,
    query_size: int,
    sigma: str = "default",
    rho: float = 0.9,
    noise: float = 0.1,
    seed: int | None = None,
) -> list[Episode]:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    Sigma = load_covariance(dim=dim, sigma=sigma, rho=rho)
    L = torch.linalg.cholesky(Sigma)
    return [
        sample_episode(
            dim=dim,
            memory_size=memory_size,
            query_size=query_size,
            covariance_cholesky=L,
            noise=noise,
            generator=generator,
        )
        for _ in range(num_episodes)
    ]


def synthetic_data(
    samples: int = 1000,
    dim: int = 20,
    sigma: str = "default",
    rho: float = 0.9,
    noise: float = 0.1,
    w: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    Sigma = load_covariance(dim=dim, sigma=sigma, rho=rho)
    L = torch.linalg.cholesky(Sigma)
    z = torch.randn(samples, dim)
    X = z @ L.t()

    if w is not None:
        if w.shape != (dim,):
            raise ValueError(f"w must have shape ({dim},), but got {w.shape}")
        true_w = w
    else:
        true_w = torch.randn(dim)

    y = X @ true_w + noise * torch.randn(X.shape[0])

    return X, y

