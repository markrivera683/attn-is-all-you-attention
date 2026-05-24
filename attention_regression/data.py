import torch 
import numpy as np
from torch.utils.data import Dataset

from pathlib import Path

class SynDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def synthetic_data(
    samples: int = 1000,
    dim: int = 20,
    sigma: str = "default",
    rho: float = 0.9,
    noise: float = 0.1,
    w: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    
    def make_covariance(dim: int, rho: float) -> torch.Tensor:
        if not 0 <= rho < 1:
            raise ValueError("rho must be in the range [0, 1)")
        sigma = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                sigma[i, j] = rho ** abs(i - j)
        return torch.tensor(sigma, dtype=torch.float32)
    
    if sigma == "default":
        Sigma = make_covariance(dim, rho)
    else:
        sigma_path = Path(sigma)

        if not sigma_path.exists():
            raise ValueError(
                f"sigma must be 'default' or a valid file path, but got: {sigma}"
            )

        Sigma = torch.load(sigma_path)

        if not isinstance(Sigma, torch.Tensor):
            Sigma = torch.tensor(Sigma)

        Sigma = Sigma.float()

        if Sigma.shape != (dim, dim):
            raise ValueError(
                f"Sigma must have shape ({dim}, {dim}), but got {Sigma.shape}"
            )
        
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

