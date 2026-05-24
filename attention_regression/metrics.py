import torch
from torch.nn import functional as F
import numpy as np

def matrix_cosine_similarity(
    A: torch.Tensor | np.ndarray,
    B: torch.Tensor | np.ndarray
) -> float:
    if isinstance(A, torch.Tensor):
        a = A.detach().reshape(-1)
    else:
        a = torch.as_tensor(A).reshape(-1)

    if isinstance(B, torch.Tensor):
        b = B.detach().reshape(-1)
    else:
        b = torch.as_tensor(B).reshape(-1)

    a = a.float()
    b = b.float()

    return F.cosine_similarity(a, b, dim=0).item()

def condition_number(A: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(A)
    return (singular.max() / singular.min()).item()

def check_covariance_matrix(X: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(dim=1)
    N = X.shape[1]
    return (X.T @ X) / N

def row_center(M: torch.Tensor) -> torch.Tensor:
    return M - M.mean(dim=-1, keepdim=True)

def row_standardize(M: torch.Tensor) -> torch.Tensor:
    return (M - M.mean(dim=-1, keepdim=True)) / (M.norm(dim=-1, keepdim=True) + 1e-8)

def col_standardize(M: torch.Tensor) -> torch.Tensor:
    return (M - M.mean(dim=-2, keepdim=True)) / (M.norm(dim=-2, keepdim=True) + 1e-8)

def frobenius_cosine(M: torch.Tensor, G: torch.Tensor) -> float:
    assert M.shape == G.shape, "M and G must have the same shape"
    similarity = torch.sum(M * G)
    scaler = torch.norm(M, p="fro") * torch.norm(G, p="fro") + 1e-8
    return (similarity / scaler).item()