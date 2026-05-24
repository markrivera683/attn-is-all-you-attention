import torch
from torch.nn import functional as F
import numpy as np

def mse(Y_pred: torch.Tensor, Y_true: torch.Tensor) -> float:
    return F.mse_loss(Y_pred, Y_true).item()

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