import torch
from torch.nn import functional as F

def mse(Y_pred: torch.Tensor, Y_true: torch.Tensor) -> float:
    return F.mse_loss(Y_pred, Y_true).item()

def matrix_cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> float:
    a = A.view(-1)
    b = B.view(-1)
    return F.cosine_similarity(A, B, dim=1).item()

def condition_number(A: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(A)
    return (singular.max() / singular.min()).item()

def check_covariance_matrix(X: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(dim=1)
    N = X.shape[1]
    return (X.T @ X) / N