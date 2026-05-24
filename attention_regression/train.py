import torch
from torch import nn
from torch.nn import functional as F

from models import LearnBilinear, Attention


def sample_memory_query_batch(
        X: torch.Tensor,
        Y: torch.Tensor,
        memory_size: int,
        query_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = X.shape[0]

    idx_m = torch.randint(0, N, (memory_size,))
    idx_q = torch.randint(0, N, (query_size,))

    Xm = X[idx_m]
    Ym = Y[idx_m]
    Xq = X[idx_q]
    Yq = Y[idx_q]

    return Xm, Ym, Xq, Yq


def train_Bilinear(
        X: torch.Tensor,
        Y: torch.Tensor,
        dim: int,
        memory_size: int = 128,
        query_size: int = 64,
        steps: int = 1000,
        epoch: float = 10.0,
        lr: float = 1e-3,
        log_interval: int = 50
) -> tuple[LearnBilinear, list[float]]:
    
    model = LearnBilinear(dim)
    opti = torch.optim.Adam(model.parameters(), lr=lr)
    loss_log = []

    for step in range(steps):
        Xm, Ym, Xq, Yq = sample_memory_query_batch(X, Y, memory_size, query_size)
        Yhat = model(Xq, Xm, Ym)
        loss = F.mse_loss(Yq, Yhat)

        opti.zero_grad()
        loss.backward()
        opti.step()

        loss_value = loss.item()
        loss_log.append(loss_value)

        if step % log_interval == 0:
            print(f"step={step:04d} loss={loss_value:.6f}")

    return model, loss_log


def train_Attn(
        X: torch.Tensor,
        Y: torch.Tensor,
        dim: int,
        memory_size: int = 128,
        query_size: int = 64,
        steps: int = 1000,
        epoch: float = 10.0,
        lr: float = 1e-3,
        log_interval: int = 50
) -> tuple[Attention, list[float]]:
    
    model = Attention(dim)
    opti = torch.optim.Adam(model.parameters(), lr=lr)
    loss_log = []

    for step in range(steps):
        Xm, Ym, Xq, Yq = sample_memory_query_batch(X, Y, memory_size, query_size)
        Yhat = model(Xq, Xm, Ym)
        loss = F.mse_loss(Yq, Yhat)

        opti.zero_grad()
        loss.backward()
        opti.step()

        loss_value = loss.item()
        loss_log.append(loss_value)

        if step % log_interval == 0:
            print(f"step={step:04d} loss={loss_value:.6f}")

    return model, loss_log