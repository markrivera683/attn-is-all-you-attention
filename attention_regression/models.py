import torch
from torch import nn
from torch.nn import functional as F

import math
    
class DotProdAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], "All inputs must have the same feature dimension"

        G = query @ key.T
        Attn = F.softmax(G, dim=-1)
        return Attn @ value


class WhitenedDotProdAttention(nn.Module):
    def __init__(self, dim: int, lambda_reg: float) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.M, self.G, self.Attn = None, None, None
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], "All inputs must have the same feature dimension"
        S = key.T @ key + self.lambda_reg * torch.eye(key.shape[-1])
        S_inv = torch.linalg.inv(S)
        self.M = S_inv
        self.G = query @ S_inv @ key.T
        self.Attn = F.softmax(self.G, dim=-1)
        return self.Attn @ value


class LinearRegression(nn.Module):
    def __init__(self, dim: int, lambda_reg: float) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.M, self.G, self.Attn = None, None, None

    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], "All inputs must have the same feature dimension"
        S = key.T @ key + self.lambda_reg * torch.eye(key.shape[-1])
        S_inv = torch.linalg.inv(S)
        self.M = S_inv
        self.G = query @ S_inv @ key.T
        self.Attn = self.G
        return self.Attn @ value


class LearnBilinear(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.M, self.G, self.Attn = None, None, None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], "All inputs must have the same feature dimension"
        self.M = self.W
        self.G = query @ self.W @ key.T
        self.Attn = F.softmax(self.G, dim = -1)
        return self.Attn @ value


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Wq = nn.Parameter(torch.empty(dim, dim))
        self.Wm = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_uniform_(self.Wq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wm, a=math.sqrt(5))
        self.M, self.G, self.Attn = None, None, None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], "All inputs must have the same feature dimension"
        self.M = self.Wq @ self.Wm.T
        self.G = query @ self.M @ key.T
        self.Attn = F.softmax(self.G, dim=-1)
        return self.Attn @ value