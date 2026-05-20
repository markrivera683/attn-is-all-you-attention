import torch
from torch import nn
from torch.nn import functional as F

from enum import Enum
import math

class AttentionMode(Enum):
    DOT = "dot"
    WHITENED = "whitened"
    LINEAR = "linear"
    BILINEAR = "bilnear"
    STD = "std"

class HandCraftAttention(nn.Module):
    def __init__(self, dim: int, mode: AttentionMode, lambda_reg: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        mode_layer = {
            AttentionMode.DOT: DotProdAttention(), 
            AttentionMode.WHITENED: WhitenedDotProdAttention(dim, lambda_reg),
            AttentionMode.LINEAR: LinearRegression(dim, lambda_reg),
            AttentionMode.BILINEAR: LearnBilinear(dim),
            AttentionMode.STD: Attention(dim)
        }

        if mode not in mode_layer:
            raise ValueError(f"Unknow mode : {mode}")
        
        self.attn = mode_layer[mode]

    def forward(self,
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        return self.attn(query, key, value)

    
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