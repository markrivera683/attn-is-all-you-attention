import math
import torch
from torch import nn
from torch.nn import functional as F


class DotProdAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._M = None
        self._G = None
        self.Attn = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], \
            "query and key must have the same feature dimension"

        self._M = None
        self._G = query @ key.T
        self.Attn = F.softmax(self._G, dim=-1)

        return self.Attn @ value

    @property
    def M(self):
        return self._M

    @property
    def G(self):
        return self._G


class WhitenedDotProdAttention(nn.Module):
    def __init__(self, dim: int, lambda_reg: float) -> None:
        super().__init__()
        self.dim = dim
        self.lambda_reg = lambda_reg
        self._M = None
        self._G = None
        self.Attn = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], \
            "query and key must have the same feature dimension"

        I = torch.eye(
            key.shape[-1],
            device=key.device,
            dtype=key.dtype,
        )

        S = key.T @ key + self.lambda_reg * I

        # Equivalent to inverse(S), but numerically more stable.
        S_inv = torch.linalg.solve(S, I)

        self._M = S_inv
        self._G = query @ S_inv @ key.T
        self.Attn = F.softmax(self._G, dim=-1)

        return self.Attn @ value

    @property
    def M(self):
        return self._M

    @property
    def G(self):
        return self._G


class LinearRegression(nn.Module):
    def __init__(self, dim: int, lambda_reg: float) -> None:
        super().__init__()
        self.dim = dim
        self.lambda_reg = lambda_reg
        self._M = None
        self._G = None
        self.Attn = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], \
            "query and key must have the same feature dimension"

        I = torch.eye(
            key.shape[-1],
            device=key.device,
            dtype=key.dtype,
        )

        S = key.T @ key + self.lambda_reg * I

        # Equivalent to inverse(S), but numerically more stable.
        S_inv = torch.linalg.solve(S, I)

        self._M = S_inv
        self._G = query @ S_inv @ key.T

        # Important:
        # For linear regression, this is not a probability attention matrix.
        # It may contain negative values and rows do not necessarily sum to 1.
        self.Attn = self._G

        return self.Attn @ value

    @property
    def M(self):
        return self._M

    @property
    def G(self):
        return self._G


class LearnBilinear(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self._M = None
        self._G = None
        self.Attn = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], \
            "query and key must have the same feature dimension"

        self._M = self.W
        self._G = query @ self.W @ key.T
        self.Attn = F.softmax(self._G, dim=-1)

        return self.Attn @ value

    @property
    def M(self):
        return self._M

    @property
    def G(self):
        return self._G


class Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.Wq = nn.Parameter(torch.empty(dim, dim))
        self.Wm = nn.Parameter(torch.empty(dim, dim))

        nn.init.kaiming_uniform_(self.Wq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wm, a=math.sqrt(5))

        self._M = None
        self._G = None
        self.Attn = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        assert query.shape[-1] == key.shape[-1], \
            "query and key must have the same feature dimension"

        self._M = self.Wq @ self.Wm.T
        self._G = query @ self._M @ key.T
        self.Attn = F.softmax(self._G, dim=-1)

        return self.Attn @ value

    @property
    def M(self):
        return self._M

    @property
    def G(self):
        return self._G