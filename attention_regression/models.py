import math
import torch
from torch import nn
from torch.nn import functional as F


class DotProdAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
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
        self._G = query @ key.T / math.sqrt(self.dim)
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
        self._G = query @ S_inv @ key.T / math.sqrt(self.dim)
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

        # Keep this as a plain tensor cache; assigning the Parameter itself
        # would register `_M` into state_dict as a duplicate parameter.
        self._M = self.W.detach()
        self._G = query @ self.W @ key.T / math.sqrt(self.dim)
        self.Attn = F.softmax(self._G, dim=-1)

        return self.Attn @ value

    @property
    def M(self):
        return self._M

    @property
    def G(self):
        return self._G


class LearnSignedBilinear(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.W = nn.Parameter(torch.zeros(dim, dim))

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

        M = 0.5 * (self.W + self.W.T)
        self._M = M.detach()
        self._G = query @ M @ key.T

        # This is a signed linear weight matrix, not probability attention.
        # It can represent subtractive regression-style memory weights.
        self.Attn = self._G

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
        self._G = query @ self._M @ key.T / math.sqrt(self.dim)
        self.Attn = F.softmax(self._G, dim=-1)

        return self.Attn @ value

    @property
    def M(self):
        return self._M

    @property
    def G(self):
        return self._G