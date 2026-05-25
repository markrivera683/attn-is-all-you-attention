import torch
from torch.nn import functional as F
import numpy as np


def _as_float_tensor(X: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return X.detach().float()
    return torch.as_tensor(X).float()


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float) -> torch.Tensor:
    return numerator / denominator.clamp_min(eps)


def matrix_cosine_similarity(
    A: torch.Tensor | np.ndarray,
    B: torch.Tensor | np.ndarray
) -> float:
    a = _as_float_tensor(A).reshape(-1)
    b = _as_float_tensor(B).reshape(-1)

    return F.cosine_similarity(a, b, dim=0).item()


def condition_number(A: torch.Tensor, eps: float = 1e-8) -> float:
    singular = torch.linalg.svdvals(A)
    return (singular.max() / singular.min().clamp_min(eps)).item()


def check_covariance_matrix(X: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(dim=0, keepdim=True)
    N = X.shape[0]
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


def best_scale(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> float:
    """Return alpha minimizing ||A - alpha * B||_F."""
    assert A.shape == B.shape, "A and B must have the same shape"
    numerator = torch.sum(A * B)
    denominator = torch.sum(B * B).clamp_min(eps)
    return (numerator / denominator).item()


def scaled_relative_error(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> float:
    """Relative Frobenius error after optimally scaling B to match A."""
    assert A.shape == B.shape, "A and B must have the same shape"
    alpha = torch.as_tensor(best_scale(A, B, eps), device=A.device, dtype=A.dtype)
    reference = alpha * B
    return _safe_divide(torch.norm(A - reference, p="fro"), torch.norm(reference, p="fro"), eps).item()


def symmetric_part(M: torch.Tensor) -> torch.Tensor:
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be a square matrix"
    return 0.5 * (M + M.T)


def symmetry_error(M: torch.Tensor, eps: float = 1e-8) -> float:
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be a square matrix"
    return _safe_divide(torch.norm(M - M.T, p="fro"), torch.norm(M, p="fro"), eps).item()


def spectrum_stats(M: torch.Tensor, *, symmetric: bool = False, eps: float = 1e-8) -> dict[str, float]:
    """Summarize eigenvalues; for non-symmetric matrices uses eigenvalue magnitudes."""
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be a square matrix"
    if symmetric:
        eigvals = torch.linalg.eigvalsh(M).float()
        magnitudes = eigvals.abs()
        real_parts = eigvals
    else:
        eigvals = torch.linalg.eigvals(M)
        magnitudes = eigvals.abs().float()
        real_parts = eigvals.real.float()

    negative = real_parts < 0
    abs_sum = magnitudes.sum().clamp_min(eps)

    return {
        "eig_real_min": real_parts.min().item(),
        "eig_real_max": real_parts.max().item(),
        "eig_abs_min": magnitudes.min().item(),
        "eig_abs_max": magnitudes.max().item(),
        "eig_abs_condition": (magnitudes.max() / magnitudes.min().clamp_min(eps)).item(),
        "negative_eig_fraction": negative.float().mean().item(),
        "negative_eig_mass": (real_parts[negative].abs().sum() / abs_sum).item(),
    }


def spd_stats(M: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """Diagnose whether the symmetric part of M behaves like an SPD precision matrix."""
    M_sym = symmetric_part(M)
    stats = spectrum_stats(M_sym, symmetric=True, eps=eps)
    eigvals = torch.linalg.eigvalsh(M_sym).float()
    positive = eigvals[eigvals > eps]
    stats["symmetry_error"] = symmetry_error(M, eps)
    stats["is_positive_definite"] = float(bool(torch.all(eigvals > eps).item()))
    if positive.numel() > 0:
        stats["positive_condition"] = (positive.max() / positive.min().clamp_min(eps)).item()
    else:
        stats["positive_condition"] = float("inf")
    return stats


def whitening_residual(
    M: torch.Tensor,
    Sigma: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Measure whether M acts like a scaled precision matrix for Sigma."""
    assert M.shape == Sigma.shape, "M and Sigma must have the same shape"
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "M and Sigma must be square"
    I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    left = M @ Sigma
    right = Sigma @ M
    left_scale = torch.trace(left) / M.shape[0]
    right_scale = torch.trace(right) / M.shape[0]
    left_ref = left_scale * I
    right_ref = right_scale * I
    return {
        "left_whitening_residual": _safe_divide(
            torch.norm(left - left_ref, p="fro"),
            torch.norm(left_ref, p="fro"),
            eps,
        ).item(),
        "right_whitening_residual": _safe_divide(
            torch.norm(right - right_ref, p="fro"),
            torch.norm(right_ref, p="fro"),
            eps,
        ).item(),
        "left_whitening_scale": left_scale.item(),
        "right_whitening_scale": right_scale.item(),
    }


def band_energy_ratio(M: torch.Tensor, bandwidth: int) -> float:
    """Fraction of Frobenius energy within |i - j| <= bandwidth."""
    assert M.ndim == 2 and M.shape[0] == M.shape[1], "M must be a square matrix"
    idx = torch.arange(M.shape[0], device=M.device)
    band_mask = (idx[:, None] - idx[None, :]).abs() <= bandwidth
    total_energy = torch.sum(M * M).clamp_min(1e-8)
    band_energy = torch.sum(M[band_mask] * M[band_mask])
    return (band_energy / total_energy).item()


def rowwise_cosine(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert A.shape == B.shape, "A and B must have the same shape"
    A_work = row_center(A) if center else A
    B_work = row_center(B) if center else B
    numerator = torch.sum(A_work * B_work, dim=-1)
    denominator = torch.norm(A_work, dim=-1) * torch.norm(B_work, dim=-1)
    return _safe_divide(numerator, denominator, eps)


def rowwise_cosine_stats(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-8,
) -> dict[str, float]:
    values = rowwise_cosine(A, B, center=center, eps=eps)
    return {
        "rowwise_cosine_mean": values.mean().item(),
        "rowwise_cosine_std": values.std(unbiased=False).item(),
        "rowwise_cosine_min": values.min().item(),
        "rowwise_cosine_max": values.max().item(),
    }


def logit_scale_stats(
    G: torch.Tensor,
    G_ref: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-8,
) -> dict[str, float]:
    assert G.shape == G_ref.shape, "G and G_ref must have the same shape"
    G_work = row_center(G) if center else G
    G_ref_work = row_center(G_ref) if center else G_ref
    numerator = torch.sum(G_work * G_ref_work, dim=-1)
    denominator = torch.sum(G_ref_work * G_ref_work, dim=-1).clamp_min(eps)
    alpha = numerator / denominator
    fitted = alpha[:, None] * G_ref_work
    rel_error = _safe_divide(
        torch.norm(G_work - fitted, dim=-1),
        torch.norm(fitted, dim=-1),
        eps,
    )
    return {
        "logit_scale_alpha_mean": alpha.mean().item(),
        "logit_scale_alpha_std": alpha.std(unbiased=False).item(),
        "logit_scaled_error_mean": rel_error.mean().item(),
        "logit_scaled_error_std": rel_error.std(unbiased=False).item(),
    }


def _normalize_rows(P: torch.Tensor, eps: float) -> torch.Tensor:
    P = P.clamp_min(eps)
    return P / P.sum(dim=-1, keepdim=True).clamp_min(eps)


def kl_divergence_rows(
    P: torch.Tensor,
    Q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert P.shape == Q.shape, "P and Q must have the same shape"
    P_norm = _normalize_rows(P, eps)
    Q_norm = _normalize_rows(Q, eps)
    return torch.sum(P_norm * (torch.log(P_norm) - torch.log(Q_norm)), dim=-1)


def js_divergence_rows(
    P: torch.Tensor,
    Q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert P.shape == Q.shape, "P and Q must have the same shape"
    P_norm = _normalize_rows(P, eps)
    Q_norm = _normalize_rows(Q, eps)
    M = 0.5 * (P_norm + Q_norm)
    return 0.5 * kl_divergence_rows(P_norm, M, eps) + 0.5 * kl_divergence_rows(Q_norm, M, eps)


def total_variation_rows(
    P: torch.Tensor,
    Q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert P.shape == Q.shape, "P and Q must have the same shape"
    P_norm = _normalize_rows(P, eps)
    Q_norm = _normalize_rows(Q, eps)
    return 0.5 * torch.sum((P_norm - Q_norm).abs(), dim=-1)


def entropy_rows(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    A_norm = _normalize_rows(A, eps)
    return -torch.sum(A_norm * torch.log(A_norm), dim=-1)


def effective_keys(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.exp(entropy_rows(A, eps))


def attention_distribution_stats(
    A: torch.Tensor,
    A_ref: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    kl_ref_model = kl_divergence_rows(A_ref, A, eps)
    kl_model_ref = kl_divergence_rows(A, A_ref, eps)
    js = js_divergence_rows(A, A_ref, eps)
    tv = total_variation_rows(A, A_ref, eps)
    entropy = entropy_rows(A, eps)
    eff_keys = effective_keys(A, eps)
    return {
        "attn_kl_ref_model_mean": kl_ref_model.mean().item(),
        "attn_kl_model_ref_mean": kl_model_ref.mean().item(),
        "attn_js_mean": js.mean().item(),
        "attn_tv_mean": tv.mean().item(),
        "attn_entropy_mean": entropy.mean().item(),
        "attn_entropy_std": entropy.std(unbiased=False).item(),
        "attn_effective_keys_mean": eff_keys.mean().item(),
        "attn_max_weight_mean": A.max(dim=-1).values.mean().item(),
    }


def topk_overlap(A: torch.Tensor, B: torch.Tensor, k: int) -> float:
    assert A.shape == B.shape, "A and B must have the same shape"
    if not 1 <= k <= A.shape[-1]:
        raise ValueError(f"k must be in [1, {A.shape[-1]}], got {k}")

    top_a = torch.topk(A, k=k, dim=-1).indices
    top_b = torch.topk(B, k=k, dim=-1).indices
    overlap = (top_a[..., :, None] == top_b[..., None, :]).any(dim=-1).float().sum(dim=-1) / k
    return overlap.mean().item()


def signed_weight_stats(W: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    negative = W < 0
    abs_W = W.abs()
    row_sum = W.sum(dim=-1)
    row_l1 = abs_W.sum(dim=-1)
    row_l2 = torch.norm(W, dim=-1)
    total_abs = abs_W.sum().clamp_min(eps)
    negative_abs = abs_W[negative].sum()

    return {
        "negative_weight_fraction": negative.float().mean().item(),
        "negative_weight_mass": (negative_abs / total_abs).item(),
        "positive_weight_mass": ((total_abs - negative_abs) / total_abs).item(),
        "row_sum_mean": row_sum.mean().item(),
        "row_sum_std": row_sum.std(unbiased=False).item(),
        "row_l1_mean": row_l1.mean().item(),
        "row_l2_mean": row_l2.mean().item(),
        "max_abs_weight_mean": abs_W.max(dim=-1).values.mean().item(),
    }