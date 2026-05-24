import torch

def make_complex_sigma(
    dim: int = 20,
    seed: int = 43,
    eps: float = 1e-3,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)

    # 1. base diagonal variance
    sigma = torch.eye(dim)

    # 2. block correlation structure
    blocks = [
        (0, 5, 0.75),
        (5, 10, 0.55),
        (10, 15, 0.35),
        (15, 20, 0.65),
    ]

    for start, end, corr in blocks:
        size = end - start
        block = torch.full((size, size), corr)
        block.fill_diagonal_(1.0)
        sigma[start:end, start:end] = block

    # 3. AR-style local decay correlation
    idx = torch.arange(dim)
    ar = 0.35 ** (idx[:, None] - idx[None, :]).abs()
    sigma = 0.65 * sigma + 0.35 * ar

    # 4. periodic correlation pattern
    distance = (idx[:, None] - idx[None, :]).abs().float()
    periodic = 0.12 * torch.cos(2.0 * torch.pi * distance / 6.0)
    sigma = sigma + periodic

    # 5. low-rank global latent factors
    rank = 3
    factors = torch.randn(dim, rank, generator=g)
    low_rank = factors @ factors.T
    low_rank = low_rank / low_rank.diag().sqrt().outer(low_rank.diag().sqrt())
    sigma = sigma + 0.18 * low_rank

    # 6. sparse cross-block interactions
    sparse_pairs = [
        (0, 12, 0.28),
        (1, 17, -0.22),
        (3, 14, 0.18),
        (4, 9, -0.20),
        (6, 16, 0.25),
        (8, 19, -0.18),
        (11, 18, 0.20),
    ]

    for i, j, value in sparse_pairs:
        sigma[i, j] += value
        sigma[j, i] += value

    # 7. symmetrize
    sigma = 0.5 * (sigma + sigma.T)

    # 8. make positive definite by eigenvalue clipping
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    eigvals = torch.clamp(eigvals, min=eps)
    sigma = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    # 9. convert covariance matrix to correlation matrix
    std = torch.sqrt(torch.diag(sigma))
    sigma = sigma / std[:, None] / std[None, :]

    # 10. final numerical cleanup
    sigma = 0.5 * (sigma + sigma.T)
    sigma.fill_diagonal_(1.0)

    return sigma.float()


sigma = make_complex_sigma(dim=20, seed=43)

print("sigma shape:", sigma.shape)
print("min eigenvalue:", torch.linalg.eigvalsh(sigma).min().item())
print("max eigenvalue:", torch.linalg.eigvalsh(sigma).max().item())
print(sigma)

torch.save(sigma, "scripts/utils/sigma_complex.pt")
print("saved to scripts/utils/sigma_complex.pt")