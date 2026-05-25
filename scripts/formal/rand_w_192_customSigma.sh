#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SIGMA_PATH="scripts/utils/sigma_complex.pt"

echo "Running with custom sigma from: ${SIGMA_PATH}"
echo "The custom sigma is:"

uv run python - <<PY
import torch

sigma = torch.load("${SIGMA_PATH}", map_location="cpu")
print(sigma)
print("shape:", tuple(sigma.shape))
print("min eigenvalue:", torch.linalg.eigvalsh(sigma).min().item())
print("max eigenvalue:", torch.linalg.eigvalsh(sigma).max().item())
PY

uv run python main.py \
    --sigma ${SIGMA_PATH} \
    --train_use_wandb \
    # --signed_weight_decay 0.0