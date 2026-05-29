#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python main.py \
    --train_use_wandb

# Optional:
# uv run python main.py --signed_weight_decay 0.0
