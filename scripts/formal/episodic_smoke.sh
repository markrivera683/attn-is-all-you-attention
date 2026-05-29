#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

uv run python main.py \
    --data_mode episodic_w \
    --device cpu \
    --dim 4 \
    --memory_size 8 \
    --query_size 4 \
    --train_num_epochs 1 \
    --train_num_episodes 8 \
    --eval_num_episodes 8 \
    --log_interval 1 \
    --run_id episodic-smoke
