#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

for rho in 0.2 0.6 0.9; do
  for noise in 0.0 0.1 0.5; do
    for memory_size in 16 64 256; do
      uv run python main.py \
        --data_mode episodic_w \
        --rho "${rho}" \
        --noise "${noise}" \
        --memory_size "${memory_size}" \
        --query_size 64 \
        --train_num_epochs 10 \
        --train_num_episodes 500 \
        --eval_num_episodes 500 \
        --lambda_reg 0.001 \
        --run_id "episodic-rho${rho}-noise${noise}-mem${memory_size}"
    done
  done
done
