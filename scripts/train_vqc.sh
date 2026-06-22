#!/usr/bin/env bash
# Default VQC run: p=17, n_qubits=6, L=3, weight_decay=1.0, 3 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-30000}"
for seed in 0 1 2; do
  echo "=== seed $seed ==="
  python -m src.train --seed "$seed" --steps "$STEPS" --lr 1e-2 --weight_decay 1.0
done
