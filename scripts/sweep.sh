#!/usr/bin/env bash
# n_qubits x L sweep (used when the default VQC does not grok).
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-15000}"
for nq in 4 6; do
  for L in 2 4; do
    echo "=== n_qubits=$nq L=$L ==="
    python -m src.train --seed 0 --steps "$STEPS" --lr 1e-2 \
      --weight_decay 1.0 --n_qubits "$nq" --n_layers "$L"
    mv results/vqc_seed0.csv "results/sweep_nq${nq}_L${L}.csv"
  done
done
