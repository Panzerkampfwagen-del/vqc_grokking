#!/usr/bin/env bash
# Analyse a trained run (auto-selects grok vs failure-mode analysis).
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-0}"
python -m src.analysis --seed "$SEED" --mode auto
