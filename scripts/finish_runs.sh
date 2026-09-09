#!/usr/bin/env bash
# Run the non-SSL sweep, wait for the WavLM features, then finish everything.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
cd src

echo "[run] non-SSL systems"
"$PY" 04_train_eval.py MFCC-LR MFCC-RBF FUNC-LR FUNC-RBF FUNC-GB FUNC-MLP 2>&1 \
  | grep -viE "warning|warn\("

echo "[wait] for data/features/ssl.npz"
until [ -f ../data/features/ssl.npz ]; do sleep 20; done
sleep 10
echo "[ok] ssl.npz present"

echo "[run] SSL systems"
"$PY" 04_train_eval.py 2>&1 | grep -viE "warning|warn\("

echo "[run] ranking analysis"
"$PY" 05_ranking_analysis.py 2>&1 | grep -viE "^\s*$" | tail -70

echo "[run] figures + tables"
"$PY" 06_figures.py 2>&1 | tail -5
"$PY" 07_make_tables.py 2>&1 | tail -45
echo "[done]"
