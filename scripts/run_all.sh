#!/usr/bin/env bash
# Reproduce the whole study from a clean checkout.
#
#   bash scripts/run_all.sh            everything, including the audio pipeline
#   bash scripts/run_all.sh --labels   only the parts that need no audio
#
# Everything runs on CPU. The WavLM pass is the slow step (~30-60 min on 16
# cores) and is resumable: it writes shards under data/features/ssl_shards.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
[ -x "$PY" ] || PY=python

echo "=== 0. annotation CSVs (~45 MB) ==="
[ -f data/raw/tabulatedVotes.csv ] || bash scripts/get_data.sh

cd src
echo "=== 1. label sets ==="
"../$PY" 01_build_labels.py
echo "=== 2. perceptual audit ==="
"../$PY" 02_perceptual_audit.py

if [ "${1:-}" = "--labels" ]; then
  "../$PY" 07_make_tables.py --audit-only
  "../$PY" 08_summary.py
  echo "label-only run complete; see RESULTS.md"
  exit 0
fi

cd ..
if [ ! -d data/cremad_audio_repo/AudioWAV ]; then
  echo "=== fetching audio via Git LFS (~600 MB) ==="
  git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/CheyneyComputerScience/CREMA-D.git data/cremad_audio_repo
  git -C data/cremad_audio_repo sparse-checkout init --cone
  git -C data/cremad_audio_repo sparse-checkout set AudioWAV
  git -C data/cremad_audio_repo checkout master
fi

cd src
echo "=== 3. acoustic features ==="
[ -f ../data/features/funcs.npz ] || "../$PY" 03_extract_features.py
echo "=== 3b. WavLM features (resumable) ==="
[ -f ../data/features/ssl.npz ] || "../$PY" 03b_extract_ssl.py
echo "=== 4. train and evaluate ==="
"../$PY" 04_train_eval.py
echo "=== 5. ranking analysis ==="
"../$PY" 05_ranking_analysis.py
echo "=== 6-8. figures, tables, summary ==="
"../$PY" 06_figures.py
"../$PY" 07_make_tables.py
"../$PY" 08_summary.py
echo "=== 9. verification ==="
"../$PY" 09_verify.py

cd ../paper
echo "=== 10. build the paper ==="
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "wrote paper/main.pdf"
