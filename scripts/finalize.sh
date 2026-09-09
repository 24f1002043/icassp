#!/usr/bin/env bash
# Regenerate every derived artefact and rebuild the paper. Safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export PATH="/c/Users/Abhiram/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"

cd src
echo "== ranking analysis =="
"$PY" 05_ranking_analysis.py 2>&1 | grep -viE "^\s*$" | tail -60
echo "== figures =="
"$PY" 06_figures.py
echo "== tables and macros =="
"$PY" 07_make_tables.py | tail -50
echo "== results summary =="
"$PY" 08_summary.py

cd ../paper
echo "== building the paper =="
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex >/tmp/final.log 2>&1
echo "-- LaTeX errors (none expected):"
grep -E "^! " /tmp/final.log || true
echo "-- pages: $(pdfinfo main.pdf | awk '/Pages/{print $2}')"
echo "-- size:  $(du -h main.pdf | cut -f1)"
echo "-- abstract words: $(sed -n '/begin{abstract}/,/end{abstract}/p' main.tex | sed '1d;$d' | wc -w)"

cd ../src
echo "== verification =="
"$PY" 09_verify.py
