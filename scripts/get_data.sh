#!/usr/bin/env bash
# Fetch the CREMA-D annotation files used by the label audit (no audio, ~45 MB).
set -euo pipefail
BASE="https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master"
mkdir -p data/raw
for f in processedResults/summaryTable.csv processedResults/tabulatedVotes.csv \
         finishedResponses.csv finishedEmoResponses.csv \
         VideoDemographics.csv SentenceFilenames.csv; do
  echo "fetching $f"
  curl -sSL -o "data/raw/$(basename "$f")" "$BASE/$f"
done
ls -la data/raw
