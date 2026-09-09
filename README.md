# Whose Label? Actor Intent vs. Listener Perception in CREMA-D

Code and derived label sets for an ICASSP submission on the gap between the
emotion CREMA-D actors were told to portray (the filename label) and the emotion
listeners actually report hearing (the corpus's own voice-only perceptual
ratings).

## The short version

Scoring against the filename label is the near-universal convention on CREMA-D.
Using the per-rater voice-only votes that ship with the corpus:

| | |
|---|---|
| Listeners recover the intended label | **45.6%** of clips |
| ... for *sad* | **18.2%** (chance = 16.7%) |
| Sad-intended clips heard as *neutral* | **71.8%** |
| Perceived label is *neutral* | 56.6% of the corpus (intended: 14.6%) |
| A held-out rater agrees with the actor | 40.2% |
| A held-out rater agrees with other raters | **58.2%** |
| Fleiss' κ / Krippendorff's α (voice) | 0.28 / 0.28 |
| Recovery: voice / face / audio-visual | 45.6% / 70.7% / 76.5% |

The perceptual target is ~18 points more predictable than the intended one, from
the same audio, by the same people.

Eight SER systems (MFCC / GeMAPS-style functionals / frozen WavLM Base+, crossed
with logistic regression, RBF, gradient boosting and an MLP), trained under each
label set with speaker-independent 5-fold CV over the 91 actors:

| | |
|---|---|
| Best system, trained and scored on **intended** | 73.5% accuracy |
| The *same predictions*, scored on **perceived** | **43.8%** |
| Ranking of systems trained alike, across the two keys | τ = 0.93 — largely preserved |
| Ranking of all 24 entries (system × training label) | **τ = 0.58**, 58/276 pairs invert |
| Model recovers acted *sad* (intended key) | **66.6%**, where listeners reach 18.2% |
| Best perceived-trained system vs. a random listener | 51.6%, above the 46.5% two listeners manage |

So the intended label is not simply the noisier target: a plain classifier reads
the actor's instruction out of the audio far better than listeners do. It is a
learnable target that has little to do with what anyone hears. Holding the
training label fixed, the two keys agree on which system is better; a table that
mixes conventions — which is what a CREMA-D leaderboard is — does not.

## Layout

```
src/
  common.py             shared paths, emotion codes, filename parsing
  01_build_labels.py    intended / perceived / soft label sets  -> data/processed
  02_perceptual_audit.py confusion structure, reliability, human ceiling
  03_extract_features.py MFCC + GeMAPS-style functionals (CPU, multiprocess)
  03b_extract_ssl.py    frozen WavLM Base+ mean-pooled embeddings (CPU,
                        resumable; also dumps all 13 layers to ssl_layers.npz)
  models.py             the eight systems; distributional-target training
  04_train_eval.py      speaker-independent 5-fold CV, 3 label regimes
  05_ranking_analysis.py bootstrap CIs, Kendall tau, rank inversions
  06_figures.py         paper figures
  07_make_tables.py     LaTeX tables + \newcommand macros for the paper
  08_summary.py         RESULTS.md, every cited number with its provenance
  09_verify.py          sanity checks: headline numbers, valid posteriors,
                        speaker-independent folds, paper artefacts present
paper/                  main.tex + official ICASSP 2027 kit (spconf.sty, IEEEbib.bst)
results/                tables (csv/json) and figures (pdf)
data/raw/               CREMA-D annotation CSVs (downloaded)
```

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
bash scripts/run_all.sh            # everything, CPU only
bash scripts/run_all.sh --labels   # just the label audit, no audio needed
```

`run_all.sh` fetches the data, builds the label sets, extracts features, runs the
sweep, writes the tables and figures, verifies the outputs and builds the PDF.
The steps individually:

Annotation CSVs (~45 MB, no audio needed for the label audit):

```bash
bash scripts/get_data.sh
```

Then:

```bash
cd src
python 01_build_labels.py       # verifies the 45.6% / 18.2% headline numbers
python 02_perceptual_audit.py
```

The modelling experiments additionally need the audio. It lives behind Git LFS
in the CREMA-D repository:

```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git data/cremad_audio_repo
cd data/cremad_audio_repo && git sparse-checkout init --cone && git sparse-checkout set AudioWAV && git checkout master
```

Then `03_extract_features.py`, `03b_extract_ssl.py`, `04_train_eval.py`,
`05_ranking_analysis.py`, `06_figures.py`, `07_make_tables.py`. Everything runs
on CPU; the WavLM pass is the slow step (~30 min on 16 cores).

## Label definitions

For every clip we derive three targets over the six categories
`{A, D, F, H, N, S}`:

* `intended` — the filename category, i.e. the direction the actor was given.
* `perceived` — the plurality of the voice-only votes. Ties (8.7% of clips)
  break towards the co-winning category with the higher mean rated intensity;
  `01_build_labels.py` also emits a random-tiebreak and a strict variant, and
  the headline number is reported under every convention.
* `soft` — the normalised vote distribution.

All three are written to `data/processed/clip_labels.csv`.

## Data

CREMA-D is by Cao et al., IEEE Trans. Affective Computing 5(4):377–390, 2014,
distributed under the Open Database License. This repository redistributes no
CREMA-D data; the scripts download it.
