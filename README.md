# Whose Label? Actor Intent vs. Listener Perception in CREMA-D

Paper: [Whose Label? Actor Intent Versus Listener Perception in CREMA-D Speech Emotion Recognition](paper/abhiram_icassp2027_preprint.pdf)
Abhiram Radhakrishnan, Gengaraj P, Praveen V. Submitted to IEEE ICASSP 2027 (under review).

Code and derived label sets for a paper on the gap between the emotion CREMA-D actors were asked to portray (the filename label) and the emotion listeners report hearing (the corpus's voice-only perceptual ratings).

To reproduce everything (CPU only; downloads data, runs all experiments, checks the numbers and builds the PDF):

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt && bash scripts/run_all.sh
```

## Summary

Most CREMA-D work scores against the filename label. Using the per-rater voice-only votes included with the corpus:

| Measure | Value |
|---|---|
| Listeners recover the intended label | 45.6% of clips |
| ... for sad | 18.2% (chance = 16.7%) |
| Sad-intended clips heard as neutral | 71.8% |
| Perceived label is neutral | 56.6% of the corpus (intended: 14.6%) |
| Held-out rater agrees with the actor | 40.2% |
| Held-out rater agrees with other raters | 58.2% |
| Fleiss' κ / Krippendorff's α (voice) | 0.28 / 0.28 |
| Recovery: voice / face / audio-visual | 45.6% / 70.7% / 76.5% |

The perceived label is about 18 points more predictable than the intended one, from the same audio and the same raters.

Eight SER systems (MFCC / GeMAPS-style functionals / frozen WavLM Base+, combined with logistic regression, RBF, gradient boosting and an MLP), trained on each label set with speaker-independent 5-fold CV over 91 actors:

| Measure | Value |
|---|---|
| Best system, trained and scored on intended | 73.5% accuracy |
| Same predictions scored on perceived | 43.8% |
| Ranking of systems trained the same way, across both keys | τ = 0.93 |
| Ranking of all 24 entries (system × training label) | τ = 0.58, 58/276 pairs flip |
| Model recovers acted sad (intended key) | 66.6%, listeners get 18.2% |
| Best perceived-trained system vs a random listener | 51.6%, two listeners agree 46.5% |

So the intended label isn't just a noisier version of the target. A simple classifier picks up the actor's instruction from the audio much better than listeners do, so it is learnable but not related to what people hear. If the training label is fixed, both keys agree on which system is better. A table that mixes the two conventions (like a CREMA-D leaderboard) does not.

## Layout

```
src/
  common.py              paths, emotion codes, filename parsing
  01_build_labels.py     intended / perceived / soft labels -> data/processed
  02_perceptual_audit.py confusions, reliability, human ceiling
  03_extract_features.py MFCC + GeMAPS-style functionals (multiprocess)
  03b_extract_ssl.py     frozen WavLM Base+ mean-pooled embeddings (resumable,
                         also saves all 13 layers to ssl_layers.npz)
  models.py              the eight systems, soft-target training
  04_train_eval.py       speaker-independent 5-fold CV, 3 label regimes
  05_ranking_analysis.py bootstrap CIs, Kendall tau, rank flips
  06_figures.py          paper figures
  07_make_tables.py      LaTeX tables and \newcommand macros for the paper
  08_summary.py          RESULTS.md with the source of every number
  09_verify.py           checks: main numbers, valid posteriors,
                         speaker-independent folds, paper files present
paper/                   main.tex + ICASSP 2027 kit (spconf.sty, IEEEbib.bst)
results/                 tables (csv/json) and figures (pdf)
data/raw/                CREMA-D annotation CSVs (downloaded)
```

## Running

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
bash scripts/run_all.sh            # everything, CPU only
bash scripts/run_all.sh --labels   # label audit only, no audio needed
```

`run_all.sh` downloads the data, builds labels, extracts features, runs the experiments, writes tables and figures, checks the outputs and builds the PDF. Step by step:

Annotation CSVs (~45 MB, enough for the label audit):

```bash
bash scripts/get_data.sh
```

Then:

```bash
cd src
python 01_build_labels.py       # checks the 45.6% / 18.2% numbers
python 02_perceptual_audit.py
```

The model experiments also need the audio, which is in Git LFS in the CREMA-D repo:

```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git data/cremad_audio_repo
cd data/cremad_audio_repo && git sparse-checkout init --cone && git sparse-checkout set AudioWAV && git checkout master
```

Then run `03_extract_features.py`, `03b_extract_ssl.py`, `04_train_eval.py`, `05_ranking_analysis.py`, `06_figures.py`, `07_make_tables.py`. All CPU; WavLM is the slow part (~30 min on 16 cores).

## Labels

Each clip gets three targets over the six classes `{A, D, F, H, N, S}`:

- `intended`: the filename class, i.e. what the actor was told to portray.
- `perceived`: plurality of the voice-only votes. Ties (8.7% of clips) go to the tied class with the higher mean rated intensity. `01_build_labels.py` also writes a random-tiebreak and a strict version, and the main number is reported for each.
- `soft`: the normalised vote distribution.

All three are saved to `data/processed/clip_labels.csv`.

## Data

CREMA-D: Cao et al., IEEE Trans. Affective Computing 5(4):377-390, 2014, Open Database License. No CREMA-D data is included here; the scripts download it.

## License

Code is MIT (see [LICENSE](LICENSE)). Datasets keep their own terms and the paper PDF is not covered.
