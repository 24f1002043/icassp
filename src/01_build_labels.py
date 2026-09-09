"""Build the three clip-level label sets used throughout the study.

Sources
-------
processedResults/tabulatedVotes.csv : the aggregation shipped with CREMA-D.
                                      Rows are keyed 100000*queryType + clipNum,
                                      so queryType 1 (= voice only) are the rows
                                      whose key starts with a 1.
finishedResponses.csv               : the individual ratings behind that table,
                                      used here as an independent cross-check
                                      and for the leave-one-rater-out ceiling.

Outputs data/processed/clip_labels.csv and results/tables/label_audit.json.
"""
import json

import numpy as np
import pandas as pd

from common import RAW, PROC, TABLES, EMOTIONS, EMO_NAME, parse_filename

SEED = 20240917


def load_voice_votes():
    """Clip x emotion vote counts for the voice-only condition."""
    tv = pd.read_csv(RAW / "tabulatedVotes.csv", index_col=0)
    voice = tv[tv.index.astype(str).str.startswith("1")].copy()
    assert len(voice) == 7442, len(voice)
    voice = voice.set_index("fileName").sort_index()
    counts = voice[EMOTIONS].astype(int)
    # mean rated intensity per emotion; -1 marks "no rater chose this emotion"
    level_cols = {e: f"mean{EMO_NAME[e]}Resp" for e in EMOTIONS}
    levels = voice[[level_cols[e] for e in EMOTIONS]].copy()
    levels.columns = EMOTIONS
    return counts, levels, voice


def main():
    rng = np.random.default_rng(SEED)
    counts, levels, voice = load_voice_votes()

    df = pd.DataFrame(index=counts.index)
    df.index.name = "clip"
    meta = [parse_filename(c) for c in df.index]
    df["actor"] = [m[0] for m in meta]
    df["sentence"] = [m[1] for m in meta]
    df["intended"] = [m[2] for m in meta]
    df["intensity"] = [m[3] for m in meta]

    V = counts.to_numpy(float)
    n = V.sum(axis=1)
    df["n_raters"] = n.astype(int)
    for i, e in enumerate(EMOTIONS):
        df[f"v_{e}"] = V[:, i].astype(int)

    # ---- soft label: the normalised voice-only vote distribution -------------
    P = V / n[:, None]
    for i, e in enumerate(EMOTIONS):
        df[f"p_{e}"] = P[:, i]

    top = V.max(axis=1)
    n_winners = (V == top[:, None]).sum(axis=1)
    df["tied"] = n_winners > 1
    df["unambiguous"] = n_winners == 1
    df["agreement"] = top / n

    # ---- hard perceived label ----------------------------------------------
    # Ties (8.7% of clips) are resolved in favour of the co-winning emotion that
    # received the higher mean rated intensity; that is a property of the votes
    # themselves rather than an arbitrary ordering.  A seeded random draw is the
    # fallback when the intensities are also equal.
    L = levels.to_numpy(float)
    perceived, tie_random = [], []
    for row, lev, k in zip(V, L, n_winners):
        win = np.flatnonzero(row == row.max())
        if k == 1:
            perceived.append(EMOTIONS[win[0]])
            tie_random.append(EMOTIONS[win[0]])
        else:
            best = win[np.argmax(lev[win])]
            if (lev[win] == lev[win].max()).sum() > 1:
                best = rng.choice(win[lev[win] == lev[win].max()])
            perceived.append(EMOTIONS[best])
            tie_random.append(EMOTIONS[rng.choice(win)])
    df["perceived"] = perceived
    df["perceived_randtie"] = tie_random
    df["perceived_strict"] = np.where(df["tied"], "TIE",
                                      np.array(EMOTIONS)[V.argmax(axis=1)])

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(P * np.where(P > 0, np.log(P), 0.0)).sum(axis=1)
    df["entropy"] = ent / np.log(len(EMOTIONS))
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    df["p_intended"] = P[np.arange(len(df)), [idx[e] for e in df["intended"]]]

    # ---- recovery of the intended label under several conventions -----------
    ua = df["unambiguous"].to_numpy()
    hit_strict = (df["perceived_strict"] == df["intended"]).to_numpy()
    inwin = np.array([df["intended"].iloc[i] in
                      [EMOTIONS[j] for j in np.flatnonzero(V[i] == V[i].max())]
                      for i in range(len(df))])
    conventions = {
        "unambiguous_only": float(hit_strict[ua].mean()),
        "tie_counts_as_miss": float(hit_strict.mean()),
        "tie_broken_by_intensity": float((df["perceived"] == df["intended"]).mean()),
        "tie_broken_at_random": float((df["perceived_randtie"] == df["intended"]).mean()),
        "tie_counts_as_hit": float(inwin.mean()),
        "mean_vote_mass_on_intended": float(df["p_intended"].mean()),
    }

    per_emotion = {}
    for e in EMOTIONS:
        m = (df["intended"] == e).to_numpy()
        per_emotion[e] = {
            "n": int(m.sum()),
            "unambiguous_only": float(hit_strict[m & ua].mean()),
            "tie_broken_by_intensity": float((df.loc[m, "perceived"] == e).mean()),
            "tie_counts_as_hit": float(inwin[m].mean()),
            "mean_vote_mass": float(df.loc[m, "p_intended"].mean()),
            "mean_agreement": float(df.loc[m, "agreement"].mean()),
            "mean_entropy": float(df.loc[m, "entropy"].mean()),
        }

    print(f"clips: {len(df)}  ratings: {int(n.sum())}  "
          f"mean raters/clip: {n.mean():.2f}")
    print("\nrecovery of the intended label (voice only):")
    for k, v in conventions.items():
        print(f"  {k:28s} {v:.4f}")
    print(f"\n  chance (uniform)            {1/6:.4f}")
    print("\nper intended emotion (unambiguous-plurality convention):")
    for e in EMOTIONS:
        d = per_emotion[e]
        print(f"  {EMO_NAME[e]:8s} n={d['n']:5d}  recovery={d['unambiguous_only']:.3f}"
              f"  agreement={d['mean_agreement']:.3f}  H={d['mean_entropy']:.3f}")

    df.to_csv(PROC / "clip_labels.csv")

    audit = {
        "n_clips": int(len(df)),
        "n_ratings": int(n.sum()),
        "mean_raters_per_clip": float(n.mean()),
        "min_raters_per_clip": int(n.min()),
        "tied_fraction": float(df["tied"].mean()),
        "n_unambiguous": int(ua.sum()),
        "chance_uniform": 1 / 6,
        "recovery": conventions,
        "per_emotion": per_emotion,
        "mean_agreement": float(df["agreement"].mean()),
        "mean_entropy": float(df["entropy"].mean()),
        "intended_counts": df["intended"].value_counts().to_dict(),
        "perceived_counts": df["perceived"].value_counts().to_dict(),
        "n_actors": int(df["actor"].nunique()),
    }
    (TABLES / "label_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nwrote {PROC/'clip_labels.csv'}")
    print(f"wrote {TABLES/'label_audit.json'}")


if __name__ == "__main__":
    main()
