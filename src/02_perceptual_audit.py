"""Characterise the gap between actor intent and listener perception.

Covers the intent->perception confusion structure, inter-rater reliability,
the effect of the acted intensity level, and a leave-one-rater-out estimate of
how well a human listener can be predicted from either label set.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from common import RAW, PROC, TABLES, EMOTIONS, EMO_NAME, QUERY_TYPE

SEED = 20240917


def fleiss_kappa(counts):
    """counts: n_items x n_categories vote counts (rows may differ in total)."""
    n = counts.sum(axis=1)
    keep = n > 1
    counts, n = counts[keep], n[keep]
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = (p_j ** 2).sum()
    return (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha_nominal(counts):
    """Nominal-scale alpha computed from per-item category counts."""
    n_u = counts.sum(axis=1)
    keep = n_u > 1
    counts, n_u = counts[keep], n_u[keep]
    n_total = n_u.sum()
    # observed coincidences
    C = counts.shape[1]
    o = np.zeros((C, C))
    for row, m in zip(counts, n_u):
        outer = np.outer(row, row) - np.diag(row)
        o += outer / (m - 1)
    n_c = o.sum(axis=1)
    D_o = (o.sum() - np.trace(o)) / o.sum()
    D_e = (n_total ** 2 - (n_c ** 2).sum()) / (n_total * (n_total - 1))
    return 1 - D_o / (D_e / 1.0) if D_e else np.nan


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(PROC / "clip_labels.csv", index_col=0)
    V = df[[f"v_{e}" for e in EMOTIONS]].to_numpy(float)
    out = {}

    # ------------------------------------------------ intent -> perception ---
    ua = df["unambiguous"].to_numpy()
    conf = pd.crosstab(df.loc[ua, "intended"], df.loc[ua, "perceived_strict"]) \
             .reindex(index=EMOTIONS, columns=EMOTIONS, fill_value=0)
    conf_row = conf.div(conf.sum(axis=1), axis=0)
    print("intended (rows) -> perceived (cols), row-normalised, "
          f"unambiguous clips only (n={int(ua.sum())}):")
    print((conf_row * 100).round(1).rename(index=EMO_NAME, columns=EMO_NAME))
    conf.to_csv(TABLES / "confusion_intent_perceived.csv")
    conf_row.to_csv(TABLES / "confusion_intent_perceived_rownorm.csv")

    # soft confusion: average vote distribution per intended class (all clips)
    soft = df.groupby("intended")[[f"p_{e}" for e in EMOTIONS]].mean()
    soft.columns = EMOTIONS
    soft = soft.reindex(index=EMOTIONS, columns=EMOTIONS)
    soft.to_csv(TABLES / "soft_confusion.csv")
    print("\nmean vote distribution per intended class (all 7442 clips):")
    print((soft * 100).round(1).rename(index=EMO_NAME, columns=EMO_NAME))

    out["neutral_attractor"] = {
        "perceived_neutral_share": float((df["perceived"] == "N").mean()),
        "intended_neutral_share": float((df["intended"] == "N").mean()),
        "nonneutral_intent_heard_as_neutral": float(
            (df.loc[df["intended"] != "N", "perceived"] == "N").mean()),
        "sad_heard_as_neutral": float(
            (df.loc[df["intended"] == "S", "perceived"] == "N").mean()),
        "mean_neutral_vote_mass_nonneutral": float(
            df.loc[df["intended"] != "N", "p_N"].mean()),
    }
    print("\nneutral attractor:", json.dumps(out["neutral_attractor"], indent=2))

    # -------------------------------------------------------- reliability ---
    # Unbiased chance that two distinct raters of the same clip agree. This is
    # the human reference for the model metric R (match with one drawn vote).
    nv = V.sum(axis=1)
    ok = nv > 1
    pair_agree = float((((V[ok] * (V[ok] - 1)).sum(axis=1)) /
                        (nv[ok] * (nv[ok] - 1))).mean())
    out["reliability"] = {
        "rater_rater_agreement": pair_agree,
        "fleiss_kappa_perceived": float(fleiss_kappa(V)),
        "krippendorff_alpha_perceived": float(krippendorff_alpha_nominal(V)),
        "mean_agreement": float(df["agreement"].mean()),
        "mean_normalised_entropy": float(df["entropy"].mean()),
        "frac_clips_majority_gt_half": float((df["agreement"] > 0.5).mean()),
    }
    print("\nreliability:", json.dumps(out["reliability"], indent=2))

    # ------------------------------------------------- acted intensity -------
    lev = {}
    for level in ["LO", "MD", "HI", "XX"]:
        m = df["intensity"] == level
        if m.sum() == 0:
            continue
        mu = m & ua
        lev[level] = {
            "n": int(m.sum()),
            "recovery": float((df.loc[mu, "perceived_strict"] ==
                               df.loc[mu, "intended"]).mean()),
            "agreement": float(df.loc[m, "agreement"].mean()),
            "entropy": float(df.loc[m, "entropy"].mean()),
        }
    out["by_intensity"] = lev
    print("\nby acted intensity:", json.dumps(lev, indent=2))

    # ------------------------------------- leave-one-rater-out human ceiling -
    resp = pd.read_csv(RAW / "finishedResponses.csv", index_col=0,
                       low_memory=False)
    resp = resp[(resp["queryType"] == 1) & resp["respEmo"].isin(EMOTIONS)]
    resp = resp[resp["clipName"].isin(df.index)]
    idx = {e: i for i, e in enumerate(EMOTIONS)}

    tally = defaultdict(lambda: np.zeros(len(EMOTIONS)))
    for clip, e in zip(resp["clipName"].to_numpy(), resp["respEmo"].to_numpy()):
        tally[clip][idx[e]] += 1

    # Independent recomputation from the raw ratings. The shipped aggregation
    # drops some responses, so this is a check that the headline figure does
    # not depend on which of the two files one reads.
    clips = sorted(tally)
    Vr = np.stack([tally[c] for c in clips])
    intended_r = np.array([df.loc[c, "intended"] for c in clips])
    top_r = Vr.max(1)
    ua_r = (Vr == top_r[:, None]).sum(1) == 1
    perc_r = np.array(EMOTIONS)[Vr.argmax(1)]
    out["raw_recount"] = {
        "n_ratings": int(Vr.sum()),
        "extra_vs_shipped": int(Vr.sum() - V.sum()),
        "recovery_unambiguous": float((perc_r[ua_r] == intended_r[ua_r]).mean()),
        "sad_recovery": float(
            (perc_r[ua_r & (intended_r == "S")] == "S").mean()),
    }
    print("\nrecount from raw responses:", json.dumps(out["raw_recount"], indent=2))

    intended = df["intended"].to_dict()
    hits_int, hits_per, n_eval = 0, 0, 0
    for clip, e in zip(resp["clipName"].to_numpy(), resp["respEmo"].to_numpy()):
        row = tally[clip].copy()
        row[idx[e]] -= 1                     # hold this rater out
        if row.sum() < 2:
            continue
        win = np.flatnonzero(row == row.max())
        if len(win) > 1:                     # no unambiguous peer consensus
            continue
        n_eval += 1
        hits_int += (e == intended[clip])
        hits_per += (idx[e] == win[0])
    out["human_ceiling"] = {
        "n_ratings_scored": int(n_eval),
        "rater_vs_intended": hits_int / n_eval,
        "rater_vs_peer_plurality": hits_per / n_eval,
    }
    print("\nleave-one-rater-out:", json.dumps(out["human_ceiling"], indent=2))

    # -------------------------------- cross-modal check: face & multimodal ---
    tv = pd.read_csv(RAW / "tabulatedVotes.csv", index_col=0)
    cross = {}
    for q, name in QUERY_TYPE.items():
        sub = tv[tv.index.astype(str).str.startswith(str(q))].set_index("fileName")
        sub = sub.reindex(df.index)
        Vq = sub[EMOTIONS].to_numpy(float)
        topq = Vq.max(axis=1)
        uq = (Vq == topq[:, None]).sum(axis=1) == 1
        pq = np.array(EMOTIONS)[Vq.argmax(axis=1)]
        cross[name] = {
            "recovery_unambiguous": float((pq[uq] == df["intended"].to_numpy()[uq]).mean()),
            "mean_agreement": float((topq / Vq.sum(axis=1)).mean()),
            "sad_recovery": float((pq[uq & (df["intended"] == "S").to_numpy()] == "S").mean()),
        }
    out["by_modality"] = cross
    print("\nby presentation modality:", json.dumps(cross, indent=2))

    (TABLES / "perceptual_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {TABLES/'perceptual_audit.json'}")


if __name__ == "__main__":
    main()
