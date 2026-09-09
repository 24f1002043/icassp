"""Score every stored system and test whether the answer key changes the ranking.

The central comparison holds the trained models fixed and swaps only the label
the predictions are scored against.  Confidence intervals come from a
clip-level bootstrap; the same resamples are reused for every system so the
paired differences and the rank correlations are computed on common draws.
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from common import PROC, TABLES, EMOTIONS

SEED = 20240917
N_BOOT = int(__import__("os").environ.get("N_BOOT", 2000))
REGIMES = ["intended", "perceived", "soft"]


def uar(y_true, y_pred, w=None):
    """Unweighted average recall (macro recall), optionally clip-weighted."""
    recalls = []
    for e in EMOTIONS:
        m = y_true == e
        if w is None:
            if m.sum():
                recalls.append((y_pred[m] == e).mean())
        else:
            den = w[m].sum()
            if den > 0:
                recalls.append((w[m] * (y_pred[m] == e)).sum() / den)
    return float(np.mean(recalls))


def wa(y_true, y_pred, w=None):
    if w is None:
        return float((y_true == y_pred).mean())
    return float((w * (y_true == y_pred)).sum() / w.sum())


def main():
    labels = pd.read_csv(PROC / "clip_labels.csv", index_col=0).sort_index()
    y_i = labels["intended"].to_numpy()
    y_p = labels["perceived"].to_numpy()
    S = labels[[f"p_{e}" for e in EMOTIONS]].to_numpy(float)
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    n = len(labels)

    pdir = PROC / "posteriors"
    runs = {}
    for f in sorted(pdir.glob("*.npy")):
        sysname, regime = f.stem.split("__")
        runs.setdefault(sysname, {})[regime] = np.load(f)
    systems = [s for s in runs if set(runs[s]) >= set(REGIMES)]
    systems.sort()
    print(f"{len(systems)} complete systems:", systems)

    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, n, size=(N_BOOT, n))

    rows, curves = [], {}
    for s in systems:
        for regime in REGIMES:
            P = runs[s][regime]
            pred = np.array(EMOTIONS)[P.argmax(1)]
            # probability the model's label matches a randomly drawn rater
            rater_hit = S[np.arange(n), [idx[e] for e in pred]]
            # soft cross-entropy against the vote distribution
            ce = -(S * np.log(np.clip(P, 1e-12, 1))).sum(1)
            rec = {
                "system": s, "train": regime,
                "WA_intended": wa(y_i, pred), "UAR_intended": uar(y_i, pred),
                "WA_perceived": wa(y_p, pred), "UAR_perceived": uar(y_p, pred),
                "rater_agreement": float(rater_hit.mean()),
                "soft_CE": float(ce.mean()),
            }
            rows.append(rec)
            curves[(s, regime)] = dict(pred=pred, rater_hit=rater_hit, ce=ce)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "main_results.csv", index=False)
    print("\n=== all systems x training regimes ===")
    print(res.round(4).to_string(index=False))

    # ------------------------------------------------ bootstrap for the CIs --
    def boot_stat(fn):
        return np.array([fn(b) for b in boot])

    ci = {}
    for s in systems:
        for regime in REGIMES:
            pred = curves[(s, regime)]["pred"]
            for key, y in [("intended", y_i), ("perceived", y_p)]:
                vals = boot_stat(lambda b, p=pred, y=y: uar(y[b], p[b]))
                ci[f"{s}|{regime}|UAR_{key}"] = [float(np.percentile(vals, 2.5)),
                                                 float(np.percentile(vals, 97.5))]

    # --------------------------- the ranking test: same models, two keys -----
    out = {}
    for regime in REGIMES:
        sub = res[res.train == regime].set_index("system").loc[systems]
        a = sub["UAR_intended"].to_numpy()
        b = sub["UAR_perceived"].to_numpy()
        tau, p = kendalltau(a, b)
        # bootstrap the rank correlation using the shared resamples
        taus = []
        for bi in boot[:min(500, N_BOOT)]:
            aa = [uar(y_i[bi], curves[(s, regime)]["pred"][bi]) for s in systems]
            bb = [uar(y_p[bi], curves[(s, regime)]["pred"][bi]) for s in systems]
            taus.append(kendalltau(aa, bb)[0])
        taus = np.array(taus)

        inversions = []
        for (i, s1), (j, s2) in combinations(list(enumerate(systems)), 2):
            d_i = a[i] - a[j]
            d_p = b[i] - b[j]
            if d_i == 0 or d_p == 0:      # exact tie: no order to invert
                continue
            if np.sign(d_i) != np.sign(d_p):
                p1, p2 = curves[(s1, regime)]["pred"], curves[(s2, regime)]["pred"]
                di = boot_stat(lambda bb: uar(y_i[bb], p1[bb]) - uar(y_i[bb], p2[bb]))
                dp = boot_stat(lambda bb: uar(y_p[bb], p1[bb]) - uar(y_p[bb], p2[bb]))
                inversions.append({
                    "pair": [s1, s2],
                    "delta_intended": float(d_i),
                    "delta_perceived": float(d_p),
                    "p_sign_intended": float((np.sign(di) == np.sign(d_i)).mean()),
                    "p_sign_perceived": float((np.sign(dp) == np.sign(d_p)).mean()),
                    "both_significant": bool(
                        (np.sign(di) == np.sign(d_i)).mean() > 0.975 and
                        (np.sign(dp) == np.sign(d_p)).mean() > 0.975),
                })
        out[regime] = {
            "kendall_tau": float(tau), "kendall_p": float(p),
            "tau_ci": [float(np.percentile(taus, 2.5)),
                       float(np.percentile(taus, 97.5))],
            "rank_intended": {s: int(r) for s, r in
                              zip(systems, (-a).argsort().argsort() + 1)},
            "rank_perceived": {s: int(r) for s, r in
                               zip(systems, (-b).argsort().argsort() + 1)},
            "n_pairs": len(systems) * (len(systems) - 1) // 2,
            "n_inversions": len(inversions),
            "n_significant_inversions": sum(i["both_significant"] for i in inversions),
            "inversions": inversions,
        }
        print(f"\n=== ranking under train={regime} ===")
        print(f"Kendall tau(UAR|intended, UAR|perceived) = {tau:.3f} "
              f"(p={p:.3g}, 95% CI [{np.percentile(taus,2.5):.2f}, "
              f"{np.percentile(taus,97.5):.2f}])")
        print(f"{len(inversions)}/{out[regime]['n_pairs']} pairs invert, "
              f"{out[regime]['n_significant_inversions']} of them with both "
              f"differences significant")
        for inv in inversions:
            flag = "*" if inv["both_significant"] else " "
            print(f"  {flag} {inv['pair'][0]:9s} vs {inv['pair'][1]:9s}  "
                  f"d_int={inv['delta_intended']:+.4f} "
                  f"d_perc={inv['delta_perceived']:+.4f}")

    # ---- pooled: every (system, training label) entry on one leaderboard ----
    # A real CREMA-D table mixes papers that made different, undocumented
    # labelling choices, so the entry -- not the architecture -- is the unit.
    entries = [(s, r) for s in systems for r in REGIMES]
    a = np.array([res[(res.system == s) & (res.train == r)].UAR_intended.iloc[0]
                  for s, r in entries])
    b = np.array([res[(res.system == s) & (res.train == r)].UAR_perceived.iloc[0]
                  for s, r in entries])
    tau, p = kendalltau(a, b)
    n_inv = sum((a[i] - a[j]) * (b[i] - b[j]) < 0
                for i, j in combinations(range(len(entries)), 2))
    top_i = entries[int(a.argmax())]
    top_p = entries[int(b.argmax())]
    out["pooled"] = {
        "n_entries": len(entries),
        "kendall_tau": float(tau), "kendall_p": float(p),
        "n_pairs": len(entries) * (len(entries) - 1) // 2,
        "n_inversions": int(n_inv),
        "top_intended": list(top_i), "top_perceived": list(top_p),
        "rank_of_intended_winner_under_perceived":
            int((-b).argsort().argsort()[int(a.argmax())] + 1),
        "rank_of_perceived_winner_under_intended":
            int((-a).argsort().argsort()[int(b.argmax())] + 1),
    }
    print(f"\n=== pooled leaderboard ({len(entries)} entries) ===")
    print(f"Kendall tau = {tau:.3f} (p={p:.3g}); "
          f"{n_inv}/{out['pooled']['n_pairs']} pairs invert")
    print(f"best under intended: {top_i} -> rank "
          f"{out['pooled']['rank_of_intended_winner_under_perceived']} under perceived")
    print(f"best under perceived: {top_p} -> rank "
          f"{out['pooled']['rank_of_perceived_winner_under_intended']} under intended")

    json.dump({"ranking": out, "uar_ci": ci},
              open(TABLES / "ranking_analysis.json", "w"), indent=2)
    print("\nwrote", TABLES / "main_results.csv", "and",
          TABLES / "ranking_analysis.json")


if __name__ == "__main__":
    main()
