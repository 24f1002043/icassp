"""Sanity checks over everything the paper depends on."""
import json
import sys

import numpy as np
import pandas as pd

from common import PROC, TABLES, FIGURES, EMOTIONS

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def main():
    print("labels")
    df = pd.read_csv(PROC / "clip_labels.csv", index_col=0)
    check("7442 clips", len(df) == 7442, str(len(df)))
    check("91 actors", df.actor.nunique() == 91, str(df.actor.nunique()))
    P = df[[f"p_{e}" for e in EMOTIONS]].to_numpy()
    check("soft labels sum to 1", np.allclose(P.sum(1), 1.0))
    check("no NaN in soft labels", not np.isnan(P).any())
    check("intended labels in the six categories",
          set(df.intended) <= set(EMOTIONS))
    check("perceived labels in the six categories",
          set(df.perceived) <= set(EMOTIONS))

    print("headline claims")
    a = json.load(open(TABLES / "label_audit.json"))
    r = a["recovery"]["unambiguous_only"]
    s = a["per_emotion"]["S"]["unambiguous_only"]
    check("overall recovery 45.6%", abs(r - 0.4559) < 5e-4, f"{r:.4f}")
    check("sad recovery 18.2%", abs(s - 0.1821) < 5e-4, f"{s:.4f}")
    check("sad above the 1/6 floor", s > 1 / 6, f"{s:.4f} vs {1/6:.4f}")

    print("posteriors")
    pdir = PROC / "posteriors"
    files = sorted(pdir.glob("*.npy"))
    have = {}
    for f in files:
        sysname, regime = f.stem.split("__")
        have.setdefault(sysname, set()).add(regime)
    complete = [s for s, v in have.items() if v >= {"intended", "perceived", "soft"}]
    check("at least 6 complete systems", len(complete) >= 6,
          f"{len(complete)}: {sorted(complete)}")
    for f in files:
        M = np.load(f)
        if M.shape != (7442, 6) or np.isnan(M).any() or \
                not np.allclose(M.sum(1), 1.0, atol=1e-4):
            check(f"{f.name} well-formed", False, str(M.shape))
            break
    else:
        check("every posterior is a valid 7442x6 distribution", True,
              f"{len(files)} files")

    print("cross-validation is speaker independent")
    folds = np.load(PROC / "cv_folds.npz", allow_pickle=True)
    fid, clips = folds["fold"], folds["clips"]
    actor = np.array([int(c.split("_")[0]) for c in clips])
    leaks = [x for x in np.unique(actor) if len(np.unique(fid[actor == x])) > 1]
    check("no actor spans two folds", not leaks, f"{len(leaks)} leaking")

    print("paper artefacts")
    for f in ["tab_audit.tex", "tab_main.tex", "numbers.tex"]:
        p = TABLES.parents[1] / "paper" / f
        check(f"paper/{f}", p.exists() and p.stat().st_size > 0)
    for f in ["fig1_confusion.pdf", "fig2_ranking.pdf", "fig3_per_emotion.pdf"]:
        check(f"figures/{f}", (FIGURES / f).exists())
    pdf = TABLES.parents[1] / "paper" / "main.pdf"
    check("paper/main.pdf", pdf.exists() and pdf.stat().st_size > 50_000,
          f"{pdf.stat().st_size//1024} KB" if pdf.exists() else "missing")

    # numbers.tex must not carry an unresolved placeholder
    nums = (TABLES.parents[1] / "paper" / "numbers.tex").read_text()
    check("no zeroed ranking macros left",
          "\\newcommand{\\PoolTau}{0.00}" not in nums)

    print()
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
