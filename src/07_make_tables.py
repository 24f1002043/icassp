"""Emit the LaTeX tables and a macro file of inline numbers for the paper."""
import json
import sys

import pandas as pd

from common import PROC, TABLES, EMOTIONS, EMO_NAME

PAPER = TABLES.parents[1] / "paper"
PAPER.mkdir(exist_ok=True)
REGIMES = ["intended", "perceived", "soft"]


def f2(x):
    return f"{100 * x:.1f}"


def write(name, lines):
    (PAPER / name).write_text("\n".join(lines) + "\n", newline="\n")
    print("wrote", name)


def table_label_audit():
    a = json.load(open(TABLES / "label_audit.json"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Voice-only listener behaviour on CREMA-D. \emph{Recovery} is"
        r" the share of clips whose plurality perceived label equals the"
        r" intended one, over the " + str(a["n_unambiguous"]) + r" clips with an"
        r" unambiguous plurality; $\alpha$ is the mean share of votes going to"
        r" the winning category, $H$ the vote entropy normalised to $[0,1]$.}",
        r"\label{tab:audit}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Intended & $n$ & Recovery (\%) & $\alpha$ & $H$ \\",
        r"\hline",
    ]
    for e in EMOTIONS:
        d = a["per_emotion"][e]
        lines.append(f"{EMO_NAME[e]} & {d['n']} & {f2(d['unambiguous_only'])} & "
                     f"{d['mean_agreement']:.2f} & {d['mean_entropy']:.2f}"
                     r" \\")
    lines += [
        r"\hline",
        f"All & {a['n_clips']} & "
        r"\textbf{" + f2(a["recovery"]["unambiguous_only"]) + r"} & "
        f"{a['mean_agreement']:.2f} & {a['mean_entropy']:.2f}" + r" \\",
        r"Chance & --- & 16.7 & --- & --- \\",
        r"\hline",
        r"\end{tabular}",
        r"\vspace{-4mm}",
        r"\end{table}",
    ]
    write("tab_audit.tex", lines)


def table_main():
    res = pd.read_csv(TABLES / "main_results.csv")
    systems = sorted(res.system.unique())
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Unweighted average recall (\%) under speaker-independent"
        r" five-fold cross-validation over the 91 actors. Each block fixes the"
        r" label a system was \emph{trained} on; within a block, UAR$_I$ and"
        r" UAR$_P$ score the same predictions against the intended and the"
        r" perceived label respectively, and $R$ is the probability that the"
        r" prediction matches one randomly drawn listener vote. Best in each"
        r" column in bold.}",
        r"\label{tab:main}",
        r"\small",
        r"\begin{tabular}{l@{\hspace{5mm}}ccc@{\hspace{5mm}}ccc@{\hspace{5mm}}ccc}",
        r"\hline",
        r" & \multicolumn{3}{c}{trained on intended}"
        r" & \multicolumn{3}{c}{trained on perceived}"
        r" & \multicolumn{3}{c}{trained on soft} \\",
        r"\cline{2-4}\cline{5-7}\cline{8-10}",
        r"System & UAR$_I$ & UAR$_P$ & $R$ & UAR$_I$ & UAR$_P$ & $R$"
        r" & UAR$_I$ & UAR$_P$ & $R$ \\",
        r"\hline",
    ]
    cols = ["UAR_intended", "UAR_perceived", "rater_agreement"]
    piv = {}
    for s in systems:
        for regime in REGIMES:
            r = res[(res.system == s) & (res.train == regime)]
            piv[(s, regime)] = r.iloc[0] if len(r) else None
    best = {}
    for regime in REGIMES:
        for c in cols:
            vals = [piv[(s, regime)][c] for s in systems if piv[(s, regime)] is not None]
            best[(regime, c)] = max(vals) if vals else None
    for s in systems:
        cells = [s]
        for regime in REGIMES:
            r = piv[(s, regime)]
            for c in cols:
                if r is None:
                    cells.append("---")
                    continue
                txt = f2(r[c])
                if abs(r[c] - best[(regime, c)]) < 1e-12:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\vspace{-4mm}", r"\end{table*}"]
    write("tab_main.tex", lines)


def macros():
    a = json.load(open(TABLES / "label_audit.json"))
    p = json.load(open(TABLES / "perceptual_audit.json"))
    m = {
        "Recovery": f2(a["recovery"]["unambiguous_only"]),
        "RecoveryStrict": f2(a["recovery"]["tie_counts_as_miss"]),
        "RecoveryGenerous": f2(a["recovery"]["tie_counts_as_hit"]),
        "SadRecovery": f2(a["per_emotion"]["S"]["unambiguous_only"]),
        "SadToNeutral": f2(p["neutral_attractor"]["sad_heard_as_neutral"]),
        "NeutralShare": f2(p["neutral_attractor"]["perceived_neutral_share"]),
        "NonNeutralAsNeutral":
            f2(p["neutral_attractor"]["nonneutral_intent_heard_as_neutral"]),
        "RawExtra": f"{p['raw_recount']['extra_vs_shipped']:,}".replace(",", "{,}"),
        "RawRecovery": f2(p["raw_recount"]["recovery_unambiguous"]),
        "RawSad": f2(p["raw_recount"]["sad_recovery"]),
        "RaterRater": f2(p["reliability"]["rater_rater_agreement"]),
        "Kappa": f"{p['reliability']['fleiss_kappa_perceived']:.2f}",
        "Alpha": f"{p['reliability']['krippendorff_alpha_perceived']:.2f}",
        "RaterVsIntended": f2(p["human_ceiling"]["rater_vs_intended"]),
        "RaterVsPeer": f2(p["human_ceiling"]["rater_vs_peer_plurality"]),
        "CeilingGap": f"{100 * (p['human_ceiling']['rater_vs_peer_plurality'] - p['human_ceiling']['rater_vs_intended']):.0f}",
        "FaceRecovery": f2(p["by_modality"]["face"]["recovery_unambiguous"]),
        "MMRecovery": f2(p["by_modality"]["multimodal"]["recovery_unambiguous"]),
        "RecoveryLO": f2(p["by_intensity"]["LO"]["recovery"]),
        "RecoveryHI": f2(p["by_intensity"]["HI"]["recovery"]),
        "NClips": str(a["n_clips"]),
        "NRatings": f"{a['n_ratings']:,}".replace(",", "{,}"),
        "NUnamb": str(a["n_unambiguous"]),
        "TiedPct": f2(a["tied_fraction"]),
        "MeanRaters": f"{a['mean_raters_per_clip']:.1f}",
    }
    for tag in ["Intended", "Perceived", "Soft"]:
        m[f"Tau{tag}"], m[f"Inv{tag}"] = "0.00", "0"
        m[f"SigInv{tag}"], m[f"NPairs{tag}"] = "0", "0"
    rk_path = TABLES / "ranking_analysis.json"
    if rk_path.exists():
        rk = json.load(open(rk_path))["ranking"]
        for regime, d in rk.items():
            tag = regime.capitalize()
            m[f"Tau{tag}"] = f"{d['kendall_tau']:.2f}"
            m[f"Inv{tag}"] = str(d["n_inversions"])
            m[f"SigInv{tag}"] = str(d.get("n_significant_inversions", 0))
            m[f"NPairs{tag}"] = str(d["n_pairs"])
        if "pooled" in rk:
            d = rk["pooled"]
            m["PoolEntries"] = str(d["n_entries"])
            m["PoolTau"] = f"{d['kendall_tau']:.2f}"
            m["PoolInv"] = str(d["n_inversions"])
            m["PoolPairs"] = str(d["n_pairs"])
            rank = d["rank_of_intended_winner_under_perceived"]
            suffix = "th" if 10 <= rank % 100 <= 20 else \
                {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
            m["PoolWinnerDrop"] = f"{rank}{suffix}"
            m["PoolTopI"] = r"\mbox{" + d["top_intended"][0] + "}/" + d["top_intended"][1]
            m["PoolTopP"] = r"\mbox{" + d["top_perceived"][0] + "}/" + d["top_perceived"][1]

    res_path = TABLES / "main_results.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        for regime in REGIMES:
            sub = res[res.train == regime]
            if not len(sub):
                continue
            tag = regime.capitalize()
            m[f"BestI{tag}"] = f2(sub.UAR_intended.max())
            m[f"BestP{tag}"] = f2(sub.UAR_perceived.max())
            m[f"BestR{tag}"] = f2(sub.rater_agreement.max())
        # what a fixed system's reported score does when only the key changes
        ti = res[res.train == "intended"].set_index("system")
        m["SwingWAmax"] = f"{100 * (ti.WA_intended - ti.WA_perceived).max():.1f}"
        m["SwingUARmax"] = f"{100 * (ti.UAR_intended - ti.UAR_perceived).max():.1f}"
        n_sys = res.system.nunique()
        m["NSystems"] = str(n_sys)
        m["NSystemsWord"] = {1: "one", 2: "two", 3: "three", 4: "four",
                             5: "five", 6: "six", 7: "seven", 8: "eight",
                             9: "nine", 10: "ten"}.get(n_sys, str(n_sys))
        # the strongest system under the standard protocol, both ways
        top = ti.WA_intended.idxmax()
        m["TopSystem"] = r"\mbox{" + top + "}"
        m["TopWAI"] = f2(ti.loc[top, "WA_intended"])
        m["TopWAP"] = f2(ti.loc[top, "WA_perceived"])
        # what training on the perceptual label buys, scored perceptually
        tp = res[res.train == "perceived"].set_index("system")
        ts = res[res.train == "soft"].set_index("system")
        m["GainPerceived"] = f"{100 * (tp.UAR_perceived - ti.UAR_perceived).mean():.1f}"
        m["GainSoft"] = f"{100 * (ts.UAR_perceived - ti.UAR_perceived).mean():.1f}"
        m["RGainSoft"] = f"{100 * (ts.rater_agreement - ti.rater_agreement).mean():.1f}"
        m["RIntMean"] = f2(ti.rater_agreement.mean())
        m["RSoftMean"] = f2(ts.rater_agreement.mean())
        m["BestUARPerc"] = f2(res.UAR_perceived.max())
        m["BestUARInt"] = f2(res.UAR_intended.max())

        # How closely does a system scored against intent mirror the listeners?
        import numpy as np
        lab = pd.read_csv(PROC / "clip_labels.csv", index_col=0).sort_index()
        P = np.load(PROC / "posteriors" / f"{top}__intended.npy")
        pred = np.array(EMOTIONS)[P.argmax(1)]
        y_i = lab["intended"].to_numpy()
        model_rec = np.array([(pred[y_i == e] == e).mean() for e in EMOTIONS])
        human_rec = np.array([a["per_emotion"][e]["unambiguous_only"]
                              for e in EMOTIONS])
        m["CorrModelHuman"] = f"{np.corrcoef(model_rec, human_rec)[0, 1]:.2f}"
        for e, tag in [("S", "Sad"), ("N", "Neu"), ("D", "Dis"), ("H", "Hap")]:
            i = EMOTIONS.index(e)
            m[f"Model{tag}"] = f2(model_rec[i])
            m[f"Human{tag}"] = f2(human_rec[i])
        # categories where the model recovers intent better than listeners do
        m["NBeatHuman"] = str(int((model_rec > human_rec).sum()))
        m["ModelBeatsBy"] = f"{100 * (model_rec - human_rec).max():.0f}"
        si = EMOTIONS.index("S")
        m["SadRatio"] = f"{model_rec[si] / max(human_rec[si], 1e-9):.1f}"
    out = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items())
    (PAPER / "numbers.tex").write_text(out + "\n", newline="\n")
    print(f"wrote numbers.tex with {len(m)} macros")
    for k, v in m.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    table_label_audit()
    if (TABLES / "main_results.csv").exists() and "--audit-only" not in sys.argv:
        table_main()
    macros()
