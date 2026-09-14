"""Figures for the paper.  Rendered as PDF at column width, grayscale-safe."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROC, TABLES, FIGURES, EMOTIONS, EMO_NAME

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42,
    # STIX matches Times closely; without this, matplotlib falls back to
    # DejaVu for maths and the PDF ends up carrying a second font family.
    "mathtext.fontset": "stix",
})
COL = 3.35          # single-column width in inches
SHORT = {e: EMO_NAME[e][:3] for e in EMOTIONS}


def fig_confusion():
    labels = pd.read_csv(PROC / "clip_labels.csv", index_col=0)
    ua = labels["unambiguous"]
    conf = pd.crosstab(labels.loc[ua, "intended"],
                       labels.loc[ua, "perceived_strict"])
    conf = conf.reindex(index=EMOTIONS, columns=EMOTIONS, fill_value=0)
    M = (conf.div(conf.sum(axis=1), axis=0) * 100).to_numpy()

    fig, ax = plt.subplots(figsize=(COL, COL * 0.92))
    im = ax.imshow(M, cmap="Greys", vmin=0, vmax=100)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(range(6), [SHORT[e] for e in EMOTIONS])
    ax.set_yticks(range(6), [SHORT[e] for e in EMOTIONS])
    ax.set_xlabel("perceived label (plurality of voice-only votes)")
    ax.set_ylabel("intended label (filename)")
    for i in range(6):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="#d62728", lw=1.1))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("% of clips", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "fig1_confusion.pdf")
    print("wrote fig1")


def fig_ranking():
    res = pd.read_csv(TABLES / "main_results.csv")
    rk = json.load(open(TABLES / "ranking_analysis.json"))["ranking"]
    sub = res[res.train == "intended"].set_index("system")
    order = sub["UAR_intended"].sort_values(ascending=False).index.tolist()
    a = sub.loc[order, "UAR_intended"].to_numpy() * 100
    b = sub.loc[order, "UAR_perceived"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(COL, COL * 0.86))
    ra = (-a).argsort().argsort()
    rb = (-b).argsort().argsort()
    for i, s in enumerate(order):
        inv = ra[i] != rb[i]
        ax.plot([0, 1], [ra[i], rb[i]], "-o", ms=2.8, lw=1.4 if inv else 0.7,
                color="#c0392b" if inv else "0.6", zorder=3 if inv else 2)
        ax.text(-0.05, ra[i], f"{s}  ", ha="right", va="center", fontsize=6.6,
                color="#c0392b" if inv else "0.15")
        ax.text(1.05, rb[i], f"  {s}", ha="left", va="center", fontsize=6.6,
                color="#c0392b" if inv else "0.15")
        ax.text(-0.05, ra[i], f"{ra[i]+1} ", ha="right", va="center",
                fontsize=6.6, color="0.55", alpha=0)
    ax.set_xlim(-0.95, 1.95)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xticks([0, 1], ["scored against\nintended label",
                           "scored against\nperceived label"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    tau = rk["intended"]["kendall_tau"]
    ax.set_title(rf"same trained systems, two answer keys ($\tau$ = {tau:.2f})",
                 fontsize=7.6, pad=3)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "fig2_ranking.pdf")
    print("wrote fig2")


def fig_ranking_pooled():
    """Every (system, training label) entry on one leaderboard, both keys."""
    res = pd.read_csv(TABLES / "main_results.csv")
    rk = json.load(open(TABLES / "ranking_analysis.json"))["ranking"]
    if "pooled" not in rk:
        return
    short = {"intended": "I", "perceived": "P", "soft": "S"}
    res = res.assign(entry=[f"{s}/{short[t]}"
                            for s, t in zip(res.system, res.train)])
    sub = res.set_index("entry")
    order = sub["UAR_intended"].sort_values(ascending=False).index.tolist()
    a = sub.loc[order, "UAR_intended"].to_numpy()
    b = sub.loc[order, "UAR_perceived"].to_numpy()
    ra, rb = (-a).argsort().argsort(), (-b).argsort().argsort()

    h = max(2.0, 0.135 * len(order) + 0.55)
    fig, ax = plt.subplots(figsize=(COL, h))
    for i, e in enumerate(order):
        moved = abs(int(ra[i]) - int(rb[i]))
        col = "#c0392b" if moved >= 3 else ("0.35" if moved else "0.72")
        ax.plot([0, 1], [ra[i], rb[i]], "-o", ms=2.2,
                lw=1.3 if moved >= 3 else 0.6, color=col,
                zorder=3 if moved >= 3 else 2)
        ax.text(-0.04, ra[i], f"{e} ", ha="right", va="center", fontsize=5.4,
                color=col if moved >= 3 else "0.15")
        ax.text(1.04, rb[i], f" {e}", ha="left", va="center", fontsize=5.4,
                color=col if moved >= 3 else "0.15")
    ax.set_xlim(-0.85, 1.85)
    ax.set_ylim(len(order) - 0.4, -0.9)
    ax.set_xticks([0, 1], ["scored against\nintended label",
                           "scored against\nperceived label"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    tau = rk["pooled"]["kendall_tau"]
    ax.set_title(rf"{len(order)} leaderboard entries ($\tau$ = {tau:.2f})",
                 fontsize=7.4, pad=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "fig2b_ranking_pooled.pdf")
    print("wrote fig2b (pooled,", len(order), "entries )")


def fig_per_emotion():
    labels = pd.read_csv(PROC / "clip_labels.csv", index_col=0).sort_index()
    audit = json.load(open(TABLES / "label_audit.json"))
    pdir = PROC / "posteriors"
    res = pd.read_csv(TABLES / "main_results.csv")
    best = res[res.train == "intended"].sort_values(
        "UAR_intended", ascending=False).iloc[0]["system"]

    y_i = labels["intended"].to_numpy()
    y_p = labels["perceived"].to_numpy()
    P_i = np.load(pdir / f"{best}__intended.npy")
    P_p = np.load(pdir / f"{best}__perceived.npy")
    pred_i = np.array(EMOTIONS)[P_i.argmax(1)]
    pred_p = np.array(EMOTIONS)[P_p.argmax(1)]

    human = [audit["per_emotion"][e]["unambiguous_only"] * 100 for e in EMOTIONS]
    m_i = [(pred_i[y_i == e] == e).mean() * 100 for e in EMOTIONS]
    m_p = [(pred_p[y_p == e] == e).mean() * 100 for e in EMOTIONS]

    x = np.arange(6)
    w = 0.27
    fig, ax = plt.subplots(figsize=(COL, COL * 0.78))
    ax.bar(x - w, human, w, label="listeners, intended key", color="0.25")
    ax.bar(x, m_i, w, label=f"{best}, intended key", color="0.62")
    # hatched so the series stays distinct from the dark bars in a B/W print
    ax.bar(x + w, m_p, w, label=f"{best}, perceived key", color="#c0392b",
           hatch="////", edgecolor="white", linewidth=0)
    ax.axhline(100 / 6, ls=(0, (2, 2)), lw=0.7, color="k", zorder=0)
    ax.set_xlim(-0.55, 6.02)
    ax.text(5.62, 100 / 6, "chance", fontsize=6.2, ha="left", va="center",
            color="0.35", bbox=dict(facecolor="white", edgecolor="none",
                                    pad=0.6))
    ax.set_xticks(x, [SHORT[e] for e in EMOTIONS])
    ax.set_ylabel("per-class recall (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", ncol=1,
              handlelength=1.0, borderpad=0.0, labelspacing=0.22,
              bbox_to_anchor=(0.0, 1.04))
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "fig3_per_emotion.pdf")
    print("wrote fig3", "(system:", best, ")")


if __name__ == "__main__":
    import sys
    fig_confusion()
    if (TABLES / "main_results.csv").exists() and "--conf-only" not in sys.argv:
        fig_ranking()
        fig_ranking_pooled()
        fig_per_emotion()
