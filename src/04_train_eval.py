"""Train every system under each label regime and score it under every key.

For each (system, training label regime) pair we run speaker-independent
5-fold cross-validation over the 91 actors, collect out-of-fold posteriors for
all 7442 clips, and then score those same posteriors against

    intended   the filename label,
    perceived  the plurality voice-only label,
    soft       the full voice-only vote distribution.

Nothing about the model changes between scoring keys: only the answer sheet.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import PROC, FEAT, TABLES, EMOTIONS
from models import make_systems

SEED = 20240917
N_FOLDS = 5
REGIMES = ["intended", "perceived", "soft"]


def load_features(key):
    z = np.load(FEAT / f"{key}.npz", allow_pickle=True)
    return pd.DataFrame(z["X"], index=z["clips"]).sort_index()


def target_matrices(df):
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    n = len(df)
    Q = {}
    for regime, col in [("intended", "intended"), ("perceived", "perceived")]:
        M = np.zeros((n, len(EMOTIONS)))
        M[np.arange(n), [idx[e] for e in df[col]]] = 1.0
        Q[regime] = M
    Q["soft"] = df[[f"p_{e}" for e in EMOTIONS]].to_numpy(float)
    return Q


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("--")] or None
    labels = pd.read_csv(PROC / "clip_labels.csv", index_col=0).sort_index()
    Q = target_matrices(labels)
    groups = labels["actor"].to_numpy()

    feats = {}
    for key in ["mfcc", "funcs", "ssl"]:
        path = FEAT / f"{key}.npz"
        if path.exists():
            F = load_features(key)
            assert list(F.index) == list(labels.index), key
            feats[key] = F.to_numpy(np.float32)
            print(f"features {key}: {feats[key].shape}")

    systems = [s for s in make_systems() if s[1] in feats]
    if only:
        systems = [s for s in systems if s[0] in only]
    print("systems:", [s[0] for s in systems])

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(np.zeros(len(labels)), groups=groups))
    fold_id = np.empty(len(labels), int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k

    suffix = "" if __import__("os").environ.get("BALANCE", "1") != "0" else "_unbal"
    outdir = PROC / f"posteriors{suffix}"
    outdir.mkdir(exist_ok=True)
    for name, fkey, build in systems:
        X = feats[fkey]
        for regime in REGIMES:
            dest = outdir / f"{name}__{regime}.npy"
            if dest.exists() and "--force" not in sys.argv:
                print(f"  {name:10s} train={regime:9s} cached", flush=True)
                continue
            t0 = time.time()
            P = np.zeros((len(labels), len(EMOTIONS)))
            for k, (tr, te) in enumerate(folds):
                # inner split by actor for early stopping / no peeking at test
                g_tr = groups[tr]
                uniq = np.unique(g_tr)
                rs = np.random.RandomState(SEED + k)
                val_actors = set(rs.choice(uniq, max(2, len(uniq) // 6),
                                           replace=False))
                is_val = np.array([g in val_actors for g in g_tr])
                fit_idx, val_idx = tr[~is_val], tr[is_val]
                mdl = build()
                mdl.fit(X[fit_idx], Q[regime][fit_idx],
                        X_val=X[val_idx], Q_val=Q[regime][val_idx])
                P[te] = mdl.predict_proba(X[te])
            np.save(outdir / f"{name}__{regime}.npy", P)
            top = P.argmax(1)
            acc_i = (np.array(EMOTIONS)[top] == labels["intended"]).mean()
            acc_p = (np.array(EMOTIONS)[top] == labels["perceived"]).mean()
            print(f"  {name:10s} train={regime:9s} "
                  f"acc|intended={acc_i:.4f} acc|perceived={acc_p:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(PROC / "cv_folds.npz", fold=fold_id,
                        clips=np.array(labels.index))
    (TABLES / "run_config.json").write_text(json.dumps({
        "n_folds": N_FOLDS, "seed": SEED, "regimes": REGIMES,
        "systems": [s[0] for s in systems],
        "feature_dims": {k: int(v.shape[1]) for k, v in feats.items()},
        "n_actors": int(len(np.unique(groups))),
    }, indent=2))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
