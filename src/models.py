"""System definitions and the label-regime-agnostic training loop.

Every target is a distribution over the six categories, so the three label
regimes differ only in that distribution:

    intended   one-hot on the filename label
    perceived  one-hot on the plurality of the voice-only votes
    soft       the normalised voice-only vote distribution

All estimators are fitted by weighted cross-entropy, which for a distributional
target is obtained exactly by replicating each clip once per category with the
category's probability mass as its sample weight.

Training is class-balanced in every regime. This matters because the three
label sets have very different priors -- 14.6% of the intended labels are
neutral against 56.6% of the perceived ones -- and unbalanced training against a
neutral-dominated target collapses onto neutral. Since we report unweighted
average recall, which is deliberately prior-insensitive, the matched training
objective is the prior-insensitive one; otherwise a regime would be penalised
for the shape of its label distribution rather than for its content.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import EMOTIONS

SEED = 20240917
K = len(EMOTIONS)


BALANCE = __import__("os").environ.get("BALANCE", "1") != "0"


def class_weights(Q):
    """Inverse-frequency weights from the target mass in each category."""
    if not BALANCE:
        return np.ones(K)
    mass = Q.sum(axis=0)
    cw = mass.sum() / (K * np.maximum(mass, 1e-9))
    return cw / cw.mean()


# --------------------------------------------------------------------- torch
class TorchMLP:
    """Two-layer MLP trained directly on distributional targets."""

    def __init__(self, hidden=(256, 128), dropout=0.3, lr=1e-3, epochs=80,
                 batch=128, weight_decay=1e-4, patience=12, seed=SEED):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.epochs, self.batch, self.weight_decay = epochs, batch, weight_decay
        self.patience, self.seed = patience, seed

    def fit(self, X, Q, X_val=None, Q_val=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(8)

        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)
        cw = class_weights(Q)
        Qw = Q * cw[None, :]

        layers, prev = [], X.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, K))
        self.net = nn.Sequential(*layers)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)

        Xt = torch.tensor(X, dtype=torch.float32)
        Qt = torch.tensor(Qw, dtype=torch.float32)
        if X_val is not None:
            Xv = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32)
            Qv = torch.tensor(Q_val * cw[None, :], dtype=torch.float32)

        n = len(Xt)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, waited = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                logp = torch.log_softmax(self.net(Xt[idx]), dim=1)
                (-(Qt[idx] * logp).sum(1).mean()).backward()
                opt.step()
            if X_val is None:
                continue
            self.net.eval()
            with torch.no_grad():
                vl = -(Qv * torch.log_softmax(self.net(Xv), 1)).sum(1).mean().item()
            if vl < best - 1e-4:
                best, waited = vl, 0
                best_state = {k: v.clone()
                              for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            return torch.softmax(self.net(Xt), 1).numpy()


# ------------------------------------------------------------------- sklearn
class SklearnSystem:
    """Wraps a scikit-learn estimator so it consumes distributional targets."""

    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, Q, X_val=None, Q_val=None):
        cw = class_weights(Q)
        rows, cls = np.nonzero(Q > 1e-8)
        w = Q[rows, cls] * cw[cls]
        self.model = self.factory()
        last = self.model.steps[-1][0]
        self.model.fit(X[rows], cls, **{f"{last}__sample_weight": w})
        self.classes_ = self.model.classes_.astype(int)
        return self

    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        full = np.zeros((len(X), K))
        full[:, self.classes_] = p
        return full


def make_systems():
    """(name, feature key, builder) for every system compared in the paper."""
    def lr(C):
        return lambda: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=3000))])

    def rbf(C, gamma, n_comp=1200):
        return lambda: Pipeline([
            ("sc", StandardScaler()),
            ("ny", Nystroem(gamma=gamma, n_components=n_comp,
                            random_state=SEED)),
            ("clf", LogisticRegression(C=C, max_iter=3000))])

    def hgb():
        return Pipeline([("clf", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEED))])

    return [
        ("MFCC-LR",  "mfcc",  lambda: SklearnSystem(lr(1.0))),
        ("MFCC-RBF", "mfcc",  lambda: SklearnSystem(rbf(1.0, 0.02))),
        ("FUNC-LR",  "funcs", lambda: SklearnSystem(lr(0.03))),
        ("FUNC-RBF", "funcs", lambda: SklearnSystem(rbf(0.3, 0.001))),
        ("FUNC-GB",  "funcs", lambda: SklearnSystem(hgb)),
        ("FUNC-MLP", "funcs", lambda: TorchMLP(hidden=(256, 128))),
        ("SSL-LR",   "ssl",   lambda: SklearnSystem(lr(0.03))),
        ("SSL-MLP",  "ssl",   lambda: TorchMLP(hidden=(256, 128))),
    ]
