"""
metrics.py — Evaluation metrics
===============================

MNIST
    fd_lenet   Frechet distance between Gaussian fits of penultimate-layer
               features of a LeNet-style classifier trained on MNIST
               (``train_lenet`` below; test accuracy stored with the weights).
               NOTE: this is *not* the InceptionV3 FID; values are not
               comparable with Inception-FID numbers in the literature.
    ndb        Number of statistically different bins (Richardson & Weiss,
               2018): K-means (K=100) bins fitted on real training images;
               two-proportion z-test (alpha = 0.05) between generated samples
               and real held-out test images.  Reported as NDB/K.

Tabular (standardised feature space; last column = label in {-1,+1})
    mean_tvd   Column-wise total variation distance on 20 equal-width bins
               spanning the real-data range (label column: 2 categories),
               averaged over all 31 columns.
    corr_diff  Mean absolute difference of off-diagonal Pearson correlations.
    ml_eff     Train-on-synthetic / test-on-real AUROC on the held-out real
               test split (logistic regression and random forest, averaged).

Time series (windows of shape (24, 50))
    disc_score |accuracy - 0.5| of a post-hoc GRU classifier separating real
               and synthetic windows (TimeGAN protocol, Yoon et al. 2019).
    pred_mae   MAE on real test windows of a GRU one-step-ahead predictor
               trained only on synthetic windows (TimeGAN protocol).
    corr_diff  Mean absolute difference of off-diagonal cross-asset
               correlations of the pooled returns.
    mean_tvd   Mean per-asset TVD of the pooled return distribution.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

from data import DATA_DIR, load_mnist


# =============================================================== MNIST
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        self.head = nn.Linear(128, 10)

    def features(self, x):
        return self.f(x.view(-1, 1, 28, 28))

    def forward(self, x):
        return self.head(self.features(x))


LENET_PATH = os.path.join(DATA_DIR, "lenet_mnist.pt")


def train_lenet(epochs: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    xtr, ytr = load_mnist(True)
    xte, yte = load_mnist(False)
    net = LeNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    X, Y = torch.from_numpy(xtr), torch.from_numpy(ytr).long()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(torch.from_numpy(xte[i:i + 1000])).argmax(1)
                          for i in range(0, len(xte), 1000)])
    acc = float((pred.numpy() == yte).mean())
    torch.save({"state": net.state_dict(), "test_acc": acc, "epochs": epochs}, LENET_PATH)
    return acc


_LENET = None


def _lenet():
    global _LENET
    if _LENET is None:
        ck = torch.load(LENET_PATH, weights_only=False)
        _LENET = LeNet()
        _LENET.load_state_dict(ck["state"])
        _LENET.eval()
    return _LENET


def lenet_features(x: np.ndarray) -> np.ndarray:
    net = _lenet()
    with torch.no_grad():
        return np.concatenate([net.features(torch.from_numpy(x[i:i + 1000]).float()).numpy()
                               for i in range(0, len(x), 1000)])


def frechet(f1: np.ndarray, f2: np.ndarray) -> float:
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def fd_lenet(fake: np.ndarray, real: np.ndarray) -> float:
    return frechet(lenet_features(real), lenet_features(fake))


_NDB_KM = None


def ndb(fake: np.ndarray, real_test: np.ndarray, K: int = 100, alpha: float = 0.05) -> float:
    from sklearn.cluster import KMeans
    from scipy.stats import norm
    global _NDB_KM
    if _NDB_KM is None:
        xtr, _ = load_mnist(True)
        rng = np.random.default_rng(0)
        _NDB_KM = KMeans(n_clusters=K, n_init=3, random_state=0).fit(
            xtr[rng.choice(len(xtr), 20000, replace=False)].astype(np.float64))
    km = _NDB_KM
    pr = np.bincount(km.predict(real_test.astype(np.float64)), minlength=K) / len(real_test)
    pf = np.bincount(km.predict(fake.astype(np.float64)), minlength=K) / len(fake)
    n1, n2 = len(real_test), len(fake)
    pp = (pr * n1 + pf * n2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    zstat = np.abs(pr - pf) / np.maximum(se, 1e-12)
    return float((zstat > norm.ppf(1 - alpha / 2)).sum() / K)


# =============================================================== tabular
def _col_tvd(r: np.ndarray, f: np.ndarray, bins: int = 20) -> float:
    lo, hi = r.min(), r.max()
    edges = np.linspace(lo, hi, bins + 1)
    rc = np.clip(r, lo, hi)
    fc = np.clip(f, lo, hi)
    pr = np.histogram(rc, edges)[0] / len(r)
    pf = np.histogram(fc, edges)[0] / len(f)
    return 0.5 * float(np.abs(pr - pf).sum())


def tabular_tvd(fake: np.ndarray, real: np.ndarray) -> tuple[float, np.ndarray]:
    tv = [_col_tvd(real[:, j], fake[:, j]) for j in range(real.shape[1] - 1)]
    pr = (real[:, -1] > 0).mean()
    pf = (fake[:, -1] > 0).mean()
    tv.append(float(abs(pr - pf)))        # TVD of a binary variable
    tv = np.array(tv)
    return float(tv.mean()), tv


def corr_diff(fake: np.ndarray, real: np.ndarray) -> float:
    cr, cf = np.corrcoef(real, rowvar=False), np.corrcoef(fake, rowvar=False)
    cf = np.nan_to_num(cf)
    iu = np.triu_indices_from(cr, 1)
    return float(np.abs(cr[iu] - cf[iu]).mean())


def ml_efficacy(train_X, train_y, test_X, test_y, seed: int = 0) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if len(np.unique(train_y)) < 2:
        return {"lr": 0.5, "rf": 0.5, "mean": 0.5, "degenerate": True}
    out = {}
    lr = LogisticRegression(max_iter=2000).fit(train_X, train_y)
    out["lr"] = float(roc_auc_score(test_y, lr.predict_proba(test_X)[:, 1]))
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=4).fit(train_X, train_y)
    out["rf"] = float(roc_auc_score(test_y, rf.predict_proba(test_X)[:, 1]))
    out["mean"] = (out["lr"] + out["rf"]) / 2
    out["degenerate"] = False
    return out


def tabular_metrics(fake: np.ndarray, train: np.ndarray, test: np.ndarray, seed=0) -> dict:
    mt, per_col = tabular_tvd(fake, train)
    ml = ml_efficacy(fake[:, :-1], (fake[:, -1] > 0).astype(int),
                     test[:, :-1], (test[:, -1] > 0).astype(int), seed)
    return {"mean_tvd": mt, "tvd_per_col": per_col.tolist(),
            "corr_diff": corr_diff(fake, train), "ml_eff": ml["mean"],
            "ml_eff_lr": ml["lr"], "ml_eff_rf": ml["rf"],
            "ml_degenerate": ml["degenerate"],
            "frac_malignant": float((fake[:, -1] > 0).mean())}


# =============================================================== time series
class _GRUHead(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_out)

    def forward(self, x):
        h, hn = self.gru(x)
        return h, hn[-1]


def disc_score(fake: np.ndarray, real: np.ndarray, seed: int = 0,
               steps: int = 2000, hidden: int = 32) -> float:
    """|acc - 0.5| of a GRU classifier (TimeGAN post-hoc discriminator)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = min(len(fake), len(real))
    r = real[rng.choice(len(real), n, replace=False)]
    f = fake[rng.choice(len(fake), n, replace=False)]
    X = np.concatenate([r, f]).astype(np.float32)
    Y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)
    idx = rng.permutation(2 * n)
    cut = int(0.8 * 2 * n)
    tr, te = idx[:cut], idx[cut:]
    net = _GRUHead(X.shape[2], hidden, 1)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for _ in range(steps):
        b = torch.from_numpy(rng.choice(tr, 128))
        _, h = net(Xt[b])
        loss = nn.functional.binary_cross_entropy_with_logits(net.out(h).view(-1), Yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, h = net(Xt[te])
        acc = float(((net.out(h).view(-1) > 0).float() == Yt[te]).float().mean())
    return abs(acc - 0.5)


def pred_mae(train_windows: np.ndarray, test_windows: np.ndarray, seed: int = 0,
             steps: int = 2000, hidden: int = 64) -> float:
    """Train GRU on ``train_windows`` to predict x_{t+1} from x_{<=t};
    return MAE on ``test_windows`` (TSTR when train_windows are synthetic)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = train_windows.shape[2]
    net = _GRUHead(d, hidden, d)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.from_numpy(train_windows.astype(np.float32))
    for _ in range(steps):
        b = torch.from_numpy(rng.integers(0, len(Xt), 128))
        h, _ = net(Xt[b, :-1])
        loss = (net.out(h) - Xt[b, 1:]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    Xe = torch.from_numpy(test_windows.astype(np.float32))
    with torch.no_grad():
        h, _ = net(Xe[:, :-1])
        return float((net.out(h) - Xe[:, 1:]).abs().mean())


def stock_metrics(fake: np.ndarray, train: np.ndarray, test: np.ndarray, seed=0) -> dict:
    """fake/train/test: (N, 24, 50)."""
    pooled_r = train.reshape(-1, train.shape[2])
    pooled_f = fake.reshape(-1, fake.shape[2])
    tv = np.mean([_col_tvd(pooled_r[:, j], pooled_f[:, j], 50) for j in range(pooled_r.shape[1])])
    return {"disc_score": disc_score(fake, train, seed),
            "pred_mae": pred_mae(fake, test, seed),
            "corr_diff": corr_diff(pooled_f, pooled_r),
            "mean_tvd": float(tv)}


if __name__ == "__main__":
    if not os.path.exists(LENET_PATH):
        print("LeNet test accuracy:", train_lenet())
    ck = torch.load(LENET_PATH, weights_only=False)
    print(json.dumps({"lenet_test_acc": ck["test_acc"]}))
