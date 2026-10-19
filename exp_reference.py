"""
exp_reference.py — Real-data reference values for the metrics
=============================================================
Values a perfect generator could at best approach, computed with exactly
the same metric code as for the generators:

tabular
    ml_eff  (TRTR)   classifiers trained on the real training split
    mean_tvd, corr_diff  of the real *test* split (114 rows) against the
                     real training split.  The small test split makes these
                     larger than the true noise floor; they are reported as
                     an indication only.
stocks
    pred_mae (TRTR)  GRU trained on real training windows
    disc_score, corr_diff, mean_tvd  of real test windows (2022-10-18 ..
                     2024-12-31) against real training windows; this
                     measures distribution shift between the two periods.
mnist
    fd_lenet, ndb_k  of 10,000 real training images against the test set.

    python exp_reference.py          -> results/reference.json
"""

import json
import os

import numpy as np
import torch

import metrics as M
from data import load_mnist, load_stocks, load_tabular

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(seeds=(0, 1, 2, 3, 4)):
    torch.set_num_threads(2)
    res = {}
    t = load_tabular()
    tr, te = t["train"], t["test"]
    ml = [M.ml_efficacy(tr[:, :-1], (tr[:, -1] > 0).astype(int),
                        te[:, :-1], (te[:, -1] > 0).astype(int), s)["mean"] for s in seeds]
    res["tabular"] = {"ml_eff_trtr": {"mean": float(np.mean(ml)), "std": float(np.std(ml, ddof=1))},
                      "mean_tvd_test_vs_train": M.tabular_tvd(te, tr)[0],
                      "corr_diff_test_vs_train": M.corr_diff(te, tr)}
    print(res["tabular"], flush=True)

    s = load_stocks()
    tr, te = s["train"], s["test"]
    mae = [M.pred_mae(tr, te, k) for k in seeds]
    ds = [M.disc_score(te, tr, k) for k in seeds]
    pr, pt = tr.reshape(-1, tr.shape[2]), te.reshape(-1, te.shape[2])
    res["stocks"] = {"pred_mae_trtr": {"mean": float(np.mean(mae)), "std": float(np.std(mae, ddof=1))},
                     "disc_score_test_vs_train": {"mean": float(np.mean(ds)), "std": float(np.std(ds, ddof=1))},
                     "corr_diff_test_vs_train": M.corr_diff(pt, pr),
                     "mean_real_offdiag_corr_train": float(np.corrcoef(pr, rowvar=False)[np.triu_indices(pr.shape[1], 1)].mean())}
    print(res["stocks"], flush=True)

    xtr, _ = load_mnist(True)
    xte, _ = load_mnist(False)
    sub = xtr[np.random.default_rng(0).choice(len(xtr), 10000, replace=False)]
    res["mnist"] = {"fd_lenet_train_vs_test": M.fd_lenet(sub, xte), "ndb_k_train_vs_test": M.ndb(sub, xte)}
    print(res["mnist"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "reference.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
