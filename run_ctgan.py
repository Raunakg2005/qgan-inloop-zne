"""
run_ctgan.py — CTGAN baseline (official ``ctgan`` package, Xu et al. 2019)
==========================================================================
CTGAN is trained on the same standardised Breast Cancer training split; the
diagnosis label is passed as a discrete column.  Evaluation uses exactly the
same metric code as train_gan.py.

    python run_ctgan.py --seed 0 [--epochs 300]
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import metrics as M
from data import load_tabular
from train_gan import RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
from ctgan import CTGAN  # noqa: E402

t = load_tabular()
cols = [f"f{j}" for j in range(t["train"].shape[1] - 1)] + ["label"]
df = pd.DataFrame(t["train"], columns=cols)
df["label"] = (df["label"] > 0).astype(int).astype(str)

name = "ctgan" if a.epochs == 300 else f"ctgan_e{a.epochs}"
out = os.path.join(RUNS, "tabular", name, f"seed{a.seed}")
os.makedirs(out, exist_ok=True)
model = CTGAN(epochs=a.epochs, batch_size=60, pac=10, verbose=False, cuda=False)
model.set_random_state(a.seed)
t0 = time.time()
model.fit(df, discrete_columns=["label"])
secs = time.time() - t0


def to_array(s: pd.DataFrame) -> np.ndarray:
    s = s.copy()
    s["label"] = np.where(s["label"].astype(int) == 1, 1.0, -1.0)
    return s[cols].values.astype(np.float32)


fid = M.tabular_metrics(to_array(model.sample(5000)), t["train"], t["test"], a.seed)
ml = M.tabular_metrics(to_array(model.sample(len(t["train"]))), t["train"], t["test"], a.seed)
fid.update({k: ml[k] for k in ("ml_eff", "ml_eff_lr", "ml_eff_rf", "ml_degenerate")})
with open(os.path.join(out, "config.json"), "w") as fh:
    json.dump(dict(vars(a), model="ctgan", batch_size=60, pac=10), fh, indent=2)
with open(os.path.join(out, "metrics.json"), "w") as fh:
    json.dump({"final": fid, "train_seconds": secs}, fh, indent=2)
print("CTGAN seed", a.seed, {k: fid[k] for k in ("mean_tvd", "corr_diff", "ml_eff")})
