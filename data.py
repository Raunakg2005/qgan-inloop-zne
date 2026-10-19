"""
data.py — Dataset preparation and loading for the three domains
================================================================

    mnist    : MNIST handwritten digits (torchvision), 28x28 -> 784, scaled to [-1, 1]
    tabular  : Breast Cancer Wisconsin (Diagnostic), 569 patients, 30 real-valued
               features + binary diagnosis label (scikit-learn bundled copy of
               the UCI dataset).  Stratified 80/20 train/test split.
    stocks   : Daily adjusted close prices of 50 US large-cap equities
               (Yahoo Finance, 2014-01-01 .. 2024-12-31) -> daily log-returns,
               standardised per asset with training-period statistics,
               sliding windows of length 24.  Chronological 80/20 split.

Run once to download / cache everything:
    python data.py --prepare
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "XOM", "PG", "V",
    "MA", "UNH", "HD", "CVX", "MRK", "PFE", "KO", "PEP", "WMT", "BAC",
    "INTC", "CSCO", "ORCL", "T", "VZ", "DIS", "MCD", "NKE", "IBM", "ADBE",
    "CRM", "NVDA", "QCOM", "TXN", "AMGN", "GILD", "BMY", "ABT", "MDT", "HON",
    "CAT", "MMM", "GE", "BA", "LMT", "UPS", "GS", "MS", "C", "WFC",
]
STOCK_START, STOCK_END = "2014-01-01", "2025-01-01"   # end is exclusive in yfinance
WINDOW = 24
SPLIT_SEED = 0


# ----------------------------------------------------------------- MNIST
def load_mnist(train: bool = True) -> np.ndarray:
    """Return MNIST images as float32 array (N, 784) in [-1, 1], plus labels."""
    from torchvision import datasets
    ds = datasets.MNIST(root=DATA_DIR, train=train, download=True)
    x = ds.data.numpy().reshape(-1, 784).astype(np.float32) / 127.5 - 1.0
    return x, ds.targets.numpy()


# ----------------------------------------------------------------- tabular
def load_tabular():
    """Breast Cancer Wisconsin.  Returns dict with standardised train/test
    matrices whose last column is the label coded as -1 (benign) / +1
    (malignant), plus the raw-unit arrays and scaler for reporting."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    bc = load_breast_cancer()
    X = bc.data.astype(np.float64)
    y = (bc.target == 0).astype(np.float64)          # 1 = malignant
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=SPLIT_SEED)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    f = lambda A, b: np.hstack([(A - mu) / sd, (2 * b - 1)[:, None]]).astype(np.float32)
    return {
        "train": f(Xtr, ytr), "test": f(Xte, yte),
        "feature_names": list(bc.feature_names) + ["diagnosis"],
        "mu": mu, "sd": sd,
    }


# ----------------------------------------------------------------- stocks
def _stock_csv() -> str:
    return os.path.join(DATA_DIR, "stocks_adjclose.csv")


def download_stocks():
    import yfinance as yf
    df = yf.download(TICKERS, start=STOCK_START, end=STOCK_END,
                     auto_adjust=True, progress=False, threads=False)["Close"]
    df = df[TICKERS]
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(_stock_csv())
    meta = {"source": "Yahoo Finance via yfinance", "tickers": TICKERS,
            "start": STOCK_START, "end_exclusive": STOCK_END,
            "rows": int(len(df)), "missing_values": int(df.isna().sum().sum())}
    with open(os.path.join(DATA_DIR, "stocks_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return df


def load_stocks():
    """Return dict with train/test windows (N, WINDOW, 50) of standardised
    log-returns.  The split is chronological on the return series."""
    import pandas as pd
    df = pd.read_csv(_stock_csv(), index_col=0, parse_dates=True)
    if df.isna().any().any():
        raise ValueError("stock price table contains missing values")
    r = np.log(df.values[1:] / df.values[:-1])               # (T-1, 50)
    T = len(r)
    cut = int(0.8 * T)
    mu, sd = r[:cut].mean(0), r[:cut].std(0)
    z = ((r - mu) / sd).astype(np.float32)

    def windows(a):
        return np.stack([a[i:i + WINDOW] for i in range(len(a) - WINDOW + 1)])

    return {"train": windows(z[:cut]), "test": windows(z[cut:]),
            "dates": df.index[1:], "cut_date": str(df.index[1:][cut].date()),
            "mu": mu, "sd": sd, "tickers": list(df.columns)}


# ----------------------------------------------------------------- common
def load_domain(domain: str):
    """Return (train_array_flat, info) with train_array_flat of shape (N, d)."""
    if domain == "mnist":
        x, _ = load_mnist(train=True)
        return x, {"shape": (1, 28, 28)}
    if domain == "tabular":
        t = load_tabular()
        return t["train"], {"shape": (t["train"].shape[1],)}
    if domain == "stocks":
        s = load_stocks()
        tr = s["train"]
        return tr.reshape(len(tr), -1), {"shape": tr.shape[1:]}
    raise ValueError(domain)


def batches(x: np.ndarray, batch: int, gen: torch.Generator):
    """Infinite iterator over shuffled minibatches (drop_last)."""
    xt = torch.from_numpy(x)
    while True:
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt) - batch + 1, batch):
            yield xt[perm[i:i + batch]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        os.makedirs(DATA_DIR, exist_ok=True)
        xtr, _ = load_mnist(True)
        xte, _ = load_mnist(False)
        print("MNIST", xtr.shape, xte.shape)
        t = load_tabular()
        print("Breast cancer train/test", t["train"].shape, t["test"].shape)
        df = download_stocks()
        print("Stocks", df.shape, "missing", int(df.isna().sum().sum()),
              df.index[0].date(), "->", df.index[-1].date())
        s = load_stocks()
        print("Stock windows train/test", s["train"].shape, s["test"].shape,
              "cut", s["cut_date"])
