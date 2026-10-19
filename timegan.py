"""
timegan.py — TimeGAN baseline (PyTorch re-implementation of Yoon et al. 2019)
=============================================================================
Follows the structure and losses of the authors' reference implementation
(github.com/jsyoon0823/TimeGAN):

    embedder  E: X -> H        recovery  R: H -> X
    generator G: Z -> E_hat    supervisor S: H -> H (one step ahead)
    discriminator D: H -> logit

Phase 1: embedding network (E, R) on reconstruction loss.
Phase 2: supervisor S on the supervised one-step loss.
Phase 3: joint training (generator twice per discriminator step, as in the
         reference code; discriminator only updated when its loss > 0.15).

Inputs are min-max scaled to [0, 1] with training-set statistics (as in the
reference implementation) and mapped back before evaluation, so the metric
code is identical to the other models.

    python timegan.py --seed 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import metrics as M
from data import load_stocks
from train_gan import RUNS


class Net(nn.Module):
    def __init__(self, d_in, hidden, d_out, layers, sigmoid=True):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, d_out)
        self.sigmoid = sigmoid

    def forward(self, x):
        h, _ = self.rnn(x)
        y = self.fc(h)
        return torch.sigmoid(y) if self.sigmoid else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5000, help="iterations per phase")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    s = load_stocks()
    tr, te = s["train"], s["test"]
    lo, hi = tr.min(axis=(0, 1)), tr.max(axis=(0, 1))
    X = torch.from_numpy(((tr - lo) / (hi - lo + 1e-7)).astype(np.float32))
    N, T, d = X.shape
    h = a.hidden
    E = Net(d, h, h, a.layers)
    R = Net(h, h, d, a.layers)
    G = Net(d, h, h, a.layers)
    S = Net(h, h, h, a.layers - 1 if a.layers > 1 else 1)
    D = Net(h, h, 1, a.layers, sigmoid=False)
    opt = lambda *ms: torch.optim.Adam([p for m in ms for p in m.parameters()], 1e-3)
    oER, oS, oGS, oD, oE = opt(E, R), opt(S), opt(G, S), opt(D), opt(E, R)
    batch = lambda: X[torch.from_numpy(rng.integers(0, N, a.batch))]
    noise = lambda: torch.rand(a.batch, T, d)
    t0 = time.time()

    for it in range(a.iters):                                   # phase 1
        x = batch()
        loss = 10 * torch.sqrt(F.mse_loss(R(E(x)), x))
        oER.zero_grad(); loss.backward(); oER.step()
    for it in range(a.iters):                                   # phase 2
        H = E(batch()).detach()
        loss = F.mse_loss(S(H)[:, :-1], H[:, 1:])
        oS.zero_grad(); loss.backward(); oS.step()
    for it in range(a.iters):                                   # phase 3
        for _ in range(2):
            x, z = batch(), noise()
            H = E(x)
            Eh = G(z)
            Hh = S(Eh)
            Xh = R(Hh)
            yf, yfe = D(Hh), D(Eh)
            g_u = F.binary_cross_entropy_with_logits(yf, torch.ones_like(yf))
            g_ue = F.binary_cross_entropy_with_logits(yfe, torch.ones_like(yfe))
            g_s = F.mse_loss(S(H.detach())[:, :-1], H.detach()[:, 1:])
            v1 = (torch.sqrt(Xh.var(0) + 1e-6) - torch.sqrt(x.var(0) + 1e-6)).abs().mean()
            v2 = (Xh.mean(0) - x.mean(0)).abs().mean()
            loss_g = g_u + a.gamma * g_ue + 100 * torch.sqrt(g_s) + 100 * (v1 + v2)
            oGS.zero_grad(); loss_g.backward(); oGS.step()
            H = E(x)
            e_t0 = 10 * torch.sqrt(F.mse_loss(R(H), x))
            loss_e = e_t0 + 0.1 * F.mse_loss(S(H)[:, :-1], H[:, 1:])
            oE.zero_grad(); loss_e.backward(); oE.step()
        x, z = batch(), noise()
        with torch.no_grad():
            H, Eh = E(x), G(z)
            Hh = S(Eh)
        yr, yf, yfe = D(H), D(Hh), D(Eh)
        loss_d = (F.binary_cross_entropy_with_logits(yr, torch.ones_like(yr))
                  + F.binary_cross_entropy_with_logits(yf, torch.zeros_like(yf))
                  + a.gamma * F.binary_cross_entropy_with_logits(yfe, torch.zeros_like(yfe)))
        if loss_d.item() > 0.15:
            oD.zero_grad(); loss_d.backward(); oD.step()
        if (it + 1) % 1000 == 0:
            print(f"[timegan s{a.seed}] joint it {it + 1}  D {loss_d.item():.3f}  "
                  f"G_u {g_u.item():.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    secs = time.time() - t0

    with torch.no_grad():
        out = []
        for i in range(0, N, 256):
            z = torch.rand(min(256, N - i), T, d)
            out.append(R(S(G(z))))
        fake = torch.cat(out).numpy() * (hi - lo + 1e-7) + lo
    res = M.stock_metrics(fake.astype(np.float32), tr, te, a.seed)
    path = os.path.join(RUNS, "stocks", "timegan", f"seed{a.seed}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(dict(vars(a), model="timegan"), fh, indent=2)
    with open(os.path.join(path, "metrics.json"), "w") as fh:
        json.dump({"final": res, "train_seconds": secs}, fh, indent=2)
    print("TimeGAN seed", a.seed, res)


if __name__ == "__main__":
    main()
