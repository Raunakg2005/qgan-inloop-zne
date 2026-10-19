"""
train_gan.py — WGAN-GP training for QGenNet and the classical baselines
=======================================================================

All models share the same critic, loss (WGAN-GP, lambda = 10), optimiser
(Adam, beta = (0.5, 0.9)), batch size (64), n_critic (5) and training budget
(number of generator iterations), so differences come from the generator.

Examples
    python train_gan.py --domain tabular --model qgennet --seed 0
    python train_gan.py --domain tabular --model qgennet --no_zne --seed 0
    python train_gan.py --domain mnist   --model dcgan   --seed 0

Everything a run produces goes to runs/<domain>/<run_name>/seed<k>/:
    config.json   all arguments + parameter counts
    log.json      losses every --log_every iterations, quick metric curve,
                  wall-clock time
    metrics.json  final metrics (and, for QGenNet, metrics of the same
                  generator evaluated on the ideal noise-free circuit)
    gen.pt        final generator weights
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import metrics as M
from data import batches, load_domain, load_stocks, load_tabular, load_mnist
from nets import (ConvCritic, DCGANGen, DecoderOnly, MLPCritic, MLPGen, QGenNet,
                  gradient_penalty, n_params)

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["mnist", "tabular", "stocks"], required=True)
    ap.add_argument("--model", choices=["qgennet", "decoder_only", "mlp", "dcgan"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen_iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--lambda_gp", type=float, default=10.0)
    ap.add_argument("--lr_q", type=float, default=1e-3, help="PQC angles")
    ap.add_argument("--lr_c", type=float, default=1e-4, help="classical generator/decoder and critic")
    ap.add_argument("--n_qubits", type=int, default=12)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--p", type=float, default=0.001, help="depolarising prob. after each CNOT")
    ap.add_argument("--no_zne", action="store_true")
    ap.add_argument("--no_entangle", action="store_true")
    ap.add_argument("--no_feat_norm", action="store_true",
                    help="QGenNet: feed raw <Z_i> to the decoder (no BatchNorm)")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", type=str, default="")
    return ap.parse_args(argv)


def run_name(a) -> str:
    if a.model == "qgennet":
        if a.no_entangle:
            name = "qgennet_noent"
        else:
            name = f"qgennet_{'nozne' if a.no_zne else 'zne'}_p{a.p:g}"
        if a.no_feat_norm:
            name += "_nonorm"
    else:
        name = a.model
    return name + (f"_{a.tag}" if a.tag else "")


def build(a, d: int):
    tanh = a.domain == "mnist"
    if a.model == "qgennet":
        G = QGenNet(d, a.n_qubits, a.n_layers, p=a.p, zne=not a.no_zne,
                    entangle=not a.no_entangle, tanh_out=tanh,
                    noise_seed=10_000 + a.seed, feat_norm=not a.no_feat_norm)
    elif a.model == "decoder_only":
        G = DecoderOnly(d, a.n_qubits, tanh_out=tanh)
    elif a.model == "mlp":
        G = MLPGen(d, tanh_out=tanh)
    elif a.model == "dcgan":
        assert a.domain == "mnist"
        G = DCGANGen()
    D = ConvCritic() if a.domain == "mnist" else MLPCritic(d)
    return G, D


@torch.no_grad()
def sample(G, n: int, chunk: int = 256) -> np.ndarray:
    G.eval()
    out = [G(torch.randn(min(chunk, n - i), G.latent_dim)) for i in range(0, n, chunk)]
    G.train()
    return torch.cat(out).numpy()


class Evaluator:
    """Quick (during training) and final metrics for one domain."""

    def __init__(self, domain: str):
        self.domain = domain
        if domain == "mnist":
            self.test, _ = load_mnist(False)
        elif domain == "tabular":
            t = load_tabular()
            self.train, self.test = t["train"], t["test"]
        else:
            s = load_stocks()
            self.train, self.test = s["train"], s["test"]

    def quick(self, G) -> dict:
        if self.domain == "mnist":
            return {"fd_lenet": M.fd_lenet(sample(G, 2000), self.test[:2000])}
        if self.domain == "tabular":
            f = sample(G, 2000)
            return {"mean_tvd": M.tabular_tvd(f, self.train)[0],
                    "corr_diff": M.corr_diff(f, self.train)}
        f = sample(G, len(self.train)).reshape(self.train.shape)
        pr, pf = self.train.reshape(-1, 50), f.reshape(-1, 50)
        return {"corr_diff": M.corr_diff(pf, pr)}

    def final(self, G, seed: int) -> dict:
        if self.domain == "mnist":
            f = sample(G, 10_000)
            return {"fd_lenet": M.fd_lenet(f, self.test), "ndb_k": M.ndb(f, self.test)}
        if self.domain == "tabular":
            fid = M.tabular_metrics(sample(G, 5000), self.train, self.test, seed)
            ml = M.tabular_metrics(sample(G, len(self.train)), self.train, self.test, seed)
            fid.update({k: ml[k] for k in ("ml_eff", "ml_eff_lr", "ml_eff_rf", "ml_degenerate")})
            return fid
        f = sample(G, len(self.train)).reshape(self.train.shape)
        return M.stock_metrics(f, self.train, self.test, seed)


def train(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    x, info = load_domain(a.domain)
    d = x.shape[1]
    G, D = build(a, d)
    out = os.path.join(RUNS, a.domain, run_name(a), f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    cfg = dict(vars(a), run_name=run_name(a), data_dim=d, n_train=len(x),
               gen_params=n_params(G), critic_params=n_params(D),
               torch_version=torch.__version__)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    opt_g = torch.optim.Adam(G.param_groups(a.lr_q, a.lr_c), betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=a.lr_c, betas=(0.5, 0.9))
    data_it = batches(x, a.batch, torch.Generator().manual_seed(a.seed))
    ev = Evaluator(a.domain)
    log = {"iter": [], "d_loss": [], "g_loss": [], "w_est": [],
           "eval_iter": [], "eval": [], "seconds": None}
    t0 = time.time()
    dl, gl, we = [], [], []
    for it in range(1, a.gen_iters + 1):
        for _ in range(a.n_critic):
            real = next(data_it)
            with torch.no_grad():
                fake = G(torch.randn(a.batch, G.latent_dim))
            d_real, d_fake = D(real).mean(), D(fake).mean()
            loss_d = d_fake - d_real + gradient_penalty(D, real, fake, a.lambda_gp)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
        dl.append(loss_d.item())
        we.append((d_real - d_fake).item())
        loss_g = -D(G(torch.randn(a.batch, G.latent_dim))).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
        gl.append(loss_g.item())
        if it % a.log_every == 0:
            log["iter"].append(it)
            log["d_loss"].append(float(np.mean(dl)))
            log["g_loss"].append(float(np.mean(gl)))
            log["w_est"].append(float(np.mean(we)))
            dl, gl, we = [], [], []
        if it % a.eval_every == 0 or it == a.gen_iters:
            q = ev.quick(G)
            log["eval_iter"].append(it)
            log["eval"].append(q)
            print(f"[{run_name(a)} s{a.seed}] it {it:5d}  W~{log['w_est'][-1]:+.3f}  "
                  f"{q}  {(time.time() - t0) / 60:.1f} min", flush=True)
    log["seconds"] = time.time() - t0
    with open(os.path.join(out, "log.json"), "w") as fh:
        json.dump(log, fh)
    torch.save(G.state_dict(), os.path.join(out, "gen.pt"))

    res = {"final": ev.final(G, a.seed)}
    if a.model == "qgennet" and not a.no_entangle and a.p > 0:
        # same trained generator, evaluated on the ideal (noise-free) circuit
        p_saved = G.p
        G.p = 0.0
        res["final_ideal_circuit"] = ev.final(G, a.seed)
        G.p = p_saved
    res["train_seconds"] = log["seconds"]
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[{run_name(a)} s{a.seed}] DONE {json.dumps(res['final'])[:300]}", flush=True)
    return res


if __name__ == "__main__":
    train(get_args())
