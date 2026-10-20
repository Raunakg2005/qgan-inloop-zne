"""
make_figures.py — Build every figure in the paper from logged results
=====================================================================
Reads runs/*/seed*/{log,metrics}.json and results/*.json; writes PDFs to
../figures/.  No plotted value is typed in by hand.

    python make_figures.py [--out DIR]    (default: outputs/)
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "outputs", "figures")   # overridden by --out

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "savefig.bbox": "tight", "savefig.dpi": 300})
C = {"qgennet_zne_p0.001": "#5B2A86", "qgennet_nozne_p0.001": "#B07CC6",
     "qgennet_noent": "#999999", "decoder_only": "#E69F00", "mlp": "#0072B2",
     "dcgan": "#009E73", "ctgan": "#009E73", "timegan": "#009E73"}
LABEL = {"qgennet_zne_p0.001": "QGenNet (ZNE)", "qgennet_nozne_p0.001": "QGenNet (no ZNE)",
         "qgennet_noent": "QGenNet (no ent.)", "decoder_only": "Decoder only",
         "mlp": "MLP WGAN-GP", "dcgan": "DCGAN-G", "ctgan": "CTGAN", "timegan": "TimeGAN"}


def curves(domain, key):
    out = {}
    for f in glob.glob(os.path.join(RUNS, domain, "*", "seed*", "log.json")):
        rn = f.split(os.sep)[-3]
        if "pilot" in rn or "smoke" in rn or rn not in LABEL:
            continue
        lg = json.load(open(f))
        out.setdefault(rn, []).append((np.array(lg["eval_iter"]),
                                       np.array([e[key] for e in lg["eval"]])))
    return out


def fig_training():
    panels = [("mnist", "fd_lenet", r"FD$_{\rm LeNet}$ (2k samples)", True),
              ("tabular", "mean_tvd", "Mean TVD (2k samples)", False),
              ("stocks", "corr_diff", "Cross-asset corr. diff.", False)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    fig.subplots_adjust(wspace=0.5)
    for ax, (dom, key, yl, logy) in zip(axes, panels):
        for rn, runs in sorted(curves(dom, key).items()):
            L = min(len(r[0]) for r in runs)
            it = runs[0][0][:L]
            ys = np.stack([r[1][:L] for r in runs])
            m, s = ys.mean(0), ys.std(0)
            ax.plot(it, m, color=C[rn], label=f"{LABEL[rn]} (n={len(runs)})", lw=1.2)
            ax.fill_between(it, m - s, m + s, color=C[rn], alpha=0.15, lw=0)
        ax.set_xlabel("Generator iteration")
        ax.set_ylabel(yl)
        ax.set_title({"mnist": "MNIST", "tabular": "Tabular", "stocks": "Equities"}[dom])
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, lw=0.5)
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    axes[-1].legend(handles.values(), handles.keys(), loc="upper left",
                    bbox_to_anchor=(1.02, 1.0))
    fig.savefig(os.path.join(FIG, "fig_training_curves.pdf"))
    plt.close(fig)


def fig_gradvar():
    f = os.path.join(RES, "gradvar.json")
    if not os.path.exists(f):
        return
    g = json.load(open(f))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rng_label = {"uniform_pi": r"$\theta\sim U[-\pi,\pi]$",
                 "small_0.1pi": r"$\theta\sim U[-0.1\pi,0.1\pi]$"}
    for init, ls in (("uniform_pi", "-"), ("small_0.1pi", "--")):
        rows = [r for r in g["rows"] if r["init"] == init]
        n = [r["n"] for r in rows]
        ax.semilogy(n, [r["var_local"] for r in rows], ls, marker="o", ms=3, color="#5B2A86",
                    label=f"local, {rng_label[init]}")
        ax.semilogy(n, [r["var_global"] for r in rows], ls, marker="s", ms=3, color="#D55E00",
                    label=f"global, {rng_label[init]}")
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel(r"Var$[\partial C/\partial\theta_1]$")
    ax.set_title(f"L = {g['L']}, K = {g['K']} parameter draws")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(FIG, "fig_gradvar.pdf"))
    plt.close(fig)


def fig_zne_bias():
    files = sorted(glob.glob(os.path.join(RES, "zne_bias_n*.json")),
                   key=lambda f: int(f.rsplit("_n", 1)[1].split(".")[0]))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for f, mk in zip(files, "osd^"):
        z = json.load(open(f))
        n = z["args"]["n"]
        p = [r["p"] for r in z["rows"]]
        ax.loglog(p, [r["bias_unmitigated"] for r in z["rows"]], "-", marker=mk, ms=3,
                  color="#D55E00", label=f"unmitigated, n={n}")
        ax.loglog(p, [r["bias_zne"] for r in z["rows"]], "-", marker=mk, ms=3,
                  color="#5B2A86", label=f"ZNE (1,3), n={n}")
    ax.set_xlabel("Depolarising probability $p$ per CNOT qubit")
    ax.set_ylabel(r"mean $|\langle Z_i\rangle - \langle Z_i\rangle_{0}|$")
    ax.grid(alpha=0.3, lw=0.5, which="both")
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(FIG, "fig_zne_bias.pdf"))
    plt.close(fig)


def fig_noise_sweep():
    rows = {}
    for f in glob.glob(os.path.join(RUNS, "tabular", "qgennet_*zne_p*", "seed*", "metrics.json")):
        rn = f.split(os.sep)[-3]
        if "pilot" in rn or "smoke" in rn or "nonorm" in rn:
            continue
        zne = rn.startswith("qgennet_zne")
        p = float(rn.split("_p")[-1])
        rows.setdefault((zne, p), []).append(json.load(open(f))["final"]["mean_tvd"])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for zne, col, lab in ((False, "#D55E00", "no ZNE"), (True, "#5B2A86", "ZNE (1,3)")):
        ps = sorted(p for (z, p) in rows if z == zne)
        m = [np.mean(rows[(zne, p)]) for p in ps]
        s = [np.std(rows[(zne, p)], ddof=1) if len(rows[(zne, p)]) > 1 else 0 for p in ps]
        ax.errorbar(ps, m, yerr=s, marker="o", ms=3, capsize=2, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Depolarising probability $p$")
    ax.set_ylabel("Mean TVD (tabular) $\\downarrow$")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend()
    fig.savefig(os.path.join(FIG, "fig_noise_sweep.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "outputs"),
                    help="figures are written to <out>/figures (use .. inside the paper folder)")
    FIG = os.path.join(ap.parse_args().out, "figures")
    os.makedirs(FIG, exist_ok=True)
    fig_training()
    fig_gradvar()
    fig_zne_bias()
    fig_noise_sweep()
    print("figures written to", os.path.abspath(FIG))
