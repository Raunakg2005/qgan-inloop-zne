"""
exp_zne_bias.py — Exact bias of unmitigated vs. ZNE expectation values
======================================================================
Exact density-matrix simulation (PennyLane ``default.mixed``) of the QGenNet
ansatz with depolarising noise after every CNOT.  For each noise rate p we
compute, for every qubit i and latent sample z,

    unmitigated bias  | <Z_i>_p           - <Z_i>_0 |
    ZNE bias          | 3/2 <Z_i>_p - 1/2 <Z_i>_{3p, folded} - <Z_i>_0 |

where the scale-3 circuit replaces every CNOT by CNOT CNOT CNOT (each
followed by the same noise channel), exactly as in qsim.RingPQC.  There is
no sampling error: these are exact expectation values.

    python exp_zne_bias.py --n 8            -> results/zne_bias_n8.json
"""

import argparse
import json
import math
import os

import numpy as np
import pennylane as qml
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def make(n, L, p, scale):
    dev = qml.device("default.mixed" if p > 0 else "default.qubit", wires=n)

    @qml.qnode(dev)
    def circ(z, th):
        for i in range(n):
            qml.RY(z[i], wires=i)
        for l in range(L):
            for i in range(n):
                qml.RY(th[0, l, i], wires=i)
                qml.RZ(th[1, l, i], wires=i)
            for i in range(n):
                for _ in range(scale):
                    qml.CNOT(wires=[i, (i + 1) % n])
                    if p > 0:
                        qml.DepolarizingChannel(p, wires=i)
                        qml.DepolarizingChannel(p, wires=(i + 1) % n)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--n_z", type=int, default=8)
    ap.add_argument("--init", type=float, default=0.1, help="theta ~ U(-init*pi, init*pi)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ps = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    rng = np.random.default_rng(a.seed)
    th = a.init * math.pi * (2 * rng.random((2, a.L, a.n)) - 1)
    zs = rng.standard_normal((a.n_z, a.n))
    ideal = np.array([make(a.n, a.L, 0.0, 1)(z, th) for z in zs])
    rows = []
    for p in ps:
        e1 = np.array([make(a.n, a.L, p, 1)(z, th) for z in zs])
        e3 = np.array([make(a.n, a.L, p, 3)(z, th) for z in zs])
        zne = 1.5 * e1 - 0.5 * e3
        row = {"p": p,
               "bias_unmitigated": float(np.abs(e1 - ideal).mean()),
               "bias_zne": float(np.abs(zne - ideal).mean()),
               "max_bias_unmitigated": float(np.abs(e1 - ideal).max()),
               "max_bias_zne": float(np.abs(zne - ideal).max()),
               "mean_abs_ideal": float(np.abs(ideal).mean())}
        rows.append(row)
        print(row, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"zne_bias_n{a.n}.json"), "w") as fh:
        json.dump({"args": vars(a), "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
