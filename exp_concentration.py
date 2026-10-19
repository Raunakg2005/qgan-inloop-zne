"""
exp_concentration.py — Spread of the circuit outputs across latent samples
==========================================================================
For the ideal QGenNet circuit (n = 12, L = 6) and R random parameter draws,
computes the standard deviation of <Z_i> over 2000 latent samples
z ~ N(0, I), averaged over qubits, with and without the ring-CNOT layers,
for small-angle (U[-0.1 pi, 0.1 pi]) and uniform (U[-pi, pi]) parameters.

    python exp_concentration.py        -> results/concentration.json
"""

import json
import math
import os

import numpy as np
import torch

from qsim import RingPQC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(n=12, L=6, n_z=2000, R=10, seed=0):
    torch.manual_seed(seed)
    rows = []
    for entangle in (True, False):
        sim = RingPQC(n, L, entangle=entangle)
        for init, scale in (("small_0.1pi", 0.1), ("uniform_pi", 1.0)):
            stds = []
            for _ in range(R):
                th = scale * math.pi * (2 * torch.rand(2, L, n) - 1)
                z = torch.randn(n_z, n)
                stds.append(sim.expvals(z, th).std(0).mean().item())
            rows.append({"entangle": entangle, "init": init, "R": R, "n_z": n_z,
                         "mean_std": float(np.mean(stds)), "sd_over_draws": float(np.std(stds, ddof=1))})
            print(rows[-1], flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "concentration.json"), "w") as fh:
        json.dump({"n": n, "L": L, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
