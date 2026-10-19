"""
exp_gradvar.py — Gradient variance vs. qubit count (barren-plateau check)
=========================================================================
For the QGenNet ansatz (R_y(z) embedding, L layers of R_y R_z + ring-CNOT,
ideal circuit) we estimate Var_theta[ dC / d theta_1 ] over K random
parameter draws for

    local cost   C_loc  = (1/n) sum_i <Z_i>
    global cost  C_glob = <0...0| rho |0...0>   (all-zero projector)

and two parameter distributions: uniform on [-pi, pi] ("random init") and
uniform on [-0.1 pi, 0.1 pi] (the small-angle init used for training).
theta_1 is the R_y angle of qubit 0 in the first variational layer.  The
latent z is drawn from N(0, I) for every sample.

    python exp_gradvar.py            -> results/gradvar.json
"""

import json
import math
import os

import numpy as np
import torch

from qsim import RingPQC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(ns=(4, 6, 8, 10, 12, 14, 16), L=6, K=400, seed=0):
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    res = {"L": L, "K": K, "param": "theta[0, layer 0, qubit 0]", "rows": []}
    for n in ns:
        sim = RingPQC(n, L, dtype=torch.complex128)
        for init_name, scale in (("uniform_pi", 1.0), ("small_0.1pi", 0.1)):
            gl, gg = [], []
            for _ in range(K):
                th = (scale * math.pi * (2 * torch.rand(2, L, n, dtype=torch.float64) - 1)).requires_grad_(True)
                z = torch.randn(1, n, dtype=torch.float64)
                pr = sim.probs(z, th)
                c_loc = (pr @ sim.zsign).mean()
                c_glob = pr[0, 0]
                g1 = torch.autograd.grad(c_loc, th, retain_graph=True)[0][0, 0, 0].item()
                g2 = torch.autograd.grad(c_glob, th)[0][0, 0, 0].item()
                gl.append(g1)
                gg.append(g2)
            row = {"n": n, "init": init_name,
                   "var_local": float(np.var(gl, ddof=1)),
                   "var_global": float(np.var(gg, ddof=1))}
            res["rows"].append(row)
            print(row, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gradvar.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
