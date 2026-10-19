"""Validate qsim.RingPQC against PennyLane (ideal and noisy).

Run:  python -m pytest code/tests/test_qsim.py -q     (or python test_qsim.py)
"""
import os
import sys

import numpy as np
import pennylane as qml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qsim import RingPQC  # noqa: E402


def pl_circuit(n, L, p, scale, entangle=True):
    dev = qml.device("default.mixed" if p > 0 else "default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(z, th):
        for i in range(n):
            qml.RY(z[i], wires=i)
        for l in range(L):
            for i in range(n):
                qml.RY(th[0, l, i], wires=i)
                qml.RZ(th[1, l, i], wires=i)
            if not entangle:
                continue
            for i in range(n):
                for _ in range(scale):
                    qml.CNOT(wires=[i, (i + 1) % n])
                    if p > 0:
                        qml.DepolarizingChannel(p, wires=i)
                        qml.DepolarizingChannel(p, wires=(i + 1) % n)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circ


def _rand(n, L, B, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, n, generator=g, dtype=torch.float64)
    th = (2 * torch.rand(2, L, n, generator=g, dtype=torch.float64) - 1) * np.pi
    return z, th


def test_ideal_matches_pennylane():
    for n, entangle in [(3, True), (5, True), (6, False)]:
        L = 3
        z, th = _rand(n, L, 4, seed=n)
        sim = RingPQC(n, L, entangle=entangle, dtype=torch.complex128)
        ours = sim.expvals(z, th).numpy()
        circ = pl_circuit(n, L, 0.0, 1, entangle)
        ref = np.array([torch.stack(circ(z[b], th)).numpy() for b in range(4)])
        assert np.allclose(ours, ref, atol=1e-10), (n, np.abs(ours - ref).max())


def test_ideal_gradient_matches_pennylane():
    n, L = 4, 2
    z, th = _rand(n, L, 1, seed=7)
    w = torch.linspace(-1, 1, n, dtype=torch.float64)
    th1 = th.clone().requires_grad_(True)
    sim = RingPQC(n, L, dtype=torch.complex128)
    (sim.expvals(z, th1)[0] * w).sum().backward()
    th2 = th.clone().requires_grad_(True)
    (torch.stack(pl_circuit(n, L, 0.0, 1)(z[0], th2)) * w).sum().backward()
    assert torch.allclose(th1.grad, th2.grad, atol=1e-10)


def test_folding_is_identity_without_noise():
    n, L = 4, 3
    z, th = _rand(n, L, 3, seed=3)
    sim = RingPQC(n, L, dtype=torch.complex128)
    assert torch.allclose(sim.expvals(z, th, scale=1), sim.expvals(z, th, scale=3), atol=1e-12)


def test_noisy_trajectories_match_density_matrix():
    n, L, p = 4, 2, 0.05
    T = 40000                                # trajectories
    z, th = _rand(n, L, 1, seed=11)
    sim = RingPQC(n, L, dtype=torch.complex128)
    for scale in (1, 3):
        g = torch.Generator().manual_seed(100 + scale)
        ours = sim.expvals(z.repeat(T, 1), th, p=p, scale=scale, gen=g)
        mean = ours.mean(0).numpy()
        se = ours.std(0).numpy() / np.sqrt(T)
        ref = torch.stack(pl_circuit(n, L, p, scale)(z[0], th)).numpy()
        ideal = torch.stack(pl_circuit(n, L, 0.0, 1)(z[0], th)).numpy()
        # agreement within 5 standard errors, and noise has a visible effect
        assert np.all(np.abs(mean - ref) < 5 * se + 1e-6), (scale, mean, ref, se)
        assert np.abs(ref - ideal).max() > 10 * se.max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
