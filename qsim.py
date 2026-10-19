"""
qsim.py — Batched state-vector simulator for the QGenNet PQC
============================================================
A small PyTorch simulator specialised to the QGenNet circuit:

    R_y(z_i) embedding  ->  L x [R_y(theta), R_z(phi) on every qubit, ring-CNOT]
    ->  <Z_i> on every qubit

Why not PennyLane directly?  Exact density-matrix simulation
(``default.mixed``) of 12 qubits costs ~1 min per sample on a laptop CPU,
which makes GAN training infeasible.  Instead, depolarising noise is
simulated with stochastic Pauli trajectories (Monte Carlo wave-function
method): after each noisy gate, each affected qubit independently receives
X, Y or Z with probability p/3 each.  Averaged over trajectories this
reproduces the depolarising channel exactly, so each per-sample expectation
value is an unbiased estimator of the noisy (density-matrix) expectation
value.  ``tests/test_qsim.py`` checks the simulator against PennyLane
``default.qubit`` (ideal) and ``default.mixed`` (noisy).

Conventions follow PennyLane: wire 0 is the most significant bit,
DepolarizingChannel(p) = (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z).
Noise is applied to both qubits after every CNOT; single-qubit gates are
noiseless.

Implementation notes (all exact, no approximation):
  * the R_y(z) embedding of |0..0> is a product state and is built directly;
  * every ring-CNOT layer is a fixed basis permutation;
  * Pauli errors inside a CNOT layer are propagated through the remaining
    CNOTs (Clifford conjugation) and applied once at the end of the layer
    as X^a Z^b (global phases dropped; they do not affect expectations).

Gradients come from PyTorch autograd.  For gates exp(-i theta P / 2) this
equals the parameter-shift gradient used on hardware.

Noise amplification for ZNE uses local unitary folding of the noisy
two-qubit gates, CNOT -> CNOT (CNOT^dagger CNOT)^k, i.e. noise scale factor
2k+1 with the ideal circuit unchanged.
"""

from __future__ import annotations

import torch


class RingPQC:
    """Batched simulator for the QGenNet ansatz on ``n`` qubits."""

    def __init__(self, n_qubits: int, n_layers: int, entangle: bool = True,
                 dtype: torch.dtype = torch.complex64):
        self.n = n_qubits
        self.L = n_layers
        self.entangle = entangle
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64
        n = n_qubits
        N = 2 ** n
        self.idx = torch.arange(N)
        self.wmask = [1 << (n - 1 - w) for w in range(n)]              # bit of wire w
        bits = torch.stack([(self.idx >> (n - 1 - w)) & 1 for w in range(n)])  # (n, N)
        self.bits_f = bits.T.to(torch.float32).contiguous()          # (N, n)
        self.zsign = (1 - 2 * bits).T.to(self.rdtype).contiguous()  # (N, n)
        # ring-CNOT layer CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0) as one gather index:
        # new[j] = old[perm[j]]  with perm = composition of the individual CNOTs
        perm = self.idx.clone()
        for c in range(n):
            t = (c + 1) % n
            pc = torch.where(((self.idx >> (n - 1 - c)) & 1) == 1,
                             self.idx ^ self.wmask[t], self.idx)
            perm = perm[pc]
        self.layer_perm = perm

    # ------------------------------------------------------------------ helpers
    def _product_state(self, z: torch.Tensor) -> torch.Tensor:
        """prod_i R_y(z_i)|0>  as a (B, 2^n) real-amplitude state."""
        c, s = torch.cos(z / 2), torch.sin(z / 2)          # (B, n)
        psi = torch.stack([c[:, 0], s[:, 0]], dim=1)       # (B, 2)
        for w in range(1, self.n):
            q = torch.stack([c[:, w], s[:, w]], dim=1)     # (B, 2)
            psi = (psi.unsqueeze(2) * q.unsqueeze(1)).reshape(psi.shape[0], -1)
        return psi.to(self.cdtype)

    def _single_qubit_layer(self, psi: torch.Tensor, ty: torch.Tensor,
                            tz: torch.Tensor) -> torch.Tensor:
        """Apply R_z(tz_w) R_y(ty_w) on every wire w (shared across the batch)."""
        B = psi.shape[0]
        c = torch.cos(ty / 2).to(self.cdtype)
        s = torch.sin(ty / 2).to(self.cdtype)
        e = torch.exp(-0.5j * tz.to(self.cdtype))           # e^{-i phi/2}
        # U = RZ(phi) RY(theta) = [[e c, -e s], [e* s, e* c]]
        U = torch.stack([torch.stack([e * c, -e * s], -1),
                         torch.stack([torch.conj(e) * s, torch.conj(e) * c], -1)], -2)  # (n,2,2)
        for w in range(self.n):
            v = psi.view(B * 2 ** w, 2, -1)
            psi = torch.matmul(U[w], v).reshape(B, -1)
        return psi

    def _noisy_cnot_layer(self, psi, p: float, scale: int, gen):
        B = psi.shape[0]
        n = self.n
        psi = psi[:, self.layer_perm]
        if p <= 0:
            return psi
        # one uniform per (noisy CNOT application, qubit); X for u < 2p/3
        # (X or Y), Z for p/3 <= u < p (Y or Z)
        u = torch.rand(n * scale, 2, B, generator=gen)
        ex_all = (u < 2 * p / 3)
        ez_all = (u >= p / 3) & (u < p)
        if not bool((u < p).any()):
            return psi
        fx = [torch.zeros(B, dtype=torch.bool) for _ in range(n)]  # Pauli frame
        fz = [torch.zeros(B, dtype=torch.bool) for _ in range(n)]
        k = 0
        for c in range(n):
            t = (c + 1) % n
            for _ in range(scale):
                # conjugate the accumulated frame through CNOT(c, t):
                # X_c -> X_c X_t,  Z_t -> Z_c Z_t
                fx[t] = fx[t] ^ fx[c]
                fz[c] = fz[c] ^ fz[t]
                fx[c] = fx[c] ^ ex_all[k, 0]
                fz[c] = fz[c] ^ ez_all[k, 0]
                fx[t] = fx[t] ^ ex_all[k, 1]
                fz[t] = fz[t] ^ ez_all[k, 1]
                k += 1
        fx = torch.stack(fx, 1)
        fz = torch.stack(fz, 1)
        if not bool(fx.any() or fz.any()):
            return psi
        # Z^b: sign (-1)^{popcount(b & j)}
        if bool(fz.any()):
            par = (fz.to(torch.float32) @ self.bits_f.T).remainder(2)   # (B, N)
            psi = psi * (1 - 2 * par).to(self.cdtype)
        # X^a: |j> -> |j xor a|, i.e. new[j] = old[j xor a]
        if bool(fx.any()):
            a = (fx.long() * torch.tensor(self.wmask)).sum(1, keepdim=True)  # (B,1)
            psi = torch.gather(psi, 1, self.idx.view(1, -1) ^ a)
        return psi

    # ---------------------------------------------------------------- circuit
    def expvals(self, z: torch.Tensor, theta: torch.Tensor, p: float = 0.0,
                scale: int = 1, gen: torch.Generator | None = None) -> torch.Tensor:
        """Return <Z_i> for every qubit, shape (B, n).

        Args:
            z:     (B, n) latent angles for the R_y embedding.
            theta: (2, L, n) trainable angles; theta[0] -> R_y, theta[1] -> R_z.
            p:     depolarising probability applied to both qubits after
                   every CNOT (0 = ideal circuit).
            scale: odd noise-scale factor (1, 3, 5, ...) via CNOT folding.
            gen:   torch.Generator for the noise trajectories.
        """
        prob = self.probs(z, theta, p, scale, gen)                  # (B, N)
        return prob @ self.zsign                                    # (B, n)

    def probs(self, z, theta, p: float = 0.0, scale: int = 1, gen=None) -> torch.Tensor:
        """Computational-basis probabilities |<j|psi>|^2, shape (B, 2^n)."""
        assert scale % 2 == 1 and scale >= 1
        B, n = z.shape
        assert n == self.n
        psi = self._product_state(z.to(self.rdtype))
        th = theta.to(self.rdtype)
        for l in range(self.L):
            psi = self._single_qubit_layer(psi, th[0, l], th[1, l])
            if self.entangle:
                psi = self._noisy_cnot_layer(psi, p, scale, gen)
        return psi.real ** 2 + psi.imag ** 2
