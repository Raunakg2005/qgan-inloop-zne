"""
nets.py — Generators and critics
================================

Generators
    QGenNet      PQC (qsim.RingPQC) -> <Z_i> -> MLP decoder (n -> 256 -> d).
                 Optional Richardson-order-2 ZNE on the expectation values:
                     <Z>_ZNE = 3/2 <Z>_{scale 1} - 1/2 <Z>_{scale 3}
    DecoderOnly  Ablation: the PQC is removed and the latent z (dim n) is fed
                 straight into the identical MLP decoder.
    MLPGen       Standard WGAN-GP MLP generator (64 -> 256 -> 256 -> d).
    DCGANGen     DCGAN-style transposed-conv generator for 28x28 images.

Critics (no batch-norm, as recommended for WGAN-GP)
    ConvCritic   for 28x28 images
    MLPCritic    for flat tabular / time-series windows
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from qsim import RingPQC


class QGenNet(nn.Module):
    def __init__(self, out_dim: int, n_qubits: int = 12, n_layers: int = 6,
                 hidden: int = 256, p: float = 0.001, zne: bool = True,
                 entangle: bool = True, tanh_out: bool = False,
                 init_scale: float = 0.1, noise_seed: int = 0,
                 feat_norm: bool = True):
        super().__init__()
        self.n, self.L = n_qubits, n_layers
        self.p, self.zne = p, zne
        self.latent_dim = n_qubits
        self.sim = RingPQC(n_qubits, n_layers, entangle=entangle)
        # small-angle initialisation (near-identity layers)
        self.theta = nn.Parameter(init_scale * math.pi * (2 * torch.rand(2, n_layers, n_qubits) - 1))
        # The entangled circuit's <Z_i> vary only weakly with z (std ~0.07),
        # so they are standardised (batch statistics, learnable affine)
        # before the decoder.
        self.feat_norm = nn.BatchNorm1d(n_qubits) if feat_norm else nn.Identity()
        self.decoder = nn.Sequential(nn.Linear(n_qubits, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))
        self.tanh_out = tanh_out
        self.noise_gen = torch.Generator().manual_seed(noise_seed)

    def quantum_features(self, z: torch.Tensor) -> torch.Tensor:
        q1 = self.sim.expvals(z, self.theta, p=self.p, scale=1, gen=self.noise_gen)
        if not self.zne or self.p == 0:
            return q1
        q3 = self.sim.expvals(z, self.theta, p=self.p, scale=3, gen=self.noise_gen)
        return 1.5 * q1 - 0.5 * q3

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(self.feat_norm(self.quantum_features(z)))
        return torch.tanh(x) if self.tanh_out else x

    def param_groups(self, lr_q: float, lr_c: float):
        return [{"params": [self.theta], "lr": lr_q},
                {"params": list(self.feat_norm.parameters()) + list(self.decoder.parameters()),
                 "lr": lr_c}]


class DecoderOnly(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int = 12, hidden: int = 256,
                 tanh_out: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))
        self.tanh_out = tanh_out

    def forward(self, z):
        x = self.decoder(z)
        return torch.tanh(x) if self.tanh_out else x

    def param_groups(self, lr_q, lr_c):
        return [{"params": self.parameters(), "lr": lr_c}]


class MLPGen(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int = 64, hidden: int = 256,
                 tanh_out: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))
        self.tanh_out = tanh_out

    def forward(self, z):
        x = self.net(z)
        return torch.tanh(x) if self.tanh_out else x

    def param_groups(self, lr_q, lr_c):
        return [{"params": self.parameters(), "lr": lr_c}]


class DCGANGen(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * 7 * 7),
                                nn.BatchNorm1d(128 * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7)).view(z.size(0), -1)

    def param_groups(self, lr_q, lr_c):
        return [{"params": self.parameters(), "lr": lr_c}]


class ConvCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))

    def forward(self, x):
        return self.net(x.view(-1, 1, 28, 28)).view(-1)


class MLPCritic(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def gradient_penalty(critic, real, fake, lam: float = 10.0):
    eps = torch.rand(real.size(0), 1)
    xh = (eps * real + (1 - eps) * fake).requires_grad_(True)
    g = torch.autograd.grad(critic(xh).sum(), xh, create_graph=True)[0]
    return lam * ((g.norm(2, dim=1) - 1) ** 2).mean()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
