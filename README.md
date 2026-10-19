# In-loop zero-noise extrapolation for hybrid quantum-classical GANs

Code, configurations and per-seed results for the manuscript
*"Zero-Noise Extrapolation Inside the Training Loop of Hybrid Quantum-Classical
Generative Adversarial Networks: A Multi-Domain Noisy-Simulation Study"*
(R. K. Gupta, V. P. Vasani, S. Pawar, S. S. H. Hajjaj, S. V. Easwaramoorthy;
manuscript in preparation).

**QGenNet** passes a latent vector through a 12-qubit, 6-layer ring-entangled parameterised
circuit and applies Richardson zero-noise extrapolation (ZNE) to the measured ⟨Z⟩ values
inside the WGAN-GP training loop, folding every CNOT for noise scales 1 and 3. The ⟨Z⟩
values then go through a normalisation layer and a small MLP decoder. Every result is a
**noisy simulation**; no quantum hardware was used.

## Main findings (5 seeds per main configuration)

- In-loop ZNE reduces the exact bias of the circuit outputs 10–17× at a depolarising
  probability of 10⁻³, but it does **not** significantly improve sample quality on any dataset
  or noise level tested (p = 10⁻³ … 2×10⁻²).
- Generators trained with ZNE degrade when evaluated on the noise-free circuit.
- QGenNet does not outperform classical generators. It ties them on tabular data, is the
  weakest model on equity returns, and is far behind a convolutional generator on MNIST.
  Ablations show no benefit from the entangling gates.

The per-seed numbers behind these statements are in `runs/` and `results/`. All tables and
figures are produced from them by `make_tables.py` and `make_figures.py`.

## Setup

```bash
python -m venv --system-site-packages .venv   # or a plain venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
```

Tested with Python 3.12.10, PyTorch 2.6.0 (CPU), PennyLane 0.43.1 with jax 0.6.2, scikit-learn
1.7.2, ctgan 0.12.1 and yfinance 1.7.0 on an Intel i7-13700HX (16 GB RAM).
PennyLane 0.43 fails to import with newer jax versions, so jax is pinned to 0.6.2.

## Reproducing everything

| Step | Command | Output |
|---|---|---|
| 1. Simulator tests | `python tests/test_qsim.py` | 4 × PASS |
| 2. Data | `python data.py --prepare` | `data/` |
| 3. LeNet feature extractor | `python metrics.py` | `data/lenet_mnist.pt` |
| 4. All 118 training runs | `python run_all.py --workers 4` | `runs/<domain>/<run>/seed<k>/` |
| 5. Diagnostics | `python exp_gradvar.py`, `python exp_concentration.py`, `python exp_zne_bias.py --n 6` (also `--n 8`, `--n 10`), `python exp_reference.py` | `results/*.json` |
| 6. Tables | `python make_tables.py` | `outputs/tables/*.tex`, `results/summary.json` |
| 7. Figures | `python make_figures.py` | `outputs/figures/*.pdf` |

Steps 6–7 work directly on the files tracked in this repository, with no retraining.

`run_all.py` is resumable: it skips runs whose `metrics.json` already exists. On our machine
the full matrix took roughly 16 hours of running time with 4 parallel workers (CPU only). Pass
`--no_new_after "YYYY-mm-dd HH:MM"` and `--hard_stop "YYYY-mm-dd HH:MM"` to pause cleanly.

## Data

| Dataset | Source | In this repository? |
|---|---|---|
| MNIST | `torchvision` download | No (downloaded by `data.py`) |
| Breast Cancer Wisconsin (Diagnostic) | UCI ML Repository copy bundled with scikit-learn | No (loaded from scikit-learn) |
| 50 US equities, 2014-01-02 to 2024-12-31 | Yahoo Finance via `yfinance` | **No.** Prices are not redistributed; `data.py --prepare` downloads them. Tickers and download metadata are in `data/stocks_meta.json` |

Yahoo Finance adjusted closing prices are revised when later dividends or splits occur, so a
fresh download can differ slightly from the one used here (downloaded 2026-10-18).

## Files

| File | Purpose |
|---|---|
| `qsim.py` | Batched state-vector simulator: Pauli-trajectory depolarising noise, CNOT folding |
| `nets.py` | QGenNet (circuit + in-loop ZNE + normalisation + decoder), classical generators, critics |
| `train_gan.py` | WGAN-GP training and evaluation for all generator types |
| `run_ctgan.py`, `timegan.py` | CTGAN (official package) and TimeGAN (PyTorch re-implementation) baselines |
| `metrics.py` | Fréchet distance on LeNet features, NDB, TVD, correlation difference, ML efficacy, TimeGAN discriminative/predictive scores |
| `data.py` | Dataset preparation |
| `exp_*.py` | Diagnostics: gradient variance, output concentration, exact ZNE bias, real-data reference values |
| `run_all.py` | Full experiment matrix (resumable) |
| `make_tables.py`, `make_figures.py` | All tables and figures from logged results |
| `audit_numbers.py` | Checks every number in a manuscript `.tex` against the generated results |

## Note on the Fréchet distance

`fd_lenet` is computed on the features of a LeNet-style MNIST classifier (99.07 % test
accuracy). It is **not** the InceptionV3 FID and cannot be compared with FID values reported
elsewhere.

## License

**All rights reserved.** This code is published for inspection only, so that the results in
the manuscript can be checked. It may not be used, copied, modified or redistributed without
written permission from the authors. See [LICENSE](LICENSE).
