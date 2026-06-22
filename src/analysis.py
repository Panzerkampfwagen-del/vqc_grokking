"""Analysis of a (possibly) grokked VQC: parameter crystallisation + Fourier.

If the VQC groks (val >= 0.95) run parameter + Fourier analysis. Otherwise run
the failure-mode analysis (gradient variance + effective rank of the output).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import VQCModel
from src.plotting import (plot_fourier_spectra, plot_grad_var,
                          plot_param_evolution, read_csv)


def load_model(ckpt: str, p: int, n_qubits: int, n_layers: int) -> VQCModel:
    m = VQCModel(p=p, n_qubits=n_qubits, n_layers=n_layers,
                 device_name="default.qubit", seed=0)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(state["state_dict"])
    m.eval()
    return m


def expectation_curve(model: VQCModel, p: int, b: int) -> np.ndarray:
    """E[Z_i](a) for a=0..p-1 at fixed b. Returns (p, n_qubits)."""
    a = torch.arange(p, dtype=torch.float64)
    bb = torch.full((p,), float(b), dtype=torch.float64)
    with torch.no_grad():
        z = model.expectations(a, bb)
    return z.cpu().numpy()


def fourier_analysis(model: VQCModel, p: int, b: int) -> tuple[np.ndarray, dict]:
    """DFT magnitude of each qubit's E[Z_i](a) sequence."""
    z = expectation_curve(model, p, b)              # (p, nq)
    freqs = np.arange(p)
    spectra = {}
    for q in range(z.shape[1]):
        sig = z[:, q] - z[:, q].mean()              # drop DC for clarity
        spectra[q] = np.abs(np.fft.fft(sig))
    return freqs, spectra


def dominant_frequencies(spectra: dict, p: int, top: int = 3) -> dict:
    """Top non-DC frequencies (folded to [1, p//2]) per qubit."""
    out = {}
    half = p // 2
    for q, mag in spectra.items():
        m = mag[1:half + 1].copy()                  # ignore DC, fold
        idx = np.argsort(m)[::-1][:top]
        out[q] = [(int(i + 1), float(m[i])) for i in idx]
    return out


def run_grok_analysis(seed: int, p: int, n_qubits: int, n_layers: int,
                      results_dir: str, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    run_tag = f"nq{n_qubits}_L{n_layers}_seed{seed}"
    ckpt_dir = os.path.join(results_dir, f"ckpt_{run_tag}")
    final = os.path.join(ckpt_dir, "final.pt")
    model = load_model(final, p, n_qubits, n_layers)

    # A. parameter crystallisation across saved checkpoints (init -> final)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step*.pt")),
                   key=lambda s: int(s.split("step")[1].split("_")[0]))
    stages = []
    init = VQCModel(p=p, n_qubits=n_qubits, n_layers=n_layers,
                    device_name="default.qubit", seed=seed)
    stages.append(("init", init.q_weights.detach().cpu().numpy()))
    pick = ckpts[:: max(1, len(ckpts) // 2)][:2] if ckpts else []
    for c in pick:
        st = torch.load(c, map_location="cpu", weights_only=False)
        lbl = f"step {st['step']} (val {st['val_acc']:.2f})"
        m2 = load_model(c, p, n_qubits, n_layers)
        stages.append((lbl, m2.q_weights.detach().cpu().numpy()))
    stages.append(("final", model.q_weights.detach().cpu().numpy()))
    plot_param_evolution(stages, os.path.join(figures_dir, "param_evolution.png"))

    # B + C. Fourier analysis for several fixed b
    print("\n=== Fourier analysis of E[Z_i](a) ===")
    for b in (0, 1, 2, 3):
        freqs, spectra = fourier_analysis(model, p, b)
        plot_fourier_spectra(freqs, spectra,
                             os.path.join(figures_dir, f"fourier_spectrum_b{b}.png"), b)
        dom = dominant_frequencies(spectra, p)
        print(f"\nb={b} dominant freqs per qubit (k, |DFT|):")
        for q, lst in dom.items():
            s = ", ".join(f"k={k}:{v:.2f}" for k, v in lst)
            print(f"  qubit {q}: {s}")
    print(f"\nfigures saved to {figures_dir}/")
    print("Compare: classical grokking uses cos/sin(2*pi*k*a/p) at a few k. "
          "Sharp DFT peaks at small k => Fourier-like quantum representation.")


def run_failure_analysis(seed: int, p: int, n_qubits: int, n_layers: int,
                         results_dir: str, figures_dir: str,
                         n_random: int = 500) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    run_tag = f"nq{n_qubits}_L{n_layers}_seed{seed}"
    csv_path = os.path.join(results_dir, f"vqc_{run_tag}.csv")
    if os.path.exists(csv_path):
        plot_grad_var(csv_path, os.path.join(figures_dir, f"grad_variance_{run_tag}.png"))
        d = read_csv(csv_path)
        print(f"grad var: start={d['grad_var'][0]:.2e} "
              f"min={d['grad_var'].min():.2e} end={d['grad_var'][-1]:.2e}")

    # Effective rank of the output space over random parameter sets.
    a = torch.arange(p, dtype=torch.float64)
    b = torch.zeros(p, dtype=torch.float64)
    outs = []
    for i in range(n_random):
        m = VQCModel(p=p, n_qubits=n_qubits, n_layers=n_layers,
                     device_name="default.qubit", seed=1000 + i)
        with torch.no_grad():
            outs.append(m.expectations(a, b).cpu().numpy().ravel())
    X = np.stack(outs)                              # (n_random, p*nq)
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    pr = (s.sum() ** 2) / (np.square(s).sum())      # participation ratio
    print(f"effective rank (participation ratio) of output space: {pr:.2f} "
          f"(p={p}); rank<<p => expressivity-limited")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--n_qubits", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--figures_dir", type=str, default="figures")
    ap.add_argument("--mode", choices=["auto", "grok", "failure"], default="auto")
    args = ap.parse_args()

    run_tag = f"nq{args.n_qubits}_L{args.n_layers}_seed{args.seed}"
    csv_path = os.path.join(args.results_dir, f"vqc_{run_tag}.csv")
    mode = args.mode
    if mode == "auto":
        d = read_csv(csv_path)
        mode = "grok" if d["val_acc"].max() >= 0.95 else "failure"
        print(f"auto mode -> {mode} (max val={d['val_acc'].max():.3f})")
    if mode == "grok":
        run_grok_analysis(args.seed, args.p, args.n_qubits, args.n_layers,
                          args.results_dir, args.figures_dir)
    else:
        run_failure_analysis(args.seed, args.p, args.n_qubits, args.n_layers,
                             args.results_dir, args.figures_dir)


if __name__ == "__main__":
    main()
