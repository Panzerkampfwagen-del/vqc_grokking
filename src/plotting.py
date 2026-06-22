"""All figure generation for the VQC grokking study."""

from __future__ import annotations

import csv
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: str) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with open(path) as f:
        r = csv.DictReader(f)
        names = r.fieldnames or []
        for n in names:
            cols[n] = []
        for row in r:
            for n in names:
                cols[n].append(float(row[n]))
    return {k: np.asarray(v) for k, v in cols.items()}


def plot_curves(csv_path: str, out: str, title: str = "VQC grokking") -> None:
    """Train/val accuracy and loss vs step (log-x to expose delayed gen.)."""
    d = read_csv(csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(d["step"], d["train_acc"], label="train", lw=2)
    ax[0].plot(d["step"], d["val_acc"], label="val", lw=2)
    ax[0].axhline(1 / 17, ls="--", c="gray", lw=1, label="chance")
    ax[0].set(xlabel="step", ylabel="accuracy", title=title, xscale="log")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(d["step"], d["train_loss"], label="train")
    ax[1].plot(d["step"], d["val_loss"], label="val")
    ax[1].set(xlabel="step", ylabel="cross-entropy", title="loss", xscale="log")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_param_evolution(stages: list[tuple[str, np.ndarray]], out: str) -> None:
    """stages: list of (label, weights[layers,qubits,3]). Histogram per stage."""
    n = len(stages)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharey=True)
    if n == 1:
        ax = [ax]
    for i, (label, w) in enumerate(stages):
        flat = w.reshape(-1, 3)
        for j, name in enumerate(["phi", "theta", "omega"]):
            ax[i].scatter(np.full(flat.shape[0], j) + 0.05 * np.random.randn(flat.shape[0]),
                          flat[:, j], s=14, alpha=0.7, label=name if i == 0 else None)
        ax[i].set(title=label, xticks=[0, 1, 2],
                  xticklabels=["phi", "theta", "omega"])
        ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("angle (rad)")
    if n: ax[0].legend(fontsize=8)
    fig.suptitle("Rotation parameter crystallisation")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_fourier_spectra(freqs: np.ndarray, spectra: dict[int, np.ndarray],
                         out: str, b_val: int) -> None:
    """spectra: qubit -> magnitude array over DFT frequencies (b fixed)."""
    nq = len(spectra)
    fig, ax = plt.subplots(2, (nq + 1) // 2, figsize=(3.2 * ((nq + 1) // 2), 6))
    ax = np.atleast_1d(ax).ravel()
    for i, q in enumerate(sorted(spectra)):
        ax[i].stem(freqs, spectra[q])
        ax[i].set(title=f"qubit {q}", xlabel="frequency k", ylabel="|DFT|")
        ax[i].grid(alpha=0.3)
    for j in range(nq, len(ax)):
        ax[j].axis("off")
    fig.suptitle(f"E[Z_i](a) magnitude spectrum, b={b_val}")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_grad_var(csv_path: str, out: str) -> None:
    """Gradient variance vs step (barren-plateau diagnostic)."""
    d = read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(d["step"], d["grad_var"], lw=1.5)
    ax.set(xlabel="step", ylabel="var(quantum grad)",
           title="Gradient variance (barren-plateau check)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
