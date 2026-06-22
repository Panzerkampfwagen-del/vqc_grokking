"""VQC training loop: full-batch AdamW, CSV logging, checkpointing on val jumps."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import make_dataset, set_seed
from src.model import VQCModel


@dataclass
class History:
    step: list[int] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    grad_var: list[float] = field(default_factory=list)


def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()


def train_vqc(
    p: int = 17,
    n_qubits: int = 6,
    n_layers: int = 3,
    seed: int = 0,
    steps: int = 30000,
    lr: float = 1e-2,
    weight_decay: float = 1.0,
    wd_on_head: bool = False,
    log_every: int = 250,
    grok_val_acc: float = 0.99,
    grok_patience: int = 200,
    results_dir: str = "results",
    device_name: str | None = None,
    verbose: bool = True,
) -> tuple[VQCModel, History]:
    set_seed(seed)
    data = make_dataset(p=p, train_frac=0.5, seed=seed)
    at, bt = data.x_train
    yt = data.y_train
    av, bv = data.x_val
    yv = data.y_val

    model = VQCModel(p=p, n_qubits=n_qubits, n_layers=n_layers,
                     device_name=device_name, seed=seed)

    # Weight decay drives grokking; by default apply it to quantum params only.
    head_wd = weight_decay if wd_on_head else 0.0
    opt = torch.optim.AdamW(
        [
            {"params": [model.q_weights], "weight_decay": weight_decay},
            {"params": model.head.parameters(), "weight_decay": head_wd},
        ],
        lr=lr,
    )

    if verbose:
        print(f"device={model.device_name} n_qubits={n_qubits} n_layers={n_layers}")
        print(f"quantum params={model.n_quantum_params()} "
              f"head params={sum(p.numel() for p in model.head.parameters())}")

    os.makedirs(results_dir, exist_ok=True)
    run_tag = f"nq{n_qubits}_L{n_layers}_seed{seed}"
    csv_path = os.path.join(results_dir, f"vqc_{run_tag}.csv")
    ckpt_dir = os.path.join(results_dir, f"ckpt_{run_tag}")
    os.makedirs(ckpt_dir, exist_ok=True)

    hist = History()
    best_val = -1.0
    above = 0
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_acc", "val_acc", "train_loss",
                         "val_loss", "grad_var"])

        for step in range(steps + 1):
            model.train()
            opt.zero_grad()
            logits = model(at, bt)
            loss = F.cross_entropy(logits, yt)
            loss.backward()
            gvar = float(model.q_weights.grad.var().item())
            opt.step()

            if step % log_every == 0:
                model.eval()
                with torch.no_grad():
                    lv = model(av, bv)
                    va = _acc(lv, yv)
                    ta = _acc(logits, yt)
                    vl = F.cross_entropy(lv, yv).item()
                hist.step.append(step)
                hist.train_acc.append(ta)
                hist.val_acc.append(va)
                hist.train_loss.append(loss.item())
                hist.val_loss.append(vl)
                hist.grad_var.append(gvar)
                writer.writerow([step, ta, va, loss.item(), vl, gvar])
                f.flush()

                if va >= best_val + 0.05:  # checkpoint on >=5% val jumps
                    best_val = va
                    torch.save(
                        {"step": step, "val_acc": va,
                         "state_dict": model.state_dict()},
                        os.path.join(ckpt_dir, f"step{step}_val{va:.2f}.pt"),
                    )
                if verbose and step % (log_every * 8) == 0:
                    el = time.time() - t0
                    print(f"  step {step:6d} train={ta:.3f} val={va:.3f} "
                          f"loss={loss.item():.3f} gvar={gvar:.2e} ({el:.0f}s)")

                above = above + 1 if va >= grok_val_acc else 0
                if above * log_every >= grok_patience:
                    print(f"early stop: val>={grok_val_acc} held at step {step}")
                    break

    torch.save({"step": hist.step[-1], "val_acc": hist.val_acc[-1],
                "state_dict": model.state_dict()},
               os.path.join(ckpt_dir, "final.pt"))
    return model, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--n_qubits", type=int, default=6)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--wd_on_head", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    _, hist = train_vqc(
        p=args.p, n_qubits=args.n_qubits, n_layers=args.n_layers, seed=args.seed,
        steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
        wd_on_head=args.wd_on_head, device_name=args.device,
    )
    grok = next((s for s, v in zip(hist.step, hist.val_acc) if v >= 0.95), None)
    print(f"final train={hist.train_acc[-1]:.3f} val={hist.val_acc[-1]:.3f} "
          f"max_val={max(hist.val_acc):.3f} grok@{grok}")


if __name__ == "__main__":
    main()
