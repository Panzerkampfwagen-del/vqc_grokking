"""Classical MLP sanity check: confirms the task/split/weight-decay grok."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import make_dataset, set_seed


class MLP(nn.Module):
    """Embedding per operand (summed) + one hidden layer -> p logits.

    Summing the operand embeddings gives the additive/translation-equivariant
    inductive bias that lets the network grok; concatenation does not grok at
    p=17 with a 50% split (it memorises and stays at ~0% val).
    """

    def __init__(self, p: int, emb_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(p, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, p)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = self.emb(a) + self.emb(b)
        return self.fc2(F.relu(self.fc1(x)))


@dataclass
class History:
    step: list[int]
    train_acc: list[float]
    val_acc: list[float]
    train_loss: list[float]
    val_loss: list[float]


def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()


def train_mlp(
    p: int = 17,
    seed: int = 0,
    steps: int = 30000,
    lr: float = 1e-2,
    weight_decay: float = 1.0,
    log_every: int = 250,
    device: str = "cpu",
) -> tuple[MLP, History]:
    set_seed(seed)
    data = make_dataset(p=p, train_frac=0.5, seed=seed)
    dev = torch.device(device)
    at, bt = (t.to(dev) for t in data.x_train)
    yt = data.y_train.to(dev)
    av, bv = (t.to(dev) for t in data.x_val)
    yv = data.y_val.to(dev)

    model = MLP(p).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist = History([], [], [], [], [])
    for step in range(steps + 1):
        model.train()
        opt.zero_grad()
        logits = model(at, bt)
        loss = F.cross_entropy(logits, yt)
        loss.backward()
        opt.step()

        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                lv = model(av, bv)
                hist.step.append(step)
                hist.train_acc.append(_acc(logits, yt))
                hist.val_acc.append(_acc(lv, yv))
                hist.train_loss.append(loss.item())
                hist.val_loss.append(F.cross_entropy(lv, yv).item())
    return model, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    args = ap.parse_args()

    _, hist = train_mlp(
        p=args.p, seed=args.seed, steps=args.steps,
        lr=args.lr, weight_decay=args.weight_decay,
    )
    grok_step = next(
        (s for s, v in zip(hist.step, hist.val_acc) if v >= 0.95), None
    )
    print(f"final train_acc={hist.train_acc[-1]:.3f} val_acc={hist.val_acc[-1]:.3f}")
    print(f"grok step (val>=0.95): {grok_step}")


if __name__ == "__main__":
    main()
