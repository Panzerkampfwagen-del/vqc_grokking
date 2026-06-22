"""Modular arithmetic dataset: (a + b) mod p over all ordered pairs."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed random, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ModAddData:
    """Container for a single train/val split of (a + b) mod p."""

    p: int
    a: torch.Tensor          # (N,) int64  first operand
    b: torch.Tensor          # (N,) int64  second operand
    y: torch.Tensor          # (N,) int64  label = (a + b) mod p
    train_idx: torch.Tensor  # (N_train,) int64
    val_idx: torch.Tensor    # (N_val,)   int64

    @property
    def x_train(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.a[self.train_idx], self.b[self.train_idx]

    @property
    def y_train(self) -> torch.Tensor:
        return self.y[self.train_idx]

    @property
    def x_val(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.a[self.val_idx], self.b[self.val_idx]

    @property
    def y_val(self) -> torch.Tensor:
        return self.y[self.val_idx]


def make_dataset(p: int = 17, train_frac: float = 0.5, seed: int = 0) -> ModAddData:
    """Build all p^2 ordered pairs and split into train/val with a fixed seed."""
    a_grid, b_grid = torch.meshgrid(
        torch.arange(p), torch.arange(p), indexing="ij"
    )
    a = a_grid.reshape(-1).long()
    b = b_grid.reshape(-1).long()
    y = (a + b) % p

    n = a.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_train = int(round(train_frac * n))
    train_idx = perm[:n_train].sort().values
    val_idx = perm[n_train:].sort().values

    return ModAddData(p=p, a=a, b=b, y=y, train_idx=train_idx, val_idx=val_idx)
