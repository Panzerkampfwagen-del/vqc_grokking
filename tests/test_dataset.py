"""Tests for the modular arithmetic dataset and splits."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import make_dataset


def test_total_size() -> None:
    d = make_dataset(p=17, train_frac=0.5, seed=0)
    assert d.a.shape[0] == 17 * 17 == 289


def test_split_sizes() -> None:
    d = make_dataset(p=17, train_frac=0.5, seed=0)
    n_train = d.train_idx.shape[0]
    n_val = d.val_idx.shape[0]
    assert n_train + n_val == 289
    assert n_train == 144 and n_val == 145  # round(0.5*289)=144


def test_no_leakage() -> None:
    d = make_dataset(p=17, train_frac=0.5, seed=0)
    inter = set(d.train_idx.tolist()) & set(d.val_idx.tolist())
    assert len(inter) == 0


def test_labels_correct() -> None:
    d = make_dataset(p=17, train_frac=0.5, seed=0)
    assert torch.equal(d.y, (d.a + d.b) % 17)
    assert int(d.y.max()) == 16 and int(d.y.min()) == 0


def test_split_deterministic() -> None:
    d1 = make_dataset(p=17, seed=0)
    d2 = make_dataset(p=17, seed=0)
    assert torch.equal(d1.train_idx, d2.train_idx)
    d3 = make_dataset(p=17, seed=1)
    assert not torch.equal(d1.train_idx, d3.train_idx)
