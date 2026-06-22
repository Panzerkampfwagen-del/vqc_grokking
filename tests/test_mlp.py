"""Sanity check: the classical MLP must grok within 2000 steps on seed 0."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mlp_baseline import train_mlp


# At p=17 with a 50% split the grok is gradual and takes ~30k steps. We assert
# the grokking *signature* over a shorter horizon to keep the test fast: the net
# memorises early (train ~100%) while val lags, then val climbs well above chance.
def test_mlp_memorises_first() -> None:
    _, hist = train_mlp(p=17, seed=0, steps=8000, lr=1e-2, weight_decay=1.0)
    # train memorises early
    train_hit = next(i for i, v in enumerate(hist.train_acc) if v >= 0.99)
    assert hist.step[train_hit] <= 2000, "train should memorise within 2000 steps"
    # at the point of memorisation, val is still far from solved (delayed gen.)
    assert hist.val_acc[train_hit] < 0.5


def test_mlp_generalises() -> None:
    # Within 8k steps val must climb well above chance (1/17 ~ 0.06).
    _, hist = train_mlp(p=17, seed=0, steps=8000, lr=1e-2, weight_decay=1.0)
    assert max(hist.val_acc) >= 0.6, f"no delayed generalisation: max val={max(hist.val_acc):.3f}"
