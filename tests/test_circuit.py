"""Tests for the VQC: output shape, gradient flow, measurement range."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.circuit import make_circuit, weight_shape


def _build(n_qubits: int = 6, n_layers: int = 3, p: int = 17):
    circuit, _, name = make_circuit(n_qubits, n_layers, p, device_name="default.qubit")
    w = 0.1 * torch.randn(*weight_shape(n_qubits, n_layers), dtype=torch.float64)
    return circuit, w, name


def test_output_shape_batched() -> None:
    circuit, w, _ = _build()
    a = torch.arange(8, dtype=torch.float64)
    b = torch.zeros(8, dtype=torch.float64)
    out = torch.stack(circuit(a, b, w), dim=-1)  # (batch, n_qubits)
    assert out.shape == (8, 6)


def test_measurement_range() -> None:
    circuit, w, _ = _build()
    a = torch.arange(17, dtype=torch.float64)
    b = torch.arange(17, dtype=torch.float64)
    out = torch.stack(circuit(a, b, w), dim=-1)
    assert torch.all(out <= 1.0 + 1e-6) and torch.all(out >= -1.0 - 1e-6)


def test_gradients_flow() -> None:
    circuit, w, _ = _build()
    w = w.clone().requires_grad_(True)
    a = torch.arange(4, dtype=torch.float64)
    b = torch.ones(4, dtype=torch.float64)
    out = torch.stack(circuit(a, b, w), dim=-1)
    loss = out.sum()
    loss.backward()
    assert w.grad is not None
    assert torch.any(w.grad.abs() > 0)  # non-trivial gradient signal
