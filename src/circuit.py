"""PennyLane VQC: angle encoding (data re-uploading) + trainable Rot/CNOT layers."""

from __future__ import annotations

import math
from typing import Callable

import pennylane as qml
import torch


def select_device(n_qubits: int) -> tuple[qml.device, str]:
    """Pick a backend. default.qubit (torch backprop) is far faster than
    lightning's adjoint diff for our batched full-batch training, so it is
    preferred over lightning.qubit; lightning.gpu is tried first if present."""
    for name in ("lightning.gpu", "default.qubit", "lightning.qubit"):
        try:
            dev = qml.device(name, wires=n_qubits)
            return dev, name
        except Exception:
            continue
    raise RuntimeError("no PennyLane device available")


def _encode(a: torch.Tensor, b: torch.Tensor, p: int, n_qubits: int) -> None:
    """Encode a, b as rotation angles, re-uploading across qubit pairs."""
    theta_a = 2.0 * math.pi * a / p
    theta_b = 2.0 * math.pi * b / p
    for w in range(0, n_qubits, 2):
        qml.RY(theta_a, wires=w)
        if w + 1 < n_qubits:
            qml.RY(theta_b, wires=w + 1)


def _layer(weights: torch.Tensor, n_qubits: int) -> None:
    """One trainable layer: Rot on each qubit + CNOT ring entangler."""
    for w in range(n_qubits):
        qml.Rot(weights[w, 0], weights[w, 1], weights[w, 2], wires=w)
    if n_qubits > 1:
        for w in range(n_qubits):
            qml.CNOT(wires=[w, (w + 1) % n_qubits])


def make_circuit(
    n_qubits: int,
    n_layers: int,
    p: int,
    device_name: str | None = None,
    diff_method: str = "best",
) -> tuple[Callable, qml.device, str]:
    """Build a TorchLayer-compatible QNode returning n PauliZ expectations."""
    if device_name is None:
        dev, device_name = select_device(n_qubits)
    else:
        dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
        _encode(a, b, p, n_qubits)
        for layer in range(n_layers):
            _layer(weights[layer], n_qubits)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit, dev, device_name


def weight_shape(n_qubits: int, n_layers: int) -> tuple[int, int, int]:
    """Shape of the trainable rotation weights: (layers, qubits, 3)."""
    return (n_layers, n_qubits, 3)
