"""VQCModel: PennyLane circuit (quantum params) + classical linear read-out head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.circuit import make_circuit, weight_shape


class VQCModel(nn.Module):
    """6-qubit VQC whose PauliZ expectations feed a trainable linear -> p logits."""

    def __init__(
        self,
        p: int,
        n_qubits: int = 6,
        n_layers: int = 3,
        device_name: str | None = None,
        init_scale: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.p = p
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit, self.dev, self.device_name = make_circuit(
            n_qubits, n_layers, p, device_name=device_name
        )

        shp = weight_shape(n_qubits, n_layers)
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        w0 = init_scale * torch.randn(*shp, dtype=torch.float64, generator=gen)
        self.q_weights = nn.Parameter(w0)              # trainable quantum params
        self.head = nn.Linear(n_qubits, p).double()    # classical read-out

    def expectations(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return (batch, n_qubits) PauliZ expectations from the circuit."""
        a = a.double()
        b = b.double()
        out = self.circuit(a, b, self.q_weights)
        return torch.stack(out, dim=-1).to(self.head.weight.dtype)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.head(self.expectations(a, b))

    def n_quantum_params(self) -> int:
        return self.q_weights.numel()
