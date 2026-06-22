"""Central hyperparameters for the VQC grokking experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # Task
    p: int = 17                       # modulus (prime); CLI --p overrides
    train_frac: float = 0.5           # fraction of pairs used for training

    # Circuit
    n_qubits: int = 6                 # keep <= 6 for tractable state-vector sim
    n_layers: int = 3                 # keep <= 4

    # Training
    lr: float = 1e-3
    weight_decay: float = 1.0         # key knob that triggers grokking
    wd_on_head: bool = False          # apply weight decay to classical head too
    steps: int = 5000
    log_every: int = 50
    grok_val_acc: float = 0.99        # early-stop threshold
    grok_patience: int = 200          # steps val_acc must hold above threshold

    # Reproducibility / bookkeeping
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    results_dir: str = "results"
    figures_dir: str = "figures"


DEFAULT = Config()
