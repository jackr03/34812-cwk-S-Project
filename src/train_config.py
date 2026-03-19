from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # Data
    max_length: int = 128
    batch_size: int = 32

    # Optimiser
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    # Schedule
    epochs: int = 10
    warmup_ratio: float = 0.1

    # Model
    num_labels: int = 2
    dropout: float = 0.1

    # Early stopping
    patience: int = 3

    # Paths
    checkpoint_dir: str = "checkpoints"

TRAIN_CONFIG = TrainConfig()
