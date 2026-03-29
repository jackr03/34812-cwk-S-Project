from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransformerHyperparameterTuningConfig:
    epochs: int = 3
    trials: int = 20
    should_run: bool = False


@dataclass(frozen=True)
class BiLSTMHyperparameterTuningConfig:
    epochs: int = 3
    trials: int = 20
    should_run: bool = True


@dataclass(frozen=True)
class TransformerConfig:
    model: str = 'distilbert-base-uncased'
    max_length: int = 64
    hyperparameter_tuning: TransformerHyperparameterTuningConfig = TransformerHyperparameterTuningConfig()


@dataclass(frozen=True)
class BiLSTMConfig:
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    max_length: int = 64
    hyperparameter_tuning: BiLSTMHyperparameterTuningConfig = BiLSTMHyperparameterTuningConfig()


@dataclass(frozen=True)
class Config:
    project_root = Path(__file__).resolve().parent.parent
    seed: int = 100
    batch_size: int = 256
    epochs: int = 20
    transformer: TransformerConfig = TransformerConfig()
    bilstm: BiLSTMConfig = BiLSTMConfig()

CONFIG = Config()