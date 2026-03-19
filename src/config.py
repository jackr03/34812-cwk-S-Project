from dataclasses import dataclass


@dataclass(frozen=True)
class HyperparameterTuningConfig:
    epochs: int = 3
    trials: int = 20
    should_run: bool = False


@dataclass(frozen=True)
class TransformerConfig:
    model: str = 'distilbert-base-uncased'
    max_length: int = 64


@dataclass(frozen=True)
class Config:
    seed: int = 100
    batch_size: int = 64
    epochs: int = 20
    hyperparameter_tuning: HyperparameterTuningConfig = HyperparameterTuningConfig()
    transformer: TransformerConfig = TransformerConfig()

CONFIG = Config()