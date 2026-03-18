from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 100
    transformer_model: str = 'microsoft/deberta-v3-base'

CONFIG = Config()