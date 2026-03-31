from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransformerHyperparameterTuningConfig:
    epochs: int = 3
    trials: int = 20
    should_run: bool = False

@dataclass(frozen=True)
class SLMHyperparameterTuningConfig:
    epochs: int = 1
    trials: int = 8
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
class SLMConfig:
    model: str = 'Qwen/Qwen3.5-9B-Base'
    max_length: int = 256
    hyperparameter_tuning: SLMHyperparameterTuningConfig = SLMHyperparameterTuningConfig()

@dataclass(frozen=True)
class SLMFinetuneConfig:
    epochs: int = 1
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = 'cosine'
    logging_steps: int = 10
    eval_steps: int = 38  # ~every 10% of a single epoch (382 steps total)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple = (
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    )
    bnb_quant_type: str = 'nf4'
    bnb_double_quant: bool = True


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
    slm: SLMConfig = SLMConfig()
    slm_finetune: SLMFinetuneConfig = SLMFinetuneConfig()

CONFIG = Config()