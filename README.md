# NLI Binary Classification

Binary Natural Language Inference classifier using DeBERTa v3.
Given a premise and hypothesis, predicts entailment (1) or non-entailment (0).

## Setup

### Mac (MPS / CPU)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
```

### HPC (SLURM + NVIDIA CUDA)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-hpc.txt
```

## Training

```bash
python -m src.run_train
```

Training config (learning rate, epochs, batch size, etc.) is in `src/train_config.py`.
Model config (transformer model, seed) is in `src/config.py`.

## Project Structure

```
├── data/                       # CSV datasets (train, dev, trial)
├── notebooks/                  # Exploration and experimentation
├── src/
│   ├── config.py               # Model configuration
│   ├── train_config.py         # Training hyperparameters
│   ├── data/
│   │   └── dataset.py          # NLIDataset and DataLoader factory
│   ├── models/
│   │   └── TransformerClassifier.py
│   ├── train/
│   │   ├── trainer.py          # Training loop with early stopping
│   │   └── utils.py            # Device detection, seeding, metrics
│   └── run_train.py            # Entry point
├── requirements-mac.txt        # Dependencies for Mac
└── requirements-hpc.txt        # Dependencies for SLURM HPC (CUDA)
```
