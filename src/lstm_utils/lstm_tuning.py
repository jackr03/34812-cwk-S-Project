import json
from pathlib import Path

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.lstm_utils.lstm_tokeniser import LSTMTokeniser
from src.lstm_utils.lstm_training import train_one_epoch
from src.models.lstm_classifier import LSTMClassifier
from src.run_transformer import validate


def run_hyperparameter_sweep(device, lstm_tokeniser: LSTMTokeniser, train_dataloader: DataLoader, val_dataloader: DataLoader, output_path: Path) -> None:
    def objective(trial) -> float:
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = LSTMClassifier(lstm_tokeniser).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        val_acc = 0.0
        for epoch in range(CONFIG.lstm.hyperparameter_tuning.epochs):
            train_one_epoch(device, model, criterion, optimiser, train_dataloader)
            _, val_acc = validate(device, model, criterion, val_dataloader)

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_acc

    print('Running hyperparameter sweep...')
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=CONFIG.lstm.hyperparameter_tuning.trials)

    print('Hyperparameter sweep completed.')
    print(f'Accuracy: {study.best_value * 100:.2f}%')
    print(f'Hyperparameters: {study.best_params}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
