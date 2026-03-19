import json
from pathlib import Path

import optuna
from matplotlib import pyplot as plt

from src.config import CONFIG


def run_sweep(objective, output_path: Path) -> None:
    print('Running hyperparameter sweep...')
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=CONFIG.hyperparameter_tuning.trials)

    print('Hyperparameter sweep completed.')
    print(f'Accuracy: {study.best_value * 100:.2f}%')
    print(f'Parameters: {study.best_params}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)

def plot_training_history(train_losses: list[float], train_accs: list[float], val_losses: list[float], val_accs: list[float], save_dir: Path) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharex=True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax1.plot(epochs, train_losses, label='Train')
    ax1.plot(epochs, val_losses, label='Validation')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, label='Train')
    ax2.plot(epochs, val_accs, label='Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved to {save_dir / "training_history.png"}')
