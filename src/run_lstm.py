import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.lstm_utils.lstm_dataset import LSTMDataset
from src.lstm_utils.lstm_tokeniser import LSTMTokeniser
from src.lstm_utils.lstm_training import validate, evaluate, train_one_epoch
from src.lstm_utils.lstm_tuning import run_hyperparameter_sweep
from src.models.lstm_classifier import LSTMClassifier
from src.utils import plot_training_history

TRAIN_DATA_PATH = Path('data/train.csv')
VAL_DATA_PATH = Path('data/dev.csv')
TEST_DATA_PATH = Path('data/NLI_trial.csv')

HYPERPARAMETERS_PATH = Path('hyperparameters/lstm.json')
MODEL_PATH = Path('models/lstm.pt')


def get_device_info(device: torch.device) -> dict:
    info = {'type': str(device)}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        info['name'] = torch.cuda.get_device_name(device)
        info['vram_gb'] = round(props.total_memory / (1024 ** 3), 2)
        info['compute_capability'] = f'{props.major}.{props.minor}'
        info['multi_processor_count'] = props.multi_processor_count
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True, help='Directory for this run\'s outputs')
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f'Using device: {device}')

    generator = torch.manual_seed(CONFIG.seed)
    lstm_tokeniser = LSTMTokeniser()

    train_dataset = LSTMDataset(TRAIN_DATA_PATH, lstm_tokeniser)
    val_dataset = LSTMDataset(VAL_DATA_PATH, lstm_tokeniser)
    test_dataset = LSTMDataset(TEST_DATA_PATH, lstm_tokeniser)

    train_dataloader = DataLoader(train_dataset, generator=generator, batch_size=CONFIG.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, generator=generator, batch_size=CONFIG.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, generator=generator, batch_size=CONFIG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if CONFIG.lstm.hyperparameter_tuning.should_run:
        run_hyperparameter_sweep(device, lstm_tokeniser, train_dataloader, val_dataloader, HYPERPARAMETERS_PATH)

    with open(HYPERPARAMETERS_PATH, 'r') as f:
        hyperparameters = json.load(f)
    print(f'Hyperparameters used: {hyperparameters}')
    print()

    model = LSTMClassifier(lstm_tokeniser, dropout=hyperparameters['dropout']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    device_info = get_device_info(device)
    cfg = CONFIG.lstm

    config_summary = {
        'model': 'LSTM',
        'max_seq_length': cfg.max_length,
        'embedding_dim': cfg.embedding_dim,
        'hidden_dim': cfg.hidden_dim,
        'epochs': CONFIG.epochs,
        'patience': CONFIG.patience,
        'batch_size': CONFIG.batch_size,
        'learning_rate': hyperparameters["lr"],
        'weight_decay': hyperparameters["weight_decay"],
        'optimizer': 'AdamW',
        'loss_function': 'CrossEntropyLoss',
        'seed': CONFIG.seed,
        'device': device_info,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'train_batches': len(train_dataloader),
        'val_batches': len(val_dataloader),
    }

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)
    print(f'Config saved to {run_dir / "config.json"}')

    print('=' * 60)
    print('Training Configuration')
    print('=' * 60)
    print(f'  Model:             LSTM')
    print(f'  Max seq length:    {cfg.max_length}')
    print(f'  Embedding dim:     {cfg.embedding_dim}')
    print(f'  Hidden dim:        {cfg.hidden_dim}')
    print(f'  Epochs:            {CONFIG.epochs}')
    print(f'  Patience:          {CONFIG.patience}')
    print(f'  Batch size:        {CONFIG.batch_size}')
    print(f'  Learning rate:     {hyperparameters["lr"]:.6f}')
    print(f'  Weight decay:      {hyperparameters["weight_decay"]:.6f}')
    print(f'  Optimizer:         AdamW')
    print(f'  Loss function:     CrossEntropyLoss')
    print(f'  Seed:              {CONFIG.seed}')
    print(f'  Device:            {device_info["type"]}')
    if device.type == 'cuda':
        print(f'  GPU name:          {device_info["name"]}')
        print(f'  GPU VRAM:          {device_info["vram_gb"]} GB')
        print(f'  Compute cap:       {device_info["compute_capability"]}')
        print(f'  SMs:               {device_info["multi_processor_count"]}')
    print(f'  Train samples:     {len(train_dataset):,}')
    print(f'  Val samples:       {len(val_dataset):,}')
    print(f'  Train batches:     {len(train_dataloader):,}')
    print(f'  Val batches:       {len(val_dataloader):,}')
    print('=' * 60)
    print()

    print('Training LSTM model...')
    best_acc = 0.0
    patience = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    total_start = time.time()
    for epoch in range(CONFIG.epochs):
        epoch_start = time.time()
        print(f'[Epoch {epoch + 1}/{CONFIG.epochs}]')

        train_loss, train_acc = train_one_epoch(device, model, criterion, optimizer, train_dataloader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = validate(device, model, criterion, val_dataloader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            # Avoid saving GloVe embedding
            state = {k: v for k, v in model.state_dict().items() if 'embedding' not in k}
            torch.save(model.state_dict(), MODEL_PATH)
            patience = 0
        else:
            patience += 1

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        print(f'Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc:.2f}%')
        print(f'Epoch time: {epoch_time:.1f}s | Total elapsed: {elapsed:.1f}s')
        print()

        if patience == CONFIG.patience:
            print(f'Finishing training early, no improvement in {CONFIG.patience} epochs')
            break
        
    total_time = time.time() - total_start
    print(f'Training complete in {total_time:.1f}s')
    print(f'Best model had an accuracy of {best_acc:.2f}%.')

    print('Running final evaluation on benchmark...')
    model.load_state_dict(torch.load(MODEL_PATH))
    test_results = evaluate(device, model, test_dataloader)
    print('[Benchmark Results]')
    print(f'Accuracy:           {test_results["accuracy"]:.2f}%')
    print(f'Macro Precision:    {test_results["macro_precision"]:.4f}')
    print(f'Macro Recall:       {test_results["macro_recall"]:.4f}')
    print(f'Macro F1:           {test_results["macro_f1"]:.4f}')
    print(f'Weighted Precision: {test_results["weighted_precision"]:.4f}')
    print(f'Weighted Recall:    {test_results["weighted_recall"]:.4f}')
    print(f'Weighted F1:        {test_results["weighted_f1"]:.4f}')

    results = {
        'training_time_seconds': round(total_time, 1),
        'best_val_accuracy': best_acc,
        'final_train_loss': train_losses[-1],
        'final_train_accuracy': train_accs[-1],
        'final_val_loss': val_losses[-1],
        'final_val_accuracy': val_accs[-1],
        'benchmark_results': test_results,
        'epoch_history': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
        },
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {run_dir / "results.json"}')

    plots_dir = run_dir / 'plots'
    plot_training_history(train_losses, train_accs, val_losses, val_accs, save_dir=plots_dir)


if __name__ == '__main__':
    main()
