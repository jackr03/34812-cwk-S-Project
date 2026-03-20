print('Starting training script...')
import argparse
import json
import time
from pathlib import Path

import optuna
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import CONFIG
from src.models.nli_dataset import NLIDataset
from src.models.transformer_classifier import TransformerClassifier
from src.utils import run_sweep, plot_training_history

print(f"Imports complete")

TRAIN_DATA_PATH = Path('data/train.csv')
VAL_DATA_PATH = Path('data/dev.csv')
TEST_DATA_PATH = Path('data/NLI_trial.csv')
HYPERPARAMETERS_PATH = Path('hyperparameters/transformer.json')
MODEL_PATH = Path('models/transformer.pt')


def get_device_info(device: torch.device) -> dict:
    info = {'type': str(device)}
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        info['name'] = torch.cuda.get_device_name(device)
        info['vram_gb'] = round(props.total_memory / (1024 ** 3), 2)
        info['compute_capability'] = f'{props.major}.{props.minor}'
        info['multi_processor_count'] = props.multi_processor_count
    return info


def train_one_epoch(device, model, criterion, optimiser, train_dataloader) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(train_dataloader, desc='Training', unit='batches'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimiser.zero_grad(set_to_none=True)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(device, model, criterion, val_dataloader) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in tqdm(val_dataloader, desc='Validating', unit='batches'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_test(device, model, tokeniser, test_path: Path) -> dict:
    test_pd = pd.read_csv(test_path)
    test_dataset = NLIDataset(test_pd, tokeniser, CONFIG.transformer.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc='Evaluating NLI_trial', unit='batches'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'num_samples': len(all_labels),
    }


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

    train_pd = pd.read_csv(TRAIN_DATA_PATH)
    val_pd = pd.read_csv(VAL_DATA_PATH)

    tokeniser = AutoTokenizer.from_pretrained(CONFIG.transformer.model)
    train_dataset = NLIDataset(train_pd, tokeniser, CONFIG.transformer.max_length)
    val_dataset = NLIDataset(val_pd, tokeniser, CONFIG.transformer.max_length)

    train_dataloader = DataLoader(train_dataset, generator=generator, batch_size=CONFIG.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, generator=generator, batch_size=CONFIG.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def objective(trial) -> float:
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        bert_lr = trial.suggest_float('bert_lr', 1e-5, 5e-5, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)

        model = TransformerClassifier(num_labels=2, dropout=dropout).to(device)
        optimiser = torch.optim.Adam(model.get_param_groups(classifier_lr=lr, bert_lr=bert_lr))
        criterion = nn.CrossEntropyLoss()

        val_acc = 0.0
        for epoch in range(CONFIG.hyperparameter_tuning.epochs):
            train_one_epoch(device, model, criterion, optimiser, train_dataloader)
            val_loss, val_acc = validate(device, model, criterion, val_dataloader)

            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_acc

    if CONFIG.hyperparameter_tuning.should_run:
        print(f'Running hyperparameter sweep...')
        run_sweep(objective, HYPERPARAMETERS_PATH)

    hyperparameters = json.load(open(HYPERPARAMETERS_PATH, 'r'))
    print(f'Hyperparameters used: {hyperparameters}')
    print()

    model = TransformerClassifier(num_labels=2, dropout=hyperparameters['dropout']).to(device)
    bert_lr = hyperparameters.get('bert_lr', 2e-5)
    optimizer = torch.optim.Adam(model.get_param_groups(classifier_lr=hyperparameters['lr'], bert_lr=bert_lr))
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    frozen_layers = [name for name, p in model.named_parameters() if not p.requires_grad]
    trainable_layers = [name for name, p in model.named_parameters() if p.requires_grad]

    device_info = get_device_info(device)

    config_summary = {
        'model': CONFIG.transformer.model,
        'max_seq_length': CONFIG.transformer.max_length,
        'num_layers': model.bert.config.num_hidden_layers,
        'hidden_size': model.bert.config.hidden_size,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'epochs': CONFIG.epochs,
        'batch_size': CONFIG.batch_size,
        'learning_rate': hyperparameters['lr'],
        'bert_learning_rate': bert_lr,
        'dropout': hyperparameters['dropout'],
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'seed': CONFIG.seed,
        'device': device_info,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'train_batches': len(train_dataloader),
        'val_batches': len(val_dataloader),
        'frozen_layers': frozen_layers,
        'trainable_layers': trainable_layers,
    }

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)
    print(f'Config saved to {run_dir / "config.json"}')

    print('=' * 60)
    print('Training Configuration')
    print('=' * 60)
    print(f'  Model:             {CONFIG.transformer.model}')
    print(f'  Max seq length:    {CONFIG.transformer.max_length}')
    print(f'  Num layers:        {model.bert.config.num_hidden_layers}')
    print(f'  Hidden size:       {model.bert.config.hidden_size}')
    print(f'  Total params:      {total_params:,}')
    print(f'  Trainable params:  {trainable_params:,}')
    print(f'  Frozen params:     {frozen_params:,}')
    print(f'  Epochs:            {CONFIG.epochs}')
    print(f'  Batch size:        {CONFIG.batch_size}')
    print(f'  Learning rate:     {hyperparameters["lr"]:.6f}')
    print(f'  Dropout:           {hyperparameters["dropout"]:.4f}')
    print(f'  Optimizer:         Adam')
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
    print()
    print('Frozen layers:')
    for name in frozen_layers:
        print(f'    {name}')
    print()
    print('Trainable layers:')
    for name in trainable_layers:
        print(f'    {name}')
    print('=' * 60)
    print()

    print(f'Training transformer model...')
    best_acc = 0.0
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
            torch.save(model.state_dict(), MODEL_PATH)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        print(f'Train Loss: {train_loss:.2f} | Train Accuracy: {train_acc * 100:.2f}% | Val Loss: {val_loss:.2f} | Val Accuracy: {val_acc * 100:.2f}%')
        print(f'Epoch time: {epoch_time:.1f}s | Total elapsed: {elapsed:.1f}s')
        print()

    total_time = time.time() - total_start
    print(f'Training complete in {total_time:.1f}s')
    print(f'Best model had an accuracy of {best_acc * 100:.2f}%.')

    # Load best model and evaluate on NLI_trial.csv
    print(f'Running final evaluation on NLI_trial.csv...')
    model.load_state_dict(torch.load(MODEL_PATH))
    test_results = evaluate_test(device, model, tokeniser, TEST_DATA_PATH)
    print(f'NLI_trial — Accuracy: {test_results["accuracy"] * 100:.2f}% | F1 (weighted): {test_results["f1_weighted"]:.4f}')

    # Save results
    results = {
        'training_time_seconds': round(total_time, 1),
        'best_val_accuracy': best_acc,
        'final_train_loss': train_losses[-1],
        'final_train_accuracy': train_accs[-1],
        'final_val_loss': val_losses[-1],
        'final_val_accuracy': val_accs[-1],
        'nli_trial': test_results,
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

    # Save plots
    plots_dir = run_dir / 'plots'
    plot_training_history(train_losses, train_accs, val_losses, val_accs, save_dir=plots_dir)


if __name__ == '__main__':
    main()
