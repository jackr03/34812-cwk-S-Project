import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CONFIG


def train_one_epoch(device, model, criterion, optimiser, train_dataloader: DataLoader) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG.lstm.show_progress:
        train_dataloader = tqdm(train_dataloader, desc='Training', unit='batches')

    for batch in train_dataloader:
        premise_ids = batch['premise_ids'].to(device)
        premise_mask = batch['premise_mask'].to(device)
        hypothesis_ids = batch['hypothesis_ids'].to(device)
        hypothesis_mask = batch['hypothesis_mask'].to(device)
        labels = batch['label'].to(device)

        optimiser.zero_grad(set_to_none=True)

        logits = model(premise_ids, premise_mask, hypothesis_ids, hypothesis_mask)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct / total * 100

    return avg_loss, accuracy

def validate(device, model, criterion, val_dataloader: DataLoader) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    if CONFIG.lstm.show_progress:
        val_dataloader = tqdm(val_dataloader, desc='Validating', unit='batches')

    with torch.inference_mode():
        for batch in val_dataloader:
            premise_ids = batch['premise_ids'].to(device)
            premise_mask = batch['premise_mask'].to(device)
            hypothesis_ids = batch['hypothesis_ids'].to(device)
            hypothesis_mask = batch['hypothesis_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(premise_ids, premise_mask, hypothesis_ids, hypothesis_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_dataloader)
    accuracy = correct / total * 100

    return avg_loss, accuracy


def evaluate(device, model, test_dataloader: DataLoader) -> dict:
    model.eval()

    all_preds = []
    all_labels = []

    if CONFIG.lstm.show_progress:
        test_dataloader = tqdm(test_dataloader, desc='Evaluating', unit='batches')

    with torch.inference_mode():
        for batch in test_dataloader:
            premise_ids = batch['premise_ids'].to(device)
            premise_mask = batch['premise_mask'].to(device)
            hypothesis_ids = batch['hypothesis_ids'].to(device)
            hypothesis_mask = batch['hypothesis_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(premise_ids, premise_mask, hypothesis_ids, hypothesis_mask)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        'accuracy': sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100,
        'macro_precision': precision_score(all_labels, all_preds, average='macro'),
        'macro_recall': recall_score(all_labels, all_preds, average='macro'),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_precision': precision_score(all_labels, all_preds, average='weighted'),
        'weighted_recall': recall_score(all_labels, all_preds, average='weighted'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted')
    }
