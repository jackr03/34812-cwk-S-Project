"""Evaluate a saved SLM LoRA adapter on dev.csv."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CONFIG
from src.models.nli_dataset import SLMNLIDatasetInference
from src.models.slm_classifier import load_finetuned_for_inference, predict_batch

DEFAULT_ADAPTER_PATH = Path('models/slm_adapter')
VAL_DATA_PATH        = Path('data/dev.csv')


def evaluate(device, model, tokeniser, data_path: Path, cfg_slm, batch_size: int = 64) -> dict:
    df      = pd.read_csv(data_path)
    dataset = SLMNLIDatasetInference(df, tokeniser, cfg_slm.max_length)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_preds  = []
    all_labels = df['label'].tolist()

    for batch in tqdm(loader, desc=f'Evaluating {data_path.stem}', unit='batches'):
        preds = predict_batch(model, tokeniser, batch['input_ids'], batch['attention_mask'], device)
        all_preds.extend(preds)

    return classification_report(all_labels, all_preds, output_dict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-path', type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument('--run-dir',      type=Path, default=Path('runs/slm/eval'))
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)

    cfg_slm = CONFIG.slm
    cfg_ft  = CONFIG.slm_finetune
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loading adapter from {args.adapter_path}...')
    print(f'Loading base model: {cfg_slm.model}')

    model, tokeniser = load_finetuned_for_inference(args.adapter_path, cfg_slm, cfg_ft)

    print('Running final evaluation on benchmark...')
    results = evaluate(device, model, tokeniser, VAL_DATA_PATH, cfg_slm)

    print('[Benchmark Results]')
    print(f'Accuracy:          {results["accuracy"] * 100:.2f}%')
    print(f'Macro Precision:   {results["macro avg"]["precision"]:.4f}')
    print(f'Macro Recall:      {results["macro avg"]["recall"]:.4f}')
    print(f'Macro F1:          {results["macro avg"]["f1-score"]:.4f}')
    print(f'Weighted Precision: {results["weighted avg"]["precision"]:.4f}')
    print(f'Weighted Recall:   {results["weighted avg"]["recall"]:.4f}')
    print(f'Weighted F1:       {results["weighted avg"]["f1-score"]:.4f}')

    out_path = args.run_dir / 'results.json'
    with open(out_path, 'w') as f:
        json.dump({'adapter': str(args.adapter_path), 'dev': results}, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
