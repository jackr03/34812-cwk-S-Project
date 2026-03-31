print('Starting SLM fine-tuning script...')
import argparse
import gc
import json
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.config import CONFIG, SLMFinetuneConfig
from src.slm_utils.slm_dataset import SLMNLIDatasetInference, format_slm_train_examples
from src.models.slm_classifier import (
    get_lora_config,
    load_base_model_and_tokenizer,
    load_finetuned_for_inference,
    predict_batch,
)
from src.utils import run_sweep

print('Imports complete')

TRAIN_DATA_PATH      = Path('data/train.csv')
VAL_DATA_PATH        = Path('data/dev.csv')
HYPERPARAMETERS_PATH = Path('hyperparameters/slm.json')


def get_device_info() -> dict:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            'type': 'cuda',
            'name': torch.cuda.get_device_name(0),
            'vram_gb': round(props.total_memory / (1024 ** 3), 2),
            'compute_capability': f'{props.major}.{props.minor}',
            'multi_processor_count': props.multi_processor_count,
        }
    return {'type': 'cpu'}


def evaluate_test(device, model, tokeniser, test_path: Path, cfg_slm, batch_size: int = 64) -> dict:
    test_pd = pd.read_csv(test_path)
    test_dataset    = SLMNLIDatasetInference(test_pd, tokeniser, cfg_slm.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_preds  = []
    all_labels = test_pd['label'].tolist()

    for batch in tqdm(test_dataloader, desc=f'Evaluating {test_path.stem}', unit='batches'):
        preds = predict_batch(model, tokeniser, batch['input_ids'], batch['attention_mask'], device)
        all_preds.extend(preds)

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'num_samples': len(all_labels),
    }


class TrainingProgressCallback(TrainerCallback):
    """Logs train loss/accuracy every N steps; computes val accuracy at epoch end."""

    def __init__(self, tokeniser, device, val_pd, cfg_slm):
        self.tokeniser = tokeniser
        self.device    = device
        self.val_pd    = val_pd
        self.cfg_slm   = cfg_slm
        self.val_accs: list = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        formatted = {k: f'{v:.4g}' if isinstance(v, float) else str(v) for k, v in logs.items()}
        tqdm.write(str(formatted))

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        tqdm.write(f'\n  [Epoch {epoch}] Computing val classification accuracy...')
        val_dataset = SLMNLIDatasetInference(self.val_pd, self.tokeniser, self.cfg_slm.max_length)
        val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        model.eval()
        all_preds  = []
        all_labels = self.val_pd['label'].tolist()
        for batch in val_loader:
            preds = predict_batch(model, self.tokeniser, batch['input_ids'], batch['attention_mask'], self.device)
            all_preds.extend(preds)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        self.val_accs.append(accuracy)
        tqdm.write(f'  [Epoch {epoch}] Val Acc: {accuracy * 100:.2f}%  F1: {f1:.4f}\n')
        model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True, help="Directory for this run's outputs")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    adapter_path = Path(f'models/slm_adapter_{timestamp}')
    adapter_path.mkdir(parents=True, exist_ok=True)

    cfg_slm = CONFIG.slm
    cfg_ft  = CONFIG.slm_finetune

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_info = get_device_info()
    print(f'Using device: {device}')
    if device_info['type'] == 'cuda':
        print(f'  GPU: {device_info["name"]} ({device_info["vram_gb"]} GB)')

    torch.manual_seed(CONFIG.seed)

    # Load model and run baseline evaluation before fine-tuning

    print('Loading model...')
    model, tokeniser = load_base_model_and_tokenizer(cfg_slm, cfg_ft)
    tokeniser.model_max_length = cfg_slm.max_length
    print(f'  Model: {cfg_slm.model}')
    print()
    print('=' * 60)
    print('Baseline Evaluation (zero-shot)')
    print('=' * 60)
    baseline_dev = evaluate_test(device, model, tokeniser, VAL_DATA_PATH, cfg_slm)
    print(f'Baseline dev — Acc: {baseline_dev["accuracy"] * 100:.2f}%  F1: {baseline_dev["f1_weighted"]:.4f}')

    train_pd = pd.read_csv(TRAIN_DATA_PATH)
    val_pd   = pd.read_csv(VAL_DATA_PATH)

    fmt = partial(format_slm_train_examples, tokeniser=tokeniser)
    train_hf = Dataset.from_pandas(train_pd).map(fmt, batched=True)
    val_hf   = Dataset.from_pandas(val_pd).map(fmt, batched=True)


    # Hyperparameter tuning with Optuna sweep
    if CONFIG.slm.hyperparameter_tuning.should_run:
        print()
        print('=' * 60)
        print('Hyperparameter Tuning')
        print('=' * 60)
        # Free baseline model before sweep to avoid OOM when loading per-trial
        del model
        gc.collect()
        torch.cuda.empty_cache()

        def objective(trial) -> float:
            lr           = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            lora_r       = trial.suggest_categorical('lora_r', [8, 16, 32])
            lora_alpha   = trial.suggest_categorical('lora_alpha', [16, 32, 64])
            lora_dropout = trial.suggest_float('lora_dropout', 0.0, 0.15)

            trial_cfg = replace(cfg_ft,
                epochs=CONFIG.slm.hyperparameter_tuning.epochs,
                learning_rate=lr,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            trial_model, trial_tok = load_base_model_and_tokenizer(cfg_slm, trial_cfg)

            with tempfile.TemporaryDirectory() as tmp_dir:
                sft_args_t = SFTConfig(
                    output_dir=tmp_dir,
                    per_device_train_batch_size=trial_cfg.batch_size,
                    gradient_accumulation_steps=trial_cfg.gradient_accumulation_steps,
                    learning_rate=trial_cfg.learning_rate,
                    lr_scheduler_type=trial_cfg.lr_scheduler_type,
                    num_train_epochs=trial_cfg.epochs,
                    logging_steps=trial_cfg.logging_steps,
                    eval_strategy='epoch',
                    bf16=True,
                    dataset_text_field='text',
                    packing=True,
                    save_strategy='no',
                    report_to='none',
                )
                trainer_t = SFTTrainer(
                    model=trial_model,
                    train_dataset=train_hf,
                    eval_dataset=val_hf,
                    peft_config=get_lora_config(trial_cfg),
                    args=sft_args_t,
                    processing_class=trial_tok,
                )
                trainer_t.train()

            eval_losses = [e['eval_loss'] for e in trainer_t.state.log_history if 'eval_loss' in e]
            best_eval_loss = min(eval_losses) if eval_losses else float('inf')
            del trial_model, trainer_t
            gc.collect()
            torch.cuda.empty_cache()
            return -best_eval_loss

        run_sweep(objective, HYPERPARAMETERS_PATH, CONFIG.slm.hyperparameter_tuning.trials)

        print('Reloading model for main training run...')
        model, tokeniser = load_base_model_and_tokenizer(cfg_slm, cfg_ft)

    if HYPERPARAMETERS_PATH.exists():
        best_params = json.load(open(HYPERPARAMETERS_PATH))
        cfg_ft = replace(cfg_ft,
            learning_rate=best_params['learning_rate'],
            lora_r=best_params['lora_r'],
            lora_alpha=best_params['lora_alpha'],
            lora_dropout=best_params['lora_dropout'],
        )
        print(f'Hyperparameters loaded from {HYPERPARAMETERS_PATH}: {best_params}')

    val_cb = TrainingProgressCallback(tokeniser, device, val_pd, cfg_slm)

    lora_config = get_lora_config(cfg_ft)

    total_params     = sum(p.numel() for p in model.parameters())
    print()
    print('=' * 60)
    print('SLM Fine-Tuning Configuration')
    print('=' * 60)
    print(f'  Model:              {cfg_slm.model}')
    print(f'  Approach:           LoRA (bf16)')
    print(f'  LoRA rank:          {cfg_ft.lora_r}  alpha: {cfg_ft.lora_alpha}')
    print(f'  Epochs:             {cfg_ft.epochs}')
    print(f'  Batch size:         {cfg_ft.batch_size} x {cfg_ft.gradient_accumulation_steps} accum = {cfg_ft.batch_size * cfg_ft.gradient_accumulation_steps} effective')
    print(f'  Learning rate:      {cfg_ft.learning_rate}')
    print(f'  Device:             {device_info["type"]}')
    if device_info['type'] == 'cuda':
        print(f'  GPU:                {device_info["name"]} ({device_info["vram_gb"]} GB)')
    print(f'  Train samples:      {len(train_pd):,}')
    print(f'  Val samples:        {len(val_pd):,}')
    print('=' * 60)
    print()

    sft_args = SFTConfig(
        output_dir=str(adapter_path),
        per_device_train_batch_size=cfg_ft.batch_size,
        gradient_accumulation_steps=cfg_ft.gradient_accumulation_steps,
        learning_rate=cfg_ft.learning_rate,
        lr_scheduler_type=cfg_ft.lr_scheduler_type,
        num_train_epochs=cfg_ft.epochs,
        logging_steps=cfg_ft.logging_steps,
        eval_strategy='no',
        bf16=True,
        dataset_text_field='text',
        packing=False,
        save_strategy='no',
        report_to='none',
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        peft_config=lora_config,
        args=sft_args,
        processing_class=tokeniser,
        callbacks=[val_cb],
    )

    total_start = time.time()
    trainer.train()
    total_time = time.time() - total_start
    print(f'\nTraining complete in {total_time:.1f}s')

    trainer.model.save_pretrained(str(adapter_path))
    tokeniser.save_pretrained(str(adapter_path))
    print(f'Adapter saved to {adapter_path}')

    # Extract per-epoch losses from trainer log history
    train_losses = [e['loss']      for e in trainer.state.log_history if 'loss'      in e and 'eval_loss' not in e]
    val_losses   = [e['eval_loss'] for e in trainer.state.log_history if 'eval_loss' in e]

    # Evaluate fine-tuned model on dev set
    print()
    print('=' * 60)
    print('Post Fine-Tuning Evaluation')
    print('=' * 60)
    best_model, best_tokeniser = load_finetuned_for_inference(adapter_path, cfg_slm, cfg_ft)
    finetuned_dev = evaluate_test(device, best_model, best_tokeniser, VAL_DATA_PATH, cfg_slm)
    print(f'Fine-tuned dev — Acc: {finetuned_dev["accuracy"] * 100:.2f}%  F1: {finetuned_dev["f1_weighted"]:.4f}')

    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'  Baseline   dev — Acc: {baseline_dev["accuracy"] * 100:.2f}%  F1: {baseline_dev["f1_weighted"]:.4f}')
    print(f'  Fine-tuned dev — Acc: {finetuned_dev["accuracy"] * 100:.2f}%  F1: {finetuned_dev["f1_weighted"]:.4f}')

    # Save config
    config_summary = {
        'model':                 cfg_slm.model,
        'max_seq_length':        cfg_slm.max_length,
        'approach':              'causal_lm_lora_sft',
        'total_params':          total_params,
        'lora_r':                cfg_ft.lora_r,
        'lora_alpha':            cfg_ft.lora_alpha,
        'lora_target_modules':   list(cfg_ft.lora_target_modules),
        'lora_dropout':          cfg_ft.lora_dropout,
        'quantisation':          'none_bf16',
        'epochs':                cfg_ft.epochs,
        'batch_size':            cfg_ft.batch_size,
        'gradient_accumulation': cfg_ft.gradient_accumulation_steps,
        'effective_batch_size':  cfg_ft.batch_size * cfg_ft.gradient_accumulation_steps,
        'learning_rate':         cfg_ft.learning_rate,
        'seed':                  CONFIG.seed,
        'device':                device_info,
        'train_samples':         len(train_pd),
        'val_samples':           len(val_pd),
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)

    best_val_accuracy = max(val_cb.val_accs) if val_cb.val_accs else None

    results = {
        'training_time_seconds': round(total_time, 1),
        'best_val_accuracy': best_val_accuracy,
        'baseline': {
            'dev': baseline_dev,
        },
        'finetuned': {
            'dev': finetuned_dev,
        },
        'epoch_history': {
            'val_accs':     val_cb.val_accs,
            'train_losses': train_losses,
            'val_losses':   val_losses,
        },
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {run_dir / "results.json"}')


if __name__ == '__main__':
    main()
