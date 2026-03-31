"""Zero-shot baseline: Qwen3-8B (no LoRA) evaluated on train/dev/NLI_trial splits."""
import json
import pandas as pd
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import CONFIG
from src.models.nli_dataset import _build_slm_messages
from src.run_slm import evaluate_test

SPLITS = {
    'dev':       Path('data/dev.csv'),
    'NLI_trial': Path('data/NLI_trial.csv'),
}

N_DEBUG = 5


def debug_samples(model, tokeniser, data_path: Path, cfg_slm, device):
    """Print raw prompt and raw model generation for the first N_DEBUG samples."""
    df = pd.read_csv(data_path).head(N_DEBUG)
    print(f'\n{"="*60}')
    print(f'DEBUG: first {N_DEBUG} samples from {data_path.name}')
    print(f'{"="*60}')
    for i, row in df.iterrows():
        messages = _build_slm_messages(row['premise'], row['hypothesis'])
        prompt = tokeniser.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokeniser(
            prompt,
            max_length=cfg_slm.max_length,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,
        ).input_ids.to(device)

        with torch.inference_mode():
            out_ids = model.generate(
                ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokeniser.pad_token_id,
            )
        decoded_prompt = tokeniser.decode(ids[0], skip_special_tokens=False)
        decoded_output = tokeniser.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=False)

        print(f'\n--- Sample {i} | label={row["label"]} ---')
        print(f'[PROMPT] (last 200 chars): ...{decoded_prompt[-200:]}')
        print(f'[OUTPUT]: {repr(decoded_output)}')
    print(f'{"="*60}\n')


def main():
    cfg_slm = CONFIG.slm
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading {cfg_slm.model} (bf16, no LoRA)...')
    model = AutoModelForCausalLM.from_pretrained(
        cfg_slm.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    tokeniser = AutoTokenizer.from_pretrained(cfg_slm.model)
    tokeniser.padding_side = 'left'
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
    model.eval()

    debug_samples(model, tokeniser, Path('data/dev.csv'), cfg_slm, device)

    all_results = {}
    for split_name, split_path in SPLITS.items():
        print(f'\nRunning zero-shot eval on {split_path.name}...')
        results = evaluate_test(device, model, tokeniser, split_path, cfg_slm)
        all_results[split_name] = results
        print(f'  [{split_name}] Accuracy: {results["accuracy"] * 100:.2f}%  |  F1 weighted: {results["f1_weighted"]:.4f}')

    print('\n=== Zero-shot summary ===')
    for split_name, results in all_results.items():
        print(f'  {split_name:<12} acc={results["accuracy"]*100:.2f}%  f1={results["f1_weighted"]:.4f}')
    print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
