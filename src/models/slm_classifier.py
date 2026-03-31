import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import SLMConfig, SLMFinetuneConfig


def load_base_model_and_tokenizer(cfg_slm: SLMConfig, cfg_ft: SLMFinetuneConfig):
    """Load bf16 base model and tokenizer. No PEFT wrapping — SFTTrainer handles that."""
    tokeniser = AutoTokenizer.from_pretrained(cfg_slm.model)
    tokeniser.padding_side = 'left'
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg_slm.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokeniser.pad_token_id
    model.generation_config.pad_token_id = tokeniser.pad_token_id
    return model, tokeniser


def get_lora_config(cfg_ft: SLMFinetuneConfig):
    """Return a LoraConfig built from SLMFinetuneConfig."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg_ft.lora_r,
        lora_alpha=cfg_ft.lora_alpha,
        target_modules=list(cfg_ft.lora_target_modules),
        lora_dropout=cfg_ft.lora_dropout,
        bias='none',
        task_type=TaskType.CAUSAL_LM,
    )


def load_finetuned_for_inference(adapter_path, cfg_slm: SLMConfig, cfg_ft: SLMFinetuneConfig):
    """Reload bf16 base model and attach saved LoRA adapter for inference."""
    from pathlib import Path

    from peft import PeftModel

    tokeniser = AutoTokenizer.from_pretrained(cfg_slm.model)
    tokeniser.padding_side = 'left'
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg_slm.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(Path(adapter_path)))
    model.eval()
    return model, tokeniser


class _LabelOnlyLogitsProcessor:
    """Masks all tokens except '0' and '1' to -inf, forcing the model to output a label."""
    def __init__(self, zero_id: int, one_id: int):
        self.zero_id = zero_id
        self.one_id  = one_id

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.zero_id] = scores[:, self.zero_id]
        mask[:, self.one_id]  = scores[:, self.one_id]
        return mask


def predict_batch(model, tokeniser, input_ids: torch.Tensor, attention_mask: torch.Tensor, device) -> list[int]:
    """Generate exactly 1 token constrained to '0' or '1' — always returns 0 or 1."""
    zero_id = tokeniser.encode('0', add_special_tokens=False)[0]
    one_id  = tokeniser.encode('1', add_special_tokens=False)[0]
    processor = _LabelOnlyLogitsProcessor(zero_id, one_id)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=1,
            do_sample=False,
            logits_processor=[processor],
            pad_token_id=tokeniser.pad_token_id,
        )
    new_tokens = outputs[:, input_ids.shape[1]:]
    decoded = tokeniser.batch_decode(new_tokens, skip_special_tokens=True)
    return [1 if s.strip() == '1' else 0 for s in decoded]
