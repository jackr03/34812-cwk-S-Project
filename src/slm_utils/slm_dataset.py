import pandas as pd
import torch
from torch import tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _build_slm_messages(premise: str, hypothesis: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are a binary Natural Language Inference classifier. "
                "Given a premise and a hypothesis, predict whether the premise entails the hypothesis."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Premise:\n{premise}\n\n"
                f"Hypothesis:\n{hypothesis}\n\n"
                "Return ONLY the number of the correct label.\n\n"
                "0 = Non-entailment\n"
                "1 = Entailment\n\n"
                "Answer:"
            ),
        },
    ]


def format_slm_train_examples(batch: dict, tokeniser) -> dict:
    """Batched map fn: converts NLI rows → {'text': full chat-template conversation} for SFTTrainer."""
    texts = []
    for premise, hyp, label in zip(batch['premise'], batch['hypothesis'], batch['label']):
        messages = _build_slm_messages(premise, hyp)
        messages.append({'role': 'assistant', 'content': str(int(label))})
        text = tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        texts.append(text)
    return {'text': texts}


class NLIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokeniser: PreTrainedTokenizerBase, max_length: int):
        self.df = df
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        encoded = self.tokeniser(
            row['premise'],
            row['hypothesis'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': tensor(row['label'], dtype=torch.long)
        }


class SLMNLIDataset(Dataset):
    """Training/validation dataset for causal LM fine-tuning.
    Appends the answer token and masks all prompt positions in labels with -100."""

    def __init__(self, df: pd.DataFrame, tokeniser: PreTrainedTokenizerBase, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        messages = _build_slm_messages(row['premise'], row['hypothesis'])
        prompt = self.tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        answer = f" {int(row['label'])}"  # space before digit matches BPE continuation

        prompt_len = len(self.tokeniser(
            prompt, add_special_tokens=False, return_tensors='pt',
        )['input_ids'].squeeze(0))

        full = self.tokeniser(
            prompt + answer,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=False,
            return_tensors='pt',
        )
        input_ids      = full['input_ids'].squeeze(0)
        attention_mask = full['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'label':          tensor(int(row['label']), dtype=torch.long),
        }


class SLMNLIDatasetInference(Dataset):
    """Inference-only dataset. Prompt only, no answer token appended."""

    def __init__(self, df: pd.DataFrame, tokeniser: PreTrainedTokenizerBase, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokeniser = tokeniser
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        messages = _build_slm_messages(row['premise'], row['hypothesis'])
        prompt = self.tokeniser.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokeniser(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            add_special_tokens=False,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label':          tensor(int(row['label']), dtype=torch.long),
        }
