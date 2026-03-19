import pandas as pd
import torch
from torch import tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


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
