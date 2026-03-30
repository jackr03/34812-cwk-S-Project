from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CONFIG
from src.lstm_utils.lstm_tokeniser import LSTMTokeniser


class LSTMDataset(Dataset):
    def __init__(self, csv_path: Path, tokeniser: LSTMTokeniser, evaluate: bool = False):
        df = pd.read_csv(csv_path)

        premise_pairs = [tokeniser.encode(text, CONFIG.lstm.max_length) for text in df['premise']]
        self.premise_ids = torch.tensor([pair[0] for pair in premise_pairs])
        self.premise_masks = torch.tensor([pair[1] for pair in premise_pairs])

        hypothesis_pairs = [tokeniser.encode(text, CONFIG.lstm.max_length) for text in df['hypothesis']]
        self.hypothesis_ids = torch.tensor([pair[0] for pair in hypothesis_pairs])
        self.hypothesis_masks = torch.tensor([pair[1] for pair in hypothesis_pairs])
        if not evaluate:
            self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.premise_ids)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        item = {
            'premise_ids': self.premise_ids[idx],
            'premise_mask': self.premise_masks[idx],
            'hypothesis_ids': self.hypothesis_ids[idx],
            'hypothesis_mask': self.hypothesis_masks[idx],
        }
        if self.labels is not None:
            item['label'] = self.labels[idx]
        return item
