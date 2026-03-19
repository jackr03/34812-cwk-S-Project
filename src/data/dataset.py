import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class NLIDataset(Dataset):
    """Pre-tokenizes premise–hypothesis pairs into tensors at init time."""

    def __init__(self, csv_path: str, tokenizer, max_length: int):
        df = pd.read_csv(csv_path)
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)

        encodings = tokenizer(
            df['premise'].tolist(),
            df['hypothesis'].tolist(),
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.labels[idx],
        }


def create_dataloader(csv_path: str, tokenizer, max_length: int,
                      batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = NLIDataset(csv_path, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
