import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer

from src.config import CONFIG


class BiLSTMClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.transformer.model)
        vocab_size = len(tokenizer)
        cfg = CONFIG.bilstm

        self.embedding = nn.Embedding(vocab_size, cfg.embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.bilstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            bidirectional=True,
            dropout=dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(cfg.hidden_dim * 2, num_labels)

    def get_param_groups(self, lr: float) -> list[dict]:
        return [{'params': self.parameters(), 'lr': lr}]

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        embedded = self.embedding(input_ids)                          # (B, L, E)
        output, _ = self.bilstm(embedded)                            # (B, L, 2H)
        mask = attention_mask.unsqueeze(-1).float()                  # (B, L, 1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)        # (B, 2H)
        return self.classifier(self.dropout(pooled))                 # (B, num_labels)
