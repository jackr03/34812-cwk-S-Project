import torch
import torch.nn as nn
from torch import Tensor

from src.config import CONFIG
from src.lstm_utils.lstm_tokeniser import LSTMTokeniser


class LSTMClassifier(nn.Module):
    def __init__(self, tokeniser: LSTMTokeniser, dropout: float):
        super().__init__()

        vectors = tokeniser.glove.vectors
        pad = torch.zeros(1, vectors.shape[1])
        unk = torch.zeros(1, vectors.shape[1])
        matrix = torch.cat([pad, unk, vectors], dim=0)

        self.embedding = nn.Embedding.from_pretrained(matrix, freeze=True, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.encode_lstm = nn.LSTM(
            input_size=CONFIG.lstm.embedding_dim,
            hidden_size=CONFIG.lstm.hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        self.projection = nn.Sequential(
            nn.LazyLinear(CONFIG.lstm.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.compose_lstm = nn.LSTM(
            input_size=CONFIG.lstm.hidden_dim,
            hidden_size=CONFIG.lstm.hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(CONFIG.lstm.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(CONFIG.lstm.hidden_dim, 2)
        )

    def forward(self, premise_ids: Tensor, premise_mask: Tensor, hypothesis_ids: Tensor, hypothesis_mask: Tensor) -> Tensor:
        p_embedding = self.dropout(self.embedding(premise_ids))
        h_embedding = self.dropout(self.embedding(hypothesis_ids))

        p_encoded, _ = self.encode_lstm(p_embedding)
        p_encoded = self.dropout(p_encoded)
        h_encoded, _ = self.encode_lstm(h_embedding)
        h_encoded = self.dropout(h_encoded)

        scores = torch.bmm(p_encoded, h_encoded.transpose(1, 2))

        p_mask = premise_mask.unsqueeze(2).float()
        h_mask = hypothesis_mask.unsqueeze(1).float()
        scores = scores.masked_fill((p_mask * h_mask) == 0, -1e4)

        p_aligned = torch.bmm(torch.softmax(scores, dim=2), h_encoded)
        h_aligned = torch.bmm(torch.softmax(scores, dim=1).transpose(1, 2), p_encoded)

        p_enhanced = torch.cat([p_encoded, p_aligned, p_encoded - p_aligned, p_encoded * p_aligned], dim=-1)
        h_enhanced = torch.cat([h_encoded, h_aligned, h_encoded - h_aligned, h_encoded * h_aligned], dim=-1)

        p_projected = self.projection(p_enhanced)
        h_projected = self.projection(h_enhanced)

        p_composed, _ = self.compose_lstm(p_projected)
        p_composed = self.dropout(p_composed)
        h_composed, _ = self.compose_lstm(h_projected)
        h_composed = self.dropout(h_composed)

        p_mask_exp = premise_mask.unsqueeze(-1).float()
        h_mask_exp = hypothesis_mask.unsqueeze(-1).float()

        p_avg = (p_composed * p_mask_exp).sum(dim=1) / p_mask_exp.sum(dim=1).clamp(min=1)
        h_avg = (h_composed * h_mask_exp).sum(dim=1) / h_mask_exp.sum(dim=1).clamp(min=1)

        p_max = p_composed.masked_fill(p_mask_exp == 0, -1e4).max(dim=1).values
        h_max = h_composed.masked_fill(h_mask_exp == 0, -1e4).max(dim=1).values

        pooled = torch.cat([p_avg, p_max, h_avg, h_max], dim=-1)
        return self.classifier(pooled)
