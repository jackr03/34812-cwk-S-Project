import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.config import CONFIG


# TODO: How big does the classifier need to be?
class TransformerClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float):
        super().__init__()
        self.bert = AutoModel.from_pretrained(CONFIG.transformer_model)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_labels)
        )

        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        last_hidden_states = self.bert(x)[0]
        return self.classifier(last_hidden_states)
