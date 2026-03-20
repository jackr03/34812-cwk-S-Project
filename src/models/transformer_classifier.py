import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.config import CONFIG


# TODO: How big does the classifier need to be?
class TransformerClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float):
        super().__init__()
        self.bert = AutoModel.from_pretrained(CONFIG.transformer.model)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_labels)
        )

        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze top 2 transformer layers and final LayerNorm
        for name, param in self.bert.named_parameters():
            if name.startswith('transformer.layer.4.') or name.startswith('transformer.layer.5.') or name.startswith('transformer.output_layer_norm.'):
                param.requires_grad = True

    def get_param_groups(self, classifier_lr: float, bert_lr: float) -> list[dict]:
        bert_params = [p for n, p in self.bert.named_parameters() if p.requires_grad]
        classifier_params = list(self.classifier.parameters())
        return [
            {'params': bert_params, 'lr': bert_lr},
            {'params': classifier_params, 'lr': classifier_lr},
        ]

    def forward(self, input_ids: Tensor, attention_masks: Tensor) -> Tensor:
        last_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_masks)
        cls_output = last_hidden_states.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)
