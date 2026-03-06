from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class ModelConfig:
    backbone: str = "roberta-base"
    num_labels: int = 2
    dropout: float = 0.1


class RobertaFakeNewsClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(cfg.backbone)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(hidden, cfg.num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls))
        return logits
