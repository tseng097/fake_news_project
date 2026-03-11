from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class ModelConfig:
    backbone: str = "roberta-base"
    num_labels: int = 2
    dropout: float = 0.1
    # Simple mHC-lite structure switch
    use_mhc_lite: bool = True
    manifold_dim: int = 128


class RobertaFakeNewsClassifier(nn.Module):
    """RoBERTa classifier with optional simple manifold-constrained hyper-connections.

    mHC-lite structure (simplified):
    1) Build two sequence views: [CLS] token and masked mean-pooled token vector.
    2) Fuse them through a lightweight hyper-connection (gated residual feature).
    3) Enforce a manifold bottleneck via projection+reconstruction; reconstruction MSE
       is exposed as an auxiliary regularization term.

    This is intentionally lightweight for this project (not full original mHC paper reproduction).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(cfg.backbone)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(hidden, cfg.num_labels)

        self.use_mhc_lite = cfg.use_mhc_lite
        if self.use_mhc_lite:
            self.hyper_gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Tanh(),
            )
            self.manifold_proj = nn.Linear(hidden, cfg.manifold_dim)
            self.manifold_recon = nn.Linear(cfg.manifold_dim, hidden)
            self.recon_loss = nn.MSELoss()

    @staticmethod
    def _masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, input_ids, attention_mask, return_aux: bool = False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]

        manifold_loss = None
        feat = cls

        if self.use_mhc_lite:
            mean_vec = self._masked_mean(out.last_hidden_state, attention_mask)
            # Hyper-connection: combine CLS with sequence-global context.
            hyper = self.hyper_gate(torch.cat([cls, mean_vec], dim=-1))
            feat = cls + hyper

            # Simple manifold constraint through bottleneck reconstruction.
            z = self.manifold_proj(feat)
            feat_recon = self.manifold_recon(z)
            manifold_loss = self.recon_loss(feat_recon, feat)

        logits = self.classifier(self.dropout(feat))

        if return_aux:
            return logits, {"manifold_loss": manifold_loss}
        return logits
