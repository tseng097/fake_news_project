from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StrategyConfig:
    name: str = "vanilla"  # vanilla|lexical_mhc_lite|style_invariance|sentiment_invariance
    consistency_weight: float = 0.5


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def consistency_kl(p_logits, q_logits):
    """KL( p || q ) where p,q are categorical distributions parameterized by logits."""
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")


def total_loss(strategy: StrategyConfig, clean_logits, labels, aug_logits=None):
    """Compute training loss for the allowed strategy scope.

    Modes (project-agreed scope):
    - vanilla:
        Standard supervised objective on clean text only.
        L = CE(clean)

    - lexical_mhc_lite / style_invariance / sentiment_invariance:
        Consistency-regularized objective.
        L = CE(clean) + lambda * KL(clean || augmented)

    Why this matches mHC-lite intent here:
    - We keep a stable prediction manifold around the clean sample by requiring
      agreement with a meaning-preserving transformed view.
    - For lexical_mhc_lite, the transformed view is synonym-perturbed text,
      which is the designated mHC-lite path in this repository.

    Notes:
    - Do not introduce extra strategies outside the agreed scope unless user asks.
    - If no augmented logits are provided, the function safely falls back to CE.
    """
    ce = classification_loss(clean_logits, labels)
    if strategy.name == "vanilla" or aug_logits is None:
        return ce
    return ce + strategy.consistency_weight * consistency_kl(clean_logits, aug_logits)
