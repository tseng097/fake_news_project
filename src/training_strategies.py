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


def consistency_kl_symmetric(clean_logits, aug_logits):
    """Symmetric KL for mHC-lite consistency.

    mHC-lite currently uses synonym-based lexical perturbations that are intended
    to preserve semantics. A symmetric agreement penalty encourages both views to
    align instead of treating one view as always privileged.
    """
    return 0.5 * (
        consistency_kl(clean_logits, aug_logits)
        + consistency_kl(aug_logits, clean_logits)
    )


def total_loss(strategy: StrategyConfig, clean_logits, labels, aug_logits=None):
    """Compute training loss for the allowed strategy scope.

    Modes (project-agreed scope):
    - vanilla:
        Standard supervised objective on clean text only.
        L = CE(clean)

    - lexical_mhc_lite:
        mHC-lite consistency with symmetric KL.
        L = CE(clean) + lambda * 0.5 * [KL(clean||aug) + KL(aug||clean)]

    - style_invariance / sentiment_invariance:
        Directional consistency objective.
        L = CE(clean) + lambda * KL(clean||aug)

    Why this matches mHC-lite intent here:
    - We keep a stable prediction manifold around the clean sample by requiring
      agreement with a meaning-preserving transformed view.
    - For lexical_mhc_lite, the transformed view is synonym-perturbed text,
      which is the designated mHC-lite path in this repository.
    - Symmetric KL reduces direction bias between clean and lexical variants,
      a lightweight robustness enhancement without changing strategy scope.

    Notes:
    - Do not introduce extra strategies outside the agreed scope unless user asks.
    - If no augmented logits are provided, the function safely falls back to CE.
    """
    ce = classification_loss(clean_logits, labels)
    if strategy.name == "vanilla" or aug_logits is None:
        return ce
    if strategy.name == "lexical_mhc_lite":
        return ce + strategy.consistency_weight * consistency_kl_symmetric(clean_logits, aug_logits)
    return ce + strategy.consistency_weight * consistency_kl(clean_logits, aug_logits)
