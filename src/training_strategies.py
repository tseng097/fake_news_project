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
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")


def total_loss(strategy: StrategyConfig, clean_logits, labels, aug_logits=None):
    """
    Generic objective:
    - vanilla: CE(clean)
    - robust variants: CE(clean) + lambda * KL(clean || augmented)

    mHC-lite mapping here = lexical invariance under synonym perturbation.
    """
    ce = classification_loss(clean_logits, labels)
    if strategy.name == "vanilla" or aug_logits is None:
        return ce
    return ce + strategy.consistency_weight * consistency_kl(clean_logits, aug_logits)
