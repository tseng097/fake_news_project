from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StrategyConfig:
    name: str = "vanilla"  # vanilla|lexical_mhc_lite|style_invariance|sentiment_invariance|rdrop
    consistency_weight: float = 0.5


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def consistency_kl(p_logits, q_logits):
    """KL( p || q ) where p,q are categorical distributions parameterized by logits."""
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")


def symmetric_kl(p_logits, q_logits):
    """0.5*(KL(p||q) + KL(q||p)) used by R-Drop-style regularization."""
    return 0.5 * (consistency_kl(p_logits, q_logits) + consistency_kl(q_logits, p_logits))


def total_loss(strategy: StrategyConfig, clean_logits, labels, aug_logits=None, clean_logits_2=None):
    """Compute training loss.

    Modes:
    - vanilla: CE(clean)
    - lexical_mhc_lite/style_invariance/sentiment_invariance:
        CE(clean) + lambda * KL(clean || augmented)
    - rdrop:
        average CE over 2 stochastic forward passes + lambda * symmetric KL between them

    Notes:
    - "mHC-lite" in this repo = lexical invariance under synonym perturbation.
    - R-Drop does not require explicit text augmentation; it regularizes prediction
      consistency under dropout noise.
    """
    if strategy.name == "rdrop":
        if clean_logits_2 is None:
            raise ValueError("rdrop requires clean_logits_2 (2nd stochastic forward pass)")
        ce1 = classification_loss(clean_logits, labels)
        ce2 = classification_loss(clean_logits_2, labels)
        return 0.5 * (ce1 + ce2) + strategy.consistency_weight * symmetric_kl(clean_logits, clean_logits_2)

    ce = classification_loss(clean_logits, labels)
    if strategy.name == "vanilla" or aug_logits is None:
        return ce
    return ce + strategy.consistency_weight * consistency_kl(clean_logits, aug_logits)
