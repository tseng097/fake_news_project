from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StrategyConfig:
    name: str = "vanilla"  # vanilla|lexical_mhc_lite|style_invariance|sentiment_invariance
    consistency_weight: float = 0.5
    manifold_weight: float = 0.05
    sentiment_use_js: bool = True


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


def consistency_js(clean_logits, aug_logits):
    """Jensen-Shannon divergence between clean and augmented predictions.

    We use this for sentiment_invariance as a bounded, symmetric consistency loss.
    It is numerically stable for strong sentiment perturbations where directional
    KL can become over-confident in one direction.
    """
    p = F.softmax(clean_logits, dim=-1)
    q = F.softmax(aug_logits, dim=-1)
    m = 0.5 * (p + q)

    log_m = torch.log(m.clamp_min(1e-12))

    kl_pm = F.kl_div(log_m, p, reduction="batchmean")
    kl_qm = F.kl_div(log_m, q, reduction="batchmean")
    return 0.5 * (kl_pm + kl_qm)


def total_loss(strategy: StrategyConfig, clean_logits, labels, aug_logits=None, manifold_loss=None):
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
    - Optional `manifold_loss` supports the simplified manifold-constrained
      hyper-connection regularizer from the model side.
    """
    ce = classification_loss(clean_logits, labels)

    if strategy.name == "vanilla" or aug_logits is None:
        base = ce
    elif strategy.name == "lexical_mhc_lite":
        # mHC-lite main line: symmetric lexical consistency under synonym perturbations.
        base = ce + strategy.consistency_weight * consistency_kl_symmetric(clean_logits, aug_logits)
    elif strategy.name == "sentiment_invariance" and strategy.sentiment_use_js:
        # Paper-grounded tweak (AdSent): keep veracity stable under sentiment flips
        # using a symmetric divergence instead of one-way KL.
        base = ce + strategy.consistency_weight * consistency_js(clean_logits, aug_logits)
    else:
        base = ce + strategy.consistency_weight * consistency_kl(clean_logits, aug_logits)

    # Simple manifold regularization term from mHC-lite structure (if available)
    if manifold_loss is not None:
        base = base + strategy.manifold_weight * manifold_loss

    return base
