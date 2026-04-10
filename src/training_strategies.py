from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StrategyConfig:
    name: str = "vanilla"  # vanilla|lexical_mhc_lite|style_invariance|sentiment_invariance
    consistency_weight: float = 0.5
    manifold_weight: float = 0.05
    sentiment_use_js: bool = True
    # Confidence gate for augmentation consistency.
    # Only clean predictions with max-prob >= threshold contribute to
    # consistency regularization, reducing noisy invariance pressure.
    confidence_threshold: float = 0.0
    # mHC-lite option: apply the same confidence gate to lexical consistency.
    # This is useful when synonym perturbations are occasionally low-quality and
    # could otherwise over-regularize uncertain samples.
    lexical_confidence_gate: bool = True
    # Optional hard-example emphasis for style/sentiment invariance.
    # Inspired by adversarial group-reweighting ideas (e.g., AdComment/IDR):
    # increase consistency pressure when clean/aug predictions diverge more.
    adaptive_consistency_focus: bool = True
    # Upper bound on adaptive scaling to avoid destabilizing optimization.
    adaptive_focus_max_scale: float = 1.5
    # Optional temperature for consistency-only comparisons.
    # temp>1 softens over-confident distributions under style/sentiment shift,
    # while keeping the supervised CE branch unchanged.
    consistency_temperature: float = 1.0


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


def _consistency_temperature_scale(logits, temperature: float):
    """Scale logits for consistency losses only.

    mHC-lite/style/sentiment consistency can become too peaky when the model is
    very confident on one view (especially under lexical paraphrases). Applying
    a temperature >1.0 softens both views before divergence computation, which
    reduces brittle over-penalization while preserving the CE supervision path.
    """
    if temperature is None or temperature <= 0:
        return logits
    if temperature == 1.0:
        return logits
    return logits / temperature


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


def _consistency_confidence_scale(clean_logits, threshold: float) -> torch.Tensor:
    """Return a scalar [0,1] weighting consistency by clean-view confidence.

    Motivation: in adversarial style/sentiment training, some perturbations can
    become semantically ambiguous. Following confidence-filtering intuition from
    robust pseudo-labeling, we down-weight consistency when the clean prediction
    itself is low-confidence.
    """
    if threshold <= 0.0:
        return clean_logits.new_tensor(1.0)

    probs = F.softmax(clean_logits, dim=-1)
    conf = probs.max(dim=-1).values
    mask = (conf >= threshold).float()
    return mask.mean()


def _adaptive_consistency_focus_scale(
    clean_logits,
    aug_logits,
    enabled: bool,
    max_scale: float,
) -> torch.Tensor:
    """Return a detached scalar >=1 emphasizing harder style/sentiment pairs.

    Paper grounding: robust fake-news work with adversarial comment groups
    (e.g., AdComment's adaptive resampling) upweights vulnerable regions during
    training. For our fixed strategy scope, we apply a lightweight analogue:
    larger clean-vs-aug disagreement yields slightly higher consistency weight.

    We use detached JS divergence as a bounded hardness signal and clamp by
    ``max_scale`` to keep optimization stable.
    """
    if not enabled:
        return clean_logits.new_tensor(1.0)

    js = consistency_js(clean_logits, aug_logits).detach()
    scale = 1.0 + js
    if max_scale > 1.0:
        scale = torch.clamp(scale, max=max_scale)
    return scale


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

    # Keep supervision unchanged; only soften logits for consistency comparisons.
    clean_cons = _consistency_temperature_scale(
        clean_logits, strategy.consistency_temperature
    )
    aug_cons = _consistency_temperature_scale(
        aug_logits, strategy.consistency_temperature
    ) if aug_logits is not None else None

    if strategy.name == "vanilla" or aug_logits is None:
        base = ce
    elif strategy.name == "lexical_mhc_lite":
        # mHC-lite main line: symmetric lexical consistency under synonym perturbations.
        # NOTE: we apply optional consistency_temperature here too, so lexical
        # disagreement pressure can be softened without changing label CE.
        #
        # Paper-grounded robustness/generalization tweak: lexical substitutions
        # can be noisy on ambiguous samples, so optionally gate mHC consistency
        # by clean-view confidence to avoid over-penalizing uncertain examples.
        conf_scale = (
            _consistency_confidence_scale(clean_logits, strategy.confidence_threshold)
            if strategy.lexical_confidence_gate
            else clean_logits.new_tensor(1.0)
        )
        base = ce + strategy.consistency_weight * conf_scale * consistency_kl_symmetric(clean_cons, aug_cons)
    elif strategy.name == "sentiment_invariance" and strategy.sentiment_use_js:
        # Paper-grounded tweak (AdSent): keep veracity stable under sentiment flips
        # using a symmetric divergence instead of one-way KL.
        conf_scale = _consistency_confidence_scale(clean_logits, strategy.confidence_threshold)
        focus_scale = _adaptive_consistency_focus_scale(
            clean_cons,
            aug_cons,
            enabled=strategy.adaptive_consistency_focus,
            max_scale=strategy.adaptive_focus_max_scale,
        )
        base = ce + strategy.consistency_weight * conf_scale * focus_scale * consistency_js(clean_cons, aug_cons)
    else:
        # Style/sentiment branch uses optional confidence gating to avoid forcing
        # invariance on uncertain clean predictions.
        conf_scale = _consistency_confidence_scale(clean_logits, strategy.confidence_threshold)
        focus_scale = _adaptive_consistency_focus_scale(
            clean_cons,
            aug_cons,
            enabled=strategy.adaptive_consistency_focus,
            max_scale=strategy.adaptive_focus_max_scale,
        )
        base = ce + strategy.consistency_weight * conf_scale * focus_scale * consistency_kl(clean_cons, aug_cons)

    # Simple manifold regularization term from mHC-lite structure (if available)
    if manifold_loss is not None:
        base = base + strategy.manifold_weight * manifold_loss

    return base
