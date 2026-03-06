"""Minimal smoke test for the R-Drop loss wiring.

Run:
  python smoke_rdrop.py

This does NOT download datasets/models; it just checks loss computation shapes.
"""

import torch

from src.training_strategies import StrategyConfig, total_loss


def main():
    torch.manual_seed(0)
    bsz, ncls = 4, 2
    logits1 = torch.randn(bsz, ncls)
    logits2 = torch.randn(bsz, ncls)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loss = total_loss(StrategyConfig(name="rdrop", consistency_weight=0.5), logits1, labels, clean_logits_2=logits2)
    assert torch.isfinite(loss).item(), "loss is not finite"
    assert loss.item() >= 0.0, "loss should be non-negative"
    print("OK: rdrop loss computed", float(loss))


if __name__ == "__main__":
    main()
