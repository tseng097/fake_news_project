import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - env dependency guard
    torch = None

if torch is None:
    raise unittest.SkipTest("torch is not installed in this environment")

from src.training_strategies import StrategyConfig, consistency_kl, consistency_kl_symmetric, total_loss


class TrainingStrategyLossTests(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("torch is not installed in this environment")
        self.clean_logits = torch.tensor([[2.0, 0.5], [0.2, 1.6]], dtype=torch.float32)
        self.aug_logits = torch.tensor([[1.6, 0.9], [0.4, 1.3]], dtype=torch.float32)
        self.labels = torch.tensor([0, 1], dtype=torch.long)

    def test_symmetric_kl_equals_average_two_directions(self):
        lhs = consistency_kl_symmetric(self.clean_logits, self.aug_logits)
        rhs = 0.5 * (
            consistency_kl(self.clean_logits, self.aug_logits)
            + consistency_kl(self.aug_logits, self.clean_logits)
        )
        self.assertTrue(torch.isclose(lhs, rhs, atol=1e-7))

    def test_lexical_mhc_uses_symmetric_penalty(self):
        mhc_cfg = StrategyConfig(name="lexical_mhc_lite", consistency_weight=0.5)
        style_cfg = StrategyConfig(name="style_invariance", consistency_weight=0.5)

        mhc_loss = total_loss(mhc_cfg, self.clean_logits, self.labels, aug_logits=self.aug_logits)
        style_loss = total_loss(style_cfg, self.clean_logits, self.labels, aug_logits=self.aug_logits)

        # Typically differs unless KL is perfectly symmetric for these logits.
        self.assertFalse(torch.isclose(mhc_loss, style_loss, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
