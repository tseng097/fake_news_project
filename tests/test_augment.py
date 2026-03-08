import unittest

from src.augment import sentiment_shift_simple


class SentimentShiftSimpleTests(unittest.TestCase):
    def test_phrase_level_swap_applies(self):
        text = "The claim is not good and was widely praised yesterday."
        out = sentiment_shift_simple(text, budget_ratio=0.4)
        self.assertIn("very bad", out.lower())
        self.assertIn("widely criticized", out.lower())

    def test_budget_limits_replacements(self):
        text = "good great excellent positive success"
        out = sentiment_shift_simple(text, budget_ratio=0.2)
        changed = sum(1 for w in ["bad", "terrible", "awful", "negative", "failure"] if w in out.lower())
        self.assertLessEqual(changed, 1)


if __name__ == "__main__":
    unittest.main()
