import unittest
from unittest.mock import patch

from src.augment import lexical_synonym_perturb, sentiment_shift_simple


class _DummyLemma:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _DummySynset:
    def __init__(self, names):
        self._names = names

    def lemmas(self):
        return [_DummyLemma(n) for n in self._names]


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


class LexicalMhcLiteSafetyTests(unittest.TestCase):
    @patch("src.augment.wn")
    def test_lexical_perturb_reverts_on_sentiment_sign_flip(self, mock_wn):
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["bad"])] if tok.lower() == "good" else []
        text = "This is a good report"
        out = lexical_synonym_perturb(text, budget_ratio=0.5, seed=7, protected_words=set())
        self.assertEqual(out, text)

    @patch("src.augment.wn")
    def test_lexical_perturb_keeps_candidate_when_no_sign_flip(self, mock_wn):
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["solid"])] if tok.lower() == "reliable" else []
        text = "A reliable source described events clearly"
        out = lexical_synonym_perturb(text, budget_ratio=0.5, seed=3, protected_words=set())
        self.assertNotEqual(out, text)
        self.assertIn("solid", out.lower())


if __name__ == "__main__":
    unittest.main()
