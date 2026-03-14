import unittest
from unittest.mock import patch

from src.augment import lexical_synonym_perturb, sentiment_shift_simple, style_reframe_simple


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
        out = sentiment_shift_simple(text, budget_ratio=0.4, seed=1)
        self.assertIn("very bad", out.lower())
        self.assertIn("widely criticized", out.lower())

    def test_budget_limits_replacements(self):
        text = "good great excellent positive success"
        out = sentiment_shift_simple(text, budget_ratio=0.2, seed=9)
        changed = sum(1 for w in ["bad", "terrible", "awful", "negative", "failure"] if w in out.lower())
        self.assertLessEqual(changed, 1)

    def test_seed_makes_sentiment_attack_deterministic(self):
        text = "good great excellent positive success benefit"
        out1 = sentiment_shift_simple(text, budget_ratio=0.5, seed=42)
        out2 = sentiment_shift_simple(text, budget_ratio=0.5, seed=42)
        self.assertEqual(out1, out2)

    def test_different_seed_changes_attack_pattern(self):
        text = "good great excellent positive success benefit"
        out1 = sentiment_shift_simple(text, budget_ratio=0.5, seed=1)
        out2 = sentiment_shift_simple(text, budget_ratio=0.5, seed=2)
        self.assertNotEqual(out1, out2)


class StyleReframeSimpleTests(unittest.TestCase):
    def test_style_reframe_normalizes_all_caps_and_punctuation(self):
        text = "BREAKING NEWS!!!! THIS REPORT IS SHOCKING!!"
        out = style_reframe_simple(text)
        self.assertIn("breaking news!", out)
        self.assertIn("this report is shocking!", out)

    def test_style_reframe_preserves_common_acronyms(self):
        text = "USA OFFICIALS SAID IT'S TRUE!!!"
        out = style_reframe_simple(text)
        self.assertIn("USA", out)
        self.assertIn("officials said", out)
        self.assertIn("it is", out)


class LexicalMhcLiteSafetyTests(unittest.TestCase):
    @patch("src.augment.wn")
    def test_lexical_perturb_protects_modifier_pivot_words(self, mock_wn):
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["merely"])] if tok.lower() == "only" else []
        text = "Only the headline was updated"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=11)
        self.assertEqual(out, text)

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
