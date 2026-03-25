import unittest
from unittest.mock import patch

from src.augment import lexical_synonym_perturb, sentiment_shift_simple, style_reframe_simple


class _DummyLemma:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _DummySynset:
    def __init__(self, names, tag=None, sim_map=None):
        self._names = names
        self._tag = tag
        self._sim_map = sim_map or {}

    def lemmas(self):
        return [_DummyLemma(n) for n in self._names]

    def path_similarity(self, other):
        other_tag = getattr(other, "_tag", None)
        return self._sim_map.get(other_tag, 0.0)


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

    def test_tone_word_swap_applies(self):
        text = "A shocking and outrageous claim spread online"
        out = sentiment_shift_simple(text, budget_ratio=0.5, seed=4)
        self.assertTrue("ordinary" in out.lower() or "acceptable" in out.lower())


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

    def test_style_reframe_reduces_elongated_spellings(self):
        text = "This is sooooo SHOCKING!!!"
        out = style_reframe_simple(text)
        self.assertIn("soo", out.lower())
        self.assertNotIn("sooooo", out.lower())
        self.assertIn("shocking!", out.lower())

    def test_style_reframe_normalizes_urls_and_mentions(self):
        text = "BREAKING!!! Read https://example.com NOW @Reporter"
        out = style_reframe_simple(text)
        self.assertIn("<url>", out)
        self.assertIn("<user>", out)
        self.assertNotIn("https://example.com", out)
        self.assertNotIn("@Reporter", out)

    def test_style_reframe_normalizes_hashtags_and_rt_prefix(self):
        text = "RT BREAKING #Election2026 update from officials"
        out = style_reframe_simple(text)
        self.assertNotIn("RT ", out)
        self.assertIn("<hashtag>", out)
        self.assertNotIn("#Election2026", out)


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

    @patch("src.augment.wn")
    def test_lexical_perturb_preserves_mid_sentence_named_entity(self, mock_wn):
        def _synsets(tok):
            if tok.lower() == "biden":
                return [_DummySynset(["leader"])]
            if tok.lower() == "announced":
                return [_DummySynset(["declared"])]
            return []

        mock_wn.synsets.side_effect = _synsets
        text = "Yesterday Biden announced new measures"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=5, protected_words=set())
        self.assertIn("Biden", out)
        self.assertIn("declared", out.lower())

    @patch("src.augment.wn")
    def test_lexical_perturb_filters_morphologically_incompatible_synonym(self, mock_wn):
        # "running" should not be replaced by a base-form verb under mHC-lite guard.
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["sprint"])] if tok.lower() == "running" else []
        text = "The source is running tests"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=17, protected_words=set())
        self.assertEqual(out, text)

    @patch("src.augment.wn")
    def test_lexical_perturb_allows_morphologically_compatible_synonym(self, mock_wn):
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["jogging"])] if tok.lower() == "running" else []
        text = "The source is running tests"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=19, protected_words=set())
        self.assertIn("jogging", out.lower())

    @patch("src.augment.wn")
    def test_lexical_perturb_protects_quantity_adjacent_anchor(self, mock_wn):
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["trillion"])] if tok.lower() == "million" else []
        text = "The report claims 5 million users"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=23, protected_words=set())
        self.assertEqual(out, text)

    @patch("src.augment.wn")
    def test_lexical_perturb_can_edit_when_not_quantity_adjacent(self, mock_wn):
        def _synsets(tok):
            if tok.lower() == "reliable":
                return [_DummySynset(["trustworthy"])]
            if tok.lower() == "million":
                return [_DummySynset(["trillion"])]
            return []

        mock_wn.synsets.side_effect = _synsets
        text = "A reliable report cites 5 million users"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=29, protected_words=set())
        self.assertIn("trustworthy", out.lower())
        self.assertIn("million", out.lower())

    @patch("src.augment.wn")
    def test_lexical_perturb_filters_semantically_distant_synonym(self, mock_wn):
        def _synsets(tok):
            low = tok.lower()
            if low == "reliable":
                return [_DummySynset(["solid", "banana"], tag="reliable", sim_map={"solid": 0.5, "banana": 0.05})]
            if low == "solid":
                return [_DummySynset([], tag="solid")]
            if low == "banana":
                return [_DummySynset([], tag="banana")]
            return []

        mock_wn.synsets.side_effect = _synsets
        text = "A reliable report described the event"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=31, protected_words=set())
        self.assertIn("solid", out.lower())
        self.assertNotIn("banana", out.lower())


if __name__ == "__main__":
    unittest.main()
