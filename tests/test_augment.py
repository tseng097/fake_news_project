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
        text = "good great excellent positive success benefit safe terrible"
        out1 = sentiment_shift_simple(text, budget_ratio=0.4, seed=1)
        out2 = sentiment_shift_simple(text, budget_ratio=0.4, seed=2)
        out3 = sentiment_shift_simple(text, budget_ratio=0.4, seed=3)
        self.assertTrue(out1 != out2 or out1 != out3 or out2 != out3)

    def test_tone_word_swap_applies(self):
        text = "A shocking and outrageous claim spread online"
        out = sentiment_shift_simple(text, budget_ratio=0.5, seed=4)
        self.assertTrue("ordinary" in out.lower() or "acceptable" in out.lower())

    def test_emoji_sentiment_swap_applies(self):
        text = "This report is amazing 🙂"
        out = sentiment_shift_simple(text, budget_ratio=0.5, seed=8)
        self.assertIn("🙁", out)

    def test_emoticon_swap_respects_budget(self):
        text = "Great work :) truly excellent"
        out = sentiment_shift_simple(text, budget_ratio=0.1, seed=10)
        # One-token-equivalent budget should allow at most one sentiment cue swap.
        changed_signals = sum(
            1
            for marker in [":(", "awful", "terrible", "negative", "failure"]
            if marker in out.lower()
        )
        self.assertLessEqual(changed_signals, 1)


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

    def test_style_reframe_normalizes_unicode_punctuation(self):
        text = "“Breaking” update — details coming…"
        out = style_reframe_simple(text)
        self.assertIn('"Breaking"', out)
        self.assertIn("- details coming...", out)

    def test_style_reframe_strips_markdown_and_clickbait_prefix(self):
        text = "[BREAKING] **SHOCKING** _claim_ was posted"
        out = style_reframe_simple(text)
        self.assertNotIn("[BREAKING]", out)
        self.assertNotIn("**", out)
        self.assertNotIn("_", out)
        self.assertIn("shocking", out.lower())

    def test_style_reframe_removes_leading_section_label(self):
        text = "OPINION: BREAKING CLAIM spreads quickly online"
        out = style_reframe_simple(text)
        self.assertFalse(out.lower().startswith("opinion:"))
        self.assertIn("breaking claim", out.lower())

    def test_style_reframe_removes_fact_check_label_with_spacing(self):
        text = "FACT CHECK : This post is going viral"
        out = style_reframe_simple(text)
        self.assertFalse(out.lower().startswith("fact check"))
        self.assertIn("this post is going viral", out.lower())

    def test_style_reframe_removes_leading_source_byline_prefix(self):
        text = "Reuters - BREAKING UPDATE: Officials deny the claim"
        out = style_reframe_simple(text)
        self.assertFalse(out.lower().startswith("reuters -"))
        self.assertIn("breaking update: officials deny the claim", out.lower())


class LexicalMhcLiteSafetyTests(unittest.TestCase):
    @patch("src.augment.wn")
    def test_lexical_perturb_protects_modifier_pivot_words(self, mock_wn):
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["merely"])] if tok.lower() == "only" else []
        text = "Only the headline was updated"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=11)
        self.assertEqual(out, text)

    @patch("src.augment.wn")
    def test_lexical_perturb_protects_discourse_pivot_words(self, mock_wn):
        mock_wn.synsets.side_effect = lambda tok: [_DummySynset(["nonetheless"])] if tok.lower() == "however" else []
        text = "However the claim remained unsupported"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=13, protected_words=set())
        self.assertEqual(out, text)

    @patch("src.augment.wn")
    def test_lexical_perturb_protects_quantifier_comparative_pivots(self, mock_wn):
        def _synsets(tok):
            if tok.lower() == "many":
                return [_DummySynset(["numerous"])]
            if tok.lower() == "reports":
                return [_DummySynset(["records"])]
            return []

        mock_wn.synsets.side_effect = _synsets
        text = "Many reports described the claim"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=41, protected_words=set())
        # "many" should be protected; non-pivot words can still be perturbed.
        self.assertIn("many", out.lower())

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
    def test_lexical_perturb_preserves_sentence_initial_acronym_entity(self, mock_wn):
        def _synsets(tok):
            if tok == "NASA":
                return [_DummySynset(["agency"])]
            if tok.lower() == "announced":
                return [_DummySynset(["declared"])]
            return []

        mock_wn.synsets.side_effect = _synsets
        text = "NASA announced a new launch window"
        out = lexical_synonym_perturb(text, budget_ratio=1.0, seed=6, protected_words=set())
        # mHC-lite guard should preserve the acronym entity anchor.
        self.assertIn("NASA", out)
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
