from __future__ import annotations

import random
import re
from typing import List


try:
    from nltk.corpus import wordnet as wn
except Exception:
    wn = None


CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "I'm": "I am",
    "it's": "it is",
    "that's": "that is",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not",
}

# Style-attack text often includes Unicode typography and markdown wrappers to
# mimic outlet-specific voice. We normalize these to reduce superficial style
# cues while preserving proposition-level content.
STYLE_CHAR_NORMALIZATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
)

SENTIMENT_SWAP = {
    "good": "bad",
    "great": "terrible",
    "excellent": "awful",
    "positive": "negative",
    "benefit": "harm",
    "success": "failure",
    "safe": "dangerous",
    "bad": "good",
    "terrible": "great",
    "awful": "excellent",
    "negative": "positive",
    "harm": "benefit",
    "failure": "success",
    "dangerous": "safe",
}

# Additional tone-bearing sentiment words often used in sensational headlines.
# This extends sentiment attacks beyond simple polarity lexemes (AdSent-style
# multi-granular sentiment perturbation) while staying within project scope.
SENTIMENT_TONE_SWAP = {
    "shocking": "ordinary",
    "outrageous": "acceptable",
    "amazing": "unremarkable",
    "terrifying": "reassuring",
    "disaster": "success",
    "ordinary": "shocking",
    "acceptable": "outrageous",
    "unremarkable": "amazing",
    "reassuring": "terrifying",
}

# Controlled, phrase-level swaps inspired by adversarial sentiment attacks.
# Phrase replacements are executed before token-level swaps.
SENTIMENT_PHRASE_SWAP = {
    "not good": "very bad",
    "not bad": "quite good",
    "highly effective": "poorly effective",
    "widely praised": "widely criticized",
    "strong evidence": "weak evidence",
    "weak evidence": "strong evidence",
}

# Emoticon/emoji sentiment pivots used in social posts and comments.
# ANN-style robustness findings note that sentiment-bearing emotes can become
# exploitable cues; we therefore include a small, controlled inverse mapping
# inside sentiment_shift_simple to stress sentiment invariance directly.
SENTIMENT_EMOJI_SWAP = {
    ":)": ":(",
    ":-)": ":-(",
    ":d": ":(",
    "🙂": "🙁",
    "😊": "😟",
    "😄": "😢",
    "😍": "😠",
    "👍": "👎",
    "❤️": "💔",
    ":(": ":)",
    ":-(": ":-)",
    "🙁": "🙂",
    "😟": "😊",
    "😢": "😄",
    "😠": "😍",
    "👎": "👍",
    "💔": "❤️",
}

# Modifier / negation pivots from adversarial benchmark-style perturbations.
# Protecting these helps lexical_mhc_lite avoid semantic inversions caused by
# edits to compositional cues (e.g., "only", "never", "barely").
MODIFIER_PIVOTS = {
    "not",
    "never",
    "no",
    "only",
    "just",
    "very",
    "too",
    "barely",
    "hardly",
    "almost",
    "nearly",
    "mostly",
    "slightly",
}

# mHC-lite discourse pivots: lexical framing studies on credibility detection
# show connectors (e.g., "however", "despite") can strongly modulate claim
# stance. We protect them from synonym swaps to keep lexical augmentation in
# paraphrastic space rather than stance-shifting space.
DISCOURSE_PIVOTS = {
    "however",
    "although",
    "despite",
    "but",
    "yet",
    "nevertheless",
    "nonetheless",
    "whereas",
}

# mHC-lite semantic pivots: quantifiers/comparatives often encode the factual
# scope or direction of a claim (e.g., "many" vs "few", "more" vs "less").
# Adversarial benchmark work on fake-news detection highlights classifier
# brittleness to compositional meaning changes in exactly these operators.
# We therefore keep them protected in lexical synonym perturbation so mHC-lite
# learns paraphrastic invariance without accidental claim-logic flips.
QUANTIFIER_COMPARATIVE_PIVOTS = {
    "many",
    "several",
    "few",
    "most",
    "more",
    "less",
    "least",
    "fewer",
    "majority",
    "minority",
}

# mHC-lite claim-logic pivots: temporal/causal operators frequently determine
# whether a claim is support/contrast/cause framed (e.g., "after" vs "before",
# "because" vs "despite"). Black-box fake-news attack studies show small edits
# on such function words can induce large prediction swings, so we keep them
# protected to preserve proposition structure during lexical perturbation.
TEMPORAL_CAUSAL_PIVOTS = {
    "before",
    "after",
    "during",
    "while",
    "when",
    "because",
    "since",
    "therefore",
    "thus",
    "hence",
}


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _detokenize_simple(tokens: List[str]) -> str:
    s = ""
    for t in tokens:
        if re.match(r"[^\w\s]", t):
            s += t
        else:
            if len(s) and not s.endswith((" ", "\n", "\t", "(", "[", "{")):
                s += " "
            s += t
    return s


def _sentiment_polarity_balance(text: str) -> int:
    """Cheap polarity proxy for augmentation safety checks.

    Positive terms contribute +1, negative terms contribute -1 using the
    project's sentiment swap lexicon. This intentionally lightweight heuristic
    is only used as a guardrail for mHC-lite lexical perturbations.
    """
    tokens = [t.lower() for t in _tokenize_simple(text) if t.isalpha()]
    pos = {"good", "great", "excellent", "positive", "benefit", "success", "safe"}
    neg = {"bad", "terrible", "awful", "negative", "harm", "failure", "dangerous"}
    return sum((1 if t in pos else -1 if t in neg else 0) for t in tokens)


def _looks_like_named_entity(tokens: List[str], idx: int) -> bool:
    """Heuristic named-entity guard for lexical_mhc_lite.

    Attack papers such as BERT-Attack show lexical substitutions often focus on
    highly informative tokens. In fake-news detection, those tokens are often
    named entities (people, organizations, locations), and perturbing them can
    alter factual claims rather than style/lexical form.

    We therefore avoid replacing likely entities. Sentence-start TitleCase words
    are usually grammatical capitalization and are allowed, but uppercase
    acronyms (e.g., "NASA", "FBI") are always protected because they are often
    entity anchors even at sentence start.
    """
    tok = tokens[idx]
    if not tok.isalpha() or not tok[0].isupper() or len(tok) < 3:
        return False

    # mHC-lite entity safety: preserve all-uppercase acronyms regardless of
    # position. Cross-domain adversarial fake-news studies show entity-level
    # shifts can induce brittle shortcuts; protecting acronyms reduces fact drift.
    if tok.isupper():
        return True

    # Sentence-start capitalization is common and usually not entity-specific.
    if idx == 0:
        return False
    prev = tokens[idx - 1]
    if prev in {".", "!", "?", ":", ";", "\n"}:
        return False

    return True


def _is_numeric_token(tok: str) -> bool:
    """Return True for simple number-like tokens (e.g., 42, 3.14, 10,000, 50%)."""
    return bool(re.fullmatch(r"\d+[\d,]*(?:\.\d+)?%?", tok))


def _near_numeric_context(tokens: List[str], idx: int) -> bool:
    """Guard claim anchors around quantities for mHC-lite lexical perturbation.

    Robustness-verification studies on credibility/fake-news classifiers show
    small lexical edits near claim-critical spans can disproportionately change
    decisions. Numbers and their neighboring unit words are frequent anchors in
    factual claims (e.g., "5 million", "12 percent", "2024 election").

    For lexical_mhc_lite we keep this conservative: if a token is adjacent to a
    number-like token, we avoid replacing it to reduce accidental fact drift
    while still learning lexical invariance elsewhere.
    """
    left = idx > 0 and _is_numeric_token(tokens[idx - 1])
    right = idx < len(tokens) - 1 and _is_numeric_token(tokens[idx + 1])
    return left or right


def _morph_compatible(source: str, replacement: str) -> bool:
    """Cheap morphology guard for lexical_mhc_lite synonym swaps.

    CLARE/BAE-style findings emphasize context-aware lexical edits for fluency.
    This repository stays lightweight (WordNet, no MLM), so we approximate
    grammatical compatibility by preserving common inflectional shapes.

    Rules are intentionally conservative: if the source token strongly signals
    an inflection (e.g., *-ing*, *-ed*, plural *-s*), we prefer replacements
    with matching surface form to avoid obvious grammar drift.
    """
    s = source.lower()
    r = replacement.lower()

    # Preserve clear inflectional morphology when present.
    if s.endswith("ing") and not r.endswith("ing"):
        return False
    if s.endswith("ed") and not r.endswith("ed"):
        return False
    if s.endswith("ly") and not r.endswith("ly"):
        return False

    # Simple plural noun guard (skip short tokens to avoid false positives).
    if len(s) > 4 and s.endswith("s") and not s.endswith("ss"):
        if not r.endswith("s"):
            return False

    # Prevent extreme token-length jumps that often look unnatural.
    if len(source) > 0:
        ratio = len(replacement) / len(source)
        if ratio < 0.5 or ratio > 2.0:
            return False

    return True


def _max_wordnet_path_similarity(source: str, replacement: str) -> float:
    """Approximate semantic closeness for mHC-lite lexical substitutions.

    Motivation: robustness papers repeatedly show that non-semantic-preserving
    perturbations add noisy invariance pressure. We keep mHC-lite lightweight by
    using WordNet path similarity as a cheap proxy and discarding very distant
    substitutions.

    Returns a score in [0, 1] when available; defaults to 1.0 if WordNet
    similarity is unavailable so existing behavior remains backward-compatible.
    """
    if wn is None:
        return 1.0

    src_synsets = wn.synsets(source)
    rep_synsets = wn.synsets(replacement)
    if not src_synsets or not rep_synsets:
        return 1.0

    best = 0.0
    for s1 in src_synsets:
        for s2 in rep_synsets:
            sim_fn = getattr(s1, "path_similarity", None)
            if sim_fn is None:
                # Mocked or limited synset object: preserve prior behavior.
                return 1.0
            sim = sim_fn(s2)
            if sim is None:
                continue
            if sim > best:
                best = sim
    return best


def lexical_synonym_perturb(
    text: str,
    budget_ratio: float = 0.08,
    seed: int | None = None,
    protected_words: set[str] | None = None,
) -> str:
    """Lexical perturbation by WordNet synonyms for mHC-lite training.

    Design choices (paper-grounded):
    1) avoid swapping sentiment-bearing lexemes because sentiment cues are a
       known attack surface in fake-news detection and can drift labels;
    2) avoid swapping modifier/negation pivots (e.g., "only", "never") because
       adversarial benchmarks show detectors are brittle to compositional cues;
    3) avoid swapping quantifier/comparative pivots (e.g., "many", "more",
       "less", "majority") because these often encode claim logic/scope;
    4) avoid swapping temporal/causal operators (e.g., "before", "because"),
       which can invert event logic under black-box lexical attacks.

    This keeps lexical_mhc_lite focused on lexical paraphrase invariance, while
    sentiment-specific perturbations remain isolated to sentiment_invariance.

    If NLTK wordnet is unavailable, falls back to no-op.
    """
    if wn is None:
        return text

    rng = random.Random(seed)
    tokens = _tokenize_simple(text)

    # mHC-lite safety rail: do not replace explicitly protected words
    # (e.g., sentiment/modifier/discourse pivots), reducing augmentation-induced
    # label drift and stance flips from lexical framing edits.
    # mHC-lite lexical path: merge caller-provided protected words with
    # built-in pivot lexicons to keep perturbations semantically conservative.
    base_protected = (
        set(SENTIMENT_SWAP.keys())
        | MODIFIER_PIVOTS
        | DISCOURSE_PIVOTS
        | QUANTIFIER_COMPARATIVE_PIVOTS
        | TEMPORAL_CAUSAL_PIVOTS
    )
    protected = set(protected_words) | base_protected if protected_words is not None else base_protected

    word_positions = [
        i
        for i, t in enumerate(tokens)
        if t.isalpha()
        and len(t) > 3
        and t.lower() not in protected
        and not _looks_like_named_entity(tokens, i)
        and not _near_numeric_context(tokens, i)
    ]
    if not word_positions:
        return text

    k = max(1, int(len(word_positions) * budget_ratio))
    cand_pos = word_positions[:]
    rng.shuffle(cand_pos)

    changed = 0
    for pos in cand_pos:
        tok = tokens[pos]
        synsets = wn.synsets(tok)
        lemmas = []
        for s in synsets:
            lemmas.extend([l.name().replace("_", " ") for l in s.lemmas()])
        lemmas = [
            w
            for w in lemmas
            if w.lower() != tok.lower()
            and w.isalpha()
            and w.lower() not in protected
            and _morph_compatible(tok, w)
            # mHC-lite lexical guard: keep only semantically close candidates
            # so consistency regularization is driven by paraphrastic changes.
            and _max_wordnet_path_similarity(tok, w) >= 0.2
        ]
        if not lemmas:
            continue

        # Prefer semantically closest candidates while preserving stochasticity.
        scored = sorted(
            ((w, _max_wordnet_path_similarity(tok, w)) for w in lemmas),
            key=lambda x: x[1],
            reverse=True,
        )
        top_k = [w for w, _ in scored[: min(3, len(scored))]]
        replacement = rng.choice(top_k)
        if tok[0].isupper():
            replacement = replacement.capitalize()
        tokens[pos] = replacement
        changed += 1
        if changed >= k:
            break

    candidate = _detokenize_simple(tokens)

    # mHC-lite safety guard (paper-grounded): lexical paraphrases can still
    # induce sentiment drift, which may break label preservation for
    # consistency training. If polarity sign flips, keep original text.
    before = _sentiment_polarity_balance(text)
    after = _sentiment_polarity_balance(candidate)
    if before != 0 and after != 0 and (before > 0) != (after > 0):
        return text

    return candidate


def style_reframe_simple(text: str) -> str:
    """Lightweight style reframing (SheepDog-style proxy without LLM API).

    Rationale (paper-grounded): fake-news detectors can overuse stylistic cues
    (e.g., all-caps emphasis, exaggerated punctuation, elongated spellings)
    rather than factual consistency. This normalizer removes a small set of
    high-variance style markers while keeping proposition content unchanged.

    Extra robustness tweak: normalize volatile social-media wrappers (URLs,
    @mentions, hashtags, and retweet headers) to placeholders. Domain/style
    robustness studies (e.g., MDFEND/FakeZero-like cross-platform settings)
    suggest source/platform artifacts can become shortcut features.

    2026-03/04 enhancement (paper-driven): strip leading section-label wrappers
    such as "BREAKING:", "OPINION:", "ANALYSIS:" and "FACT CHECK:", and remove
    source/byline prefixes like "Reuters -" / "AP |" / "BBC News:". Stylometric
    disinformation studies highlight these as high-signal style cues that can
    become detector shortcuts across outlets/domains.
    """
    out = text.translate(STYLE_CHAR_NORMALIZATION)

    # Normalize platform wrappers before token-level style edits.
    out = re.sub(r"https?://\S+|www\.\S+", "<url>", out)
    out = re.sub(r"@[A-Za-z0-9_]+", "<user>", out)
    out = re.sub(r"#[A-Za-z0-9_]+", "<hashtag>", out)

    # Remove social repost headers that are style/platform-specific wrappers.
    out = re.sub(r"^\s*RT\s+", "", out, flags=re.IGNORECASE)

    # Normalize outlet-style section labels at the start of the message.
    # Examples: "BREAKING:", "OPINION:", "ANALYSIS:", "FACT CHECK:".
    out = re.sub(
        r"^\s*(breaking|opinion|analysis|exclusive|fact\s*check|live\s*update)\s*:\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )

    # Paper-grounded style cue strip: some robustness papers (Style-News,
    # adversarial style augmentation) show source/byline wrappers can become
    # shortcut features. Remove leading source tags while preserving the claim
    # text, e.g., "Reuters -", "AP |", "BBC News:".
    out = re.sub(
        r"^\s*(?:[A-Za-z]{2,10}(?:\s+[A-Za-z]{2,10}){0,2})\s*(?:\||-|:)\s+",
        "",
        out,
    )

    # Strip markdown emphasis wrappers (e.g., **shocking**, _urgent_, ~~fake~~)
    # often used in style-conversion attacks; keep the inner lexical content.
    out = re.sub(r"(\*\*|__|\*|_|~~)(\S(?:.*?\S)?)\1", r"\2", out)

    # Remove leading bracketed clickbait wrappers such as [BREAKING] or
    # (EXCLUSIVE), which are high-variance style cues.
    out = re.sub(r"^\s*[\[(][A-Za-z\s]{3,30}[\])]\s*", "", out)

    for k, v in CONTRACTIONS.items():
        out = re.sub(re.escape(k), v, out, flags=re.IGNORECASE)

    # Normalize sensational all-caps words (>=4 chars) while preserving common
    # short acronyms that often carry factual meaning.
    acronym_whitelist = {"USA", "UK", "EU", "UN", "NATO", "FBI", "CIA"}

    def _caps_norm(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in acronym_whitelist:
            return token
        return token.lower()

    out = re.sub(r"\b[A-Z]{2,}\b", _caps_norm, out)

    # reduce emphatic punctuation style
    out = re.sub(r"!{2,}", "!", out)
    out = re.sub(r"\?{2,}", "?", out)

    # normalize exaggerated elongated spellings common in sensational rewrites
    # (e.g., "soooo shocking" -> "soo shocking") while preserving readability.
    out = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", out)

    # normalize repeated spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out


def sentiment_shift_simple(text: str, budget_ratio: float = 0.2, seed: int | None = None) -> str:
    """Controlled sentiment swapping (AdSent-style proxy without LLM API).

    Compared with the previous token-only version, this introduces:
    1) phrase-level swaps for stronger sentiment perturbations,
    2) a replacement budget so perturbations stay controlled,
    3) seedable stochastic selection for attack diversity,
    4) lightweight emoji/emoticon swaps for social-text sentiment cues.

    The seed option is useful for reproducible ablations while still supporting
    diverse sentiment attacks across training samples.
    """
    budget_ratio = min(max(budget_ratio, 0.0), 1.0)
    swap_budget = max(1, int(len(_tokenize_simple(text)) * budget_ratio))
    rng = random.Random(seed)

    # Phase 0: emoji/emoticon swaps (paper-grounded by ANN-style findings that
    # sentiment-bearing emotes are salient features in fake-news text streams).
    shifted = text
    emoji_hits = 0
    emoji_items = list(SENTIMENT_EMOJI_SWAP.items())
    rng.shuffle(emoji_items)
    for src, dst in emoji_items:
        if emoji_hits >= swap_budget:
            break

        # Case-insensitive replace for ASCII emoticons; direct replace for emoji.
        if src.isascii():
            pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
            if pattern.search(shifted):
                shifted = pattern.sub(dst, shifted, count=1)
                emoji_hits += 1
                # Prevent flip-flop cycles (e.g., 🙂 -> 🙁 -> 🙂) within one call.
                break
        else:
            if src in shifted:
                shifted = shifted.replace(src, dst, 1)
                emoji_hits += 1
                # Prevent inverse map from immediately undoing the same edit.
                break

    # Phase 1: phrase-level swaps (case-insensitive, bounded by remaining budget)
    phrase_hits = 0
    phrase_items = list(SENTIMENT_PHRASE_SWAP.items())
    rng.shuffle(phrase_items)
    for src, dst in phrase_items:
        if emoji_hits + phrase_hits >= swap_budget:
            break
        pattern = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
        if pattern.search(shifted):
            shifted = pattern.sub(dst, shifted, count=1)
            phrase_hits += 1

    # Phase 2: token-level swaps consume remaining budget
    remaining_budget = max(swap_budget - emoji_hits - phrase_hits, 0)
    # Avoid polarity flip-flop: when phrase swaps already fired, stop here.
    # This keeps controlled attacks directional and easier to interpret.
    if remaining_budget == 0 or phrase_hits > 0:
        return shifted

    tokens = _tokenize_simple(shifted)
    out = tokens[:]

    # Merge polarity + tone lexicons for broader sentiment-only perturbations.
    # This keeps attacks in sentiment space (not factual entity edits), aligned
    # with sentiment_invariance objectives.
    sentiment_lexicon = {**SENTIMENT_SWAP, **SENTIMENT_TONE_SWAP}
    candidate_positions = [
        i for i, t in enumerate(tokens) if t.lower() in sentiment_lexicon
    ]
    rng.shuffle(candidate_positions)

    for pos in candidate_positions[:remaining_budget]:
        t = tokens[pos]
        low = t.lower()
        r = sentiment_lexicon[low]
        if t and t[0].isupper():
            r = r.capitalize()
        out[pos] = r

    return _detokenize_simple(out)
