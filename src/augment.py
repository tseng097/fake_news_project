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


def lexical_synonym_perturb(
    text: str,
    budget_ratio: float = 0.08,
    seed: int | None = None,
    protected_words: set[str] | None = None,
) -> str:
    """Lexical perturbation by WordNet synonyms for mHC-lite training.

    Design choice (paper-grounded): we avoid swapping sentiment-bearing lexemes
    by default, because sentiment cues are a known attack surface in fake-news
    detection and replacing them can unintentionally drift label semantics.

    This keeps lexical_mhc_lite focused on lexical paraphrase invariance, while
    sentiment-specific perturbations remain isolated to sentiment_invariance.

    If NLTK wordnet is unavailable, falls back to no-op.
    """
    if wn is None:
        return text

    rng = random.Random(seed)
    tokens = _tokenize_simple(text)

    # mHC-lite safety rail: do not replace explicitly protected words
    # (e.g., sentiment pivots), reducing augmentation-induced label drift.
    protected = protected_words if protected_words is not None else set(SENTIMENT_SWAP.keys())

    word_positions = [
        i
        for i, t in enumerate(tokens)
        if t.isalpha() and len(t) > 3 and t.lower() not in protected
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
        lemmas = [w for w in lemmas if w.lower() != tok.lower() and w.isalpha() and w.lower() not in protected]
        if not lemmas:
            continue
        replacement = rng.choice(lemmas)
        if tok[0].isupper():
            replacement = replacement.capitalize()
        tokens[pos] = replacement
        changed += 1
        if changed >= k:
            break

    return _detokenize_simple(tokens)


def style_reframe_simple(text: str) -> str:
    """Lightweight style reframing (SheepDog-style proxy without LLM API)."""
    out = text
    for k, v in CONTRACTIONS.items():
        out = out.replace(k, v)
    # reduce emphatic punctuation style
    out = re.sub(r"!{2,}", "!", out)
    out = re.sub(r"\?{2,}", "?", out)
    # normalize repeated spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out


def sentiment_shift_simple(text: str, budget_ratio: float = 0.2) -> str:
    """Controlled sentiment swapping (AdSent-style proxy without LLM API).

    Compared with the previous token-only version, this introduces:
    1) phrase-level swaps for stronger sentiment perturbations,
    2) a replacement budget so perturbations stay controlled.

    This better matches "controlled sentiment attacks" used in recent
    fake-news robustness work while keeping implementation lightweight.
    """
    budget_ratio = min(max(budget_ratio, 0.0), 1.0)
    swap_budget = max(1, int(len(_tokenize_simple(text)) * budget_ratio))

    # Phase 1: phrase-level swaps (case-insensitive, bounded by budget)
    phrase_hits = 0
    shifted = text
    for src, dst in SENTIMENT_PHRASE_SWAP.items():
        if phrase_hits >= swap_budget:
            break
        pattern = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
        if pattern.search(shifted):
            shifted = pattern.sub(dst, shifted, count=1)
            phrase_hits += 1

    # Phase 2: token-level swaps consume remaining budget
    remaining_budget = max(swap_budget - phrase_hits, 0)
    # Avoid polarity flip-flop: when phrase swaps already fired, stop here.
    # This keeps controlled attacks directional and easier to interpret.
    if remaining_budget == 0 or phrase_hits > 0:
        return shifted

    tokens = _tokenize_simple(shifted)
    out = []
    token_hits = 0
    for t in tokens:
        low = t.lower()
        if token_hits < remaining_budget and low in SENTIMENT_SWAP:
            r = SENTIMENT_SWAP[low]
            if t and t[0].isupper():
                r = r.capitalize()
            out.append(r)
            token_hits += 1
        else:
            out.append(t)
    return _detokenize_simple(out)
