"""Word n-gram primitives for contamination and duplicate detection.

N-gram overlap is the standard, embedding-free way to detect text
contamination — the GPT-3 evaluation used 13-gram dedup against training
data (Brown et al. 2020, *Language Models are Few-Shot Learners*, app. C).
Eval prompts are far shorter than training corpora, so holdout defaults to
5-grams and pairs the check with exact-substring matching.
"""

import re

_TOKEN = re.compile(r"\w+")


def normalize(text: str) -> str:
    """Casefold and collapse whitespace — the comparison canon."""
    return " ".join(text.casefold().split())


def tokens(text: str) -> list[str]:
    """Extract word tokens (casefolded, punctuation-free)."""
    return _TOKEN.findall(text.casefold())


def word_ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    """Return the set of word n-grams of ``text``.

    Texts shorter than ``n`` tokens yield their full token tuple as a
    single gram, so short cases still participate in containment checks.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    toks = tokens(text)
    if not toks:
        return set()
    if len(toks) < n:
        return {tuple(toks)}
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def containment(needle: set[tuple[str, ...]], corpus: set[tuple[str, ...]]) -> float:
    """Fraction of ``needle``'s grams found in ``corpus`` (0.0 for empty needle).

    Containment (rather than Jaccard) is the right asymmetric measure for
    "is this small case inside that big prompt": it does not get diluted by
    the corpus's size.
    """
    if not needle:
        return 0.0
    return len(needle & corpus) / len(needle)


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    """Jaccard similarity of two gram sets (0.0 when both are empty)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
