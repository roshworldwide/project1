"""Near-duplicate detection within an eval.

Near-copies in a test set quietly inflate the effective sample size: 200
cases that are 150 unique problems do not buy the confidence interval of
n=200. Statistical tests assume (conditionally) independent cases; this
check surfaces the pairs that break the assumption.
"""

from dataclasses import dataclass

from holdout.core.evalset import Eval
from holdout.leakage.ngram import jaccard, normalize, word_ngrams


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """Two cases that look like the same problem.

    Parameters
    ----------
    case_a, case_b
        The two case ids.
    similarity
        Jaccard similarity of their input n-gram sets (1.0 for identical
        normalized inputs).
    """

    case_a: str
    case_b: str
    similarity: float

    def __str__(self) -> str:
        return f"{self.case_a} ~ {self.case_b} (similarity={self.similarity:.3f})"


def find_near_duplicates(
    ev: Eval, *, ngram_size: int = 3, threshold: float = 0.8
) -> list[DuplicatePair]:
    """Find case pairs whose inputs are near-duplicates.

    Compares every pair of case inputs by word ``ngram_size``-gram Jaccard
    similarity (plus an exact check on normalized text) and returns pairs
    at or above ``threshold``, most similar first. O(n^2) pairs — fine for
    eval-sized sets (thousands of cases).

    Parameters
    ----------
    ev
        The eval to audit.
    ngram_size
        Word n-gram size (default 3 — inputs are often short).
    threshold
        Jaccard similarity at or above which a pair is reported.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    cases = ev.cases
    norms = [normalize(c.input) for c in cases]
    grams = [word_ngrams(c.input, ngram_size) for c in cases]

    pairs: list[DuplicatePair] = []
    for i in range(len(cases)):
        id_i = cases[i].id
        assert id_i is not None  # Eval normalization guarantees ids
        for j in range(i + 1, len(cases)):
            id_j = cases[j].id
            assert id_j is not None
            if norms[i] and norms[i] == norms[j]:
                pairs.append(DuplicatePair(id_i, id_j, 1.0))
                continue
            sim = jaccard(grams[i], grams[j])
            if sim >= threshold:
                pairs.append(DuplicatePair(id_i, id_j, sim))
    pairs.sort(key=lambda p: -p.similarity)
    return pairs
