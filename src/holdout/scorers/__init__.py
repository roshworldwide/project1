"""Built-in scorers and the base class for custom ones.

Custom scorers subclass :class:`holdout.core.scoring.Scorer` and implement
``name`` and ``score``; override ``config`` so your parameters participate
in run fingerprints.
"""

from holdout.core.scoring import Score, Scorer
from holdout.scorers.embedding import EmbeddingBackend, EmbeddingSimilarity, cosine_similarity
from holdout.scorers.exact import ExactMatch
from holdout.scorers.regex_match import RegexMatch

__all__ = [
    "EmbeddingBackend",
    "EmbeddingSimilarity",
    "ExactMatch",
    "RegexMatch",
    "Score",
    "Scorer",
    "cosine_similarity",
]
