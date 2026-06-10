"""Embedding-similarity scoring."""

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from holdout.core.case import Case
from holdout.core.scoring import Score, Scorer


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that can embed a batch of texts.

    Implementations live in :mod:`holdout.providers.embeddings`
    (Ollama for local/air-gapped use, OpenAI behind an optional extra).
    """

    @property
    def name(self) -> str:
        """Identifier for fingerprinting (e.g. ``"ollama:nomic-embed-text"``)."""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts``, returning one vector per text, in order."""
        ...


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute the cosine similarity of two vectors.

    Returns 0.0 if either vector has zero norm (no direction, no similarity).
    """
    if len(a) != len(b):
        raise ValueError(f"vector dimensions differ: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingSimilarity(Scorer):
    """Scores semantic similarity between the output and the reference.

    Without a threshold, reports the raw cosine similarity (continuous, in
    [-1, 1]). With a threshold, reports binary pass/fail and surfaces the
    underlying similarity in the score's detail — so the statistics engine
    can run McNemar on the passes while a human can still see the margins.

    Parameters
    ----------
    backend
        The embedding backend to use.
    threshold
        If set, score 1.0 when similarity >= threshold, else 0.0.
    """

    requires_reference = True

    def __init__(self, backend: EmbeddingBackend, *, threshold: float | None = None) -> None:
        if threshold is not None and not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
        self._backend = backend
        self._threshold = threshold

    @property
    def name(self) -> str:
        """Metric name: ``embedding_similarity``."""
        return "embedding_similarity"

    def config(self) -> Mapping[str, object]:
        """Return the scorer's configuration for fingerprinting."""
        return {"backend": self._backend.name, "threshold": self._threshold}

    async def score(self, case: Case, output: str) -> Score:
        """Embed output and reference, return their cosine similarity."""
        assert case.reference is not None  # enforced by Eval validation
        vec_out, vec_ref = await self._backend.embed([output, case.reference])
        sim = cosine_similarity(vec_out, vec_ref)
        if self._threshold is None:
            return Score(value=sim, kind="continuous", detail=f"backend={self._backend.name}")
        return Score(
            value=1.0 if sim >= self._threshold else 0.0,
            kind="binary",
            detail=f"cosine={sim:.4f}, threshold={self._threshold}",
        )
