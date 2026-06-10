"""Tests for holdout.scorers: ExactMatch, RegexMatch, cosine_similarity, EmbeddingSimilarity."""

from collections.abc import Sequence

import pytest

from holdout import Case, Eval
from holdout.scorers import (
    EmbeddingBackend,
    EmbeddingSimilarity,
    ExactMatch,
    RegexMatch,
    cosine_similarity,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_case(reference: str | None = "expected") -> Case:
    """A minimal case with a controllable reference."""
    return Case(input="prompt", reference=reference)


class FakeBackend:
    """Local embedding backend returning fixed vectors per text."""

    def __init__(self, vectors: dict[str, list[float]], name: str = "fake:test") -> None:
        self._vectors = vectors
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vectors[t]) for t in texts]


# ---------------------------------------------------------------------------
# ExactMatch
# ---------------------------------------------------------------------------


async def test_exact_match_normalized_strips_casefolds_and_collapses_whitespace() -> None:
    scorer = ExactMatch(normalize=True)
    score = await scorer.score(Case(input="q", reference="  Hello   World "), "hello world")
    assert score.value == 1.0
    assert score.kind == "binary"


async def test_exact_match_normalized_handles_tabs_and_newlines() -> None:
    scorer = ExactMatch()  # normalize defaults to True
    score = await scorer.score(Case(input="q", reference="A\tB\nC"), "  a b   c ")
    assert score.value == 1.0


async def test_exact_match_normalized_fails_on_content_difference() -> None:
    scorer = ExactMatch(normalize=True)
    score = await scorer.score(Case(input="q", reference="hello world"), "hello there")
    assert score.value == 0.0
    assert score.kind == "binary"


async def test_exact_match_strict_requires_identical_text() -> None:
    scorer = ExactMatch(normalize=False)
    case = Case(input="q", reference="Hello World")
    assert (await scorer.score(case, "Hello World")).value == 1.0
    assert (await scorer.score(case, "hello world")).value == 0.0
    assert (await scorer.score(case, "Hello World ")).value == 0.0
    assert (await scorer.score(case, "Hello  World")).value == 0.0


async def test_exact_match_scores_are_binary_kind() -> None:
    scorer = ExactMatch(normalize=False)
    case = Case(input="q", reference="x")
    hit = await scorer.score(case, "x")
    miss = await scorer.score(case, "y")
    assert hit.kind == "binary"
    assert miss.kind == "binary"
    assert {hit.value, miss.value} == {1.0, 0.0}


def test_exact_match_config_reflects_normalize() -> None:
    assert dict(ExactMatch(normalize=True).config()) == {"normalize": True}
    assert dict(ExactMatch(normalize=False).config()) == {"normalize": False}


def test_exact_match_fingerprint_differs_by_normalize() -> None:
    assert ExactMatch(normalize=True).fingerprint != ExactMatch(normalize=False).fingerprint
    assert ExactMatch(normalize=True).fingerprint == ExactMatch(normalize=True).fingerprint


# ---------------------------------------------------------------------------
# RegexMatch
# ---------------------------------------------------------------------------


async def test_regex_match_searches_anywhere() -> None:
    scorer = RegexMatch(r"\d{3}")
    assert (await scorer.score(make_case(), "order id is 123, thanks")).value == 1.0
    assert (await scorer.score(make_case(), "no digits here")).value == 0.0


async def test_regex_match_anchored_full_match() -> None:
    scorer = RegexMatch(r"\Aabc\Z")
    assert (await scorer.score(make_case(), "abc")).value == 1.0
    assert (await scorer.score(make_case(), "xabc")).value == 0.0
    assert (await scorer.score(make_case(), "abc!")).value == 0.0


async def test_regex_match_ignore_case_flag() -> None:
    sensitive = RegexMatch("HELLO")
    insensitive = RegexMatch("HELLO", ignore_case=True)
    assert (await sensitive.score(make_case(), "say hello")).value == 0.0
    assert (await insensitive.score(make_case(), "say hello")).value == 1.0


async def test_regex_match_needs_no_reference() -> None:
    scorer = RegexMatch("ok")
    assert RegexMatch.requires_reference is False
    score = await scorer.score(Case(input="q"), "ok then")  # reference-less case
    assert score.value == 1.0
    # An Eval with reference-less cases accepts a regex scorer.
    ev = Eval(name="regex-only", cases=[Case(input="q")], scorers=[RegexMatch("ok")])
    assert len(ev) == 1


async def test_regex_match_detail_includes_pattern() -> None:
    scorer = RegexMatch(r"\d+")
    score = await scorer.score(make_case(), "42")
    assert score.detail is not None
    assert r"\d+" in score.detail


async def test_regex_match_scores_are_binary_kind() -> None:
    scorer = RegexMatch("yes")
    hit = await scorer.score(make_case(), "yes")
    miss = await scorer.score(make_case(), "no")
    assert hit.kind == "binary"
    assert miss.kind == "binary"
    assert hit.value == 1.0
    assert miss.value == 0.0


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors() -> None:
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# EmbeddingSimilarity
# ---------------------------------------------------------------------------


async def test_embedding_continuous_mode_returns_raw_cosine() -> None:
    backend = FakeBackend({"out": [1.0, 0.0], "ref": [1.0, 1.0]})
    scorer = EmbeddingSimilarity(backend)
    score = await scorer.score(Case(input="q", reference="ref"), "out")
    assert score.kind == "continuous"
    assert score.value == pytest.approx(2.0**0.5 / 2.0)
    assert score.detail is not None
    assert "fake:test" in score.detail


async def test_embedding_continuous_identical_texts_score_one() -> None:
    backend = FakeBackend({"same": [3.0, 4.0]})
    scorer = EmbeddingSimilarity(backend)
    score = await scorer.score(Case(input="q", reference="same"), "same")
    assert score.value == pytest.approx(1.0)
    assert score.kind == "continuous"


async def test_embedding_threshold_passes_at_exactly_threshold() -> None:
    # Orthogonal vectors: cosine is exactly 0.0, and 0.0 >= 0.0 must pass.
    backend = FakeBackend({"out": [1.0, 0.0], "ref": [0.0, 1.0]})
    scorer = EmbeddingSimilarity(backend, threshold=0.0)
    score = await scorer.score(Case(input="q", reference="ref"), "out")
    assert score.value == 1.0
    assert score.kind == "binary"


async def test_embedding_threshold_fails_below_threshold() -> None:
    backend = FakeBackend({"out": [1.0, 0.0], "ref": [-1.0, 0.0]})  # cosine -1.0
    scorer = EmbeddingSimilarity(backend, threshold=0.0)
    score = await scorer.score(Case(input="q", reference="ref"), "out")
    assert score.value == 0.0
    assert score.kind == "binary"


async def test_embedding_threshold_passes_above_threshold() -> None:
    backend = FakeBackend({"out": [2.0, 0.0], "ref": [5.0, 0.0]})  # cosine 1.0
    scorer = EmbeddingSimilarity(backend, threshold=0.5)
    score = await scorer.score(Case(input="q", reference="ref"), "out")
    assert score.value == 1.0


async def test_embedding_threshold_detail_includes_cosine_and_threshold() -> None:
    backend = FakeBackend({"out": [1.0, 0.0], "ref": [0.0, 1.0]})
    scorer = EmbeddingSimilarity(backend, threshold=0.25)
    score = await scorer.score(Case(input="q", reference="ref"), "out")
    assert score.detail is not None
    assert "cosine=0.0000" in score.detail
    assert "threshold=0.25" in score.detail


def test_embedding_threshold_out_of_range_raises() -> None:
    backend = FakeBackend({})
    with pytest.raises(ValueError, match="threshold"):
        EmbeddingSimilarity(backend, threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        EmbeddingSimilarity(backend, threshold=-1.01)


def test_embedding_config_includes_backend_and_threshold() -> None:
    backend = FakeBackend({}, name="fake:alpha")
    scorer = EmbeddingSimilarity(backend, threshold=0.8)
    assert dict(scorer.config()) == {"backend": "fake:alpha", "threshold": 0.8}
    assert dict(EmbeddingSimilarity(backend).config()) == {
        "backend": "fake:alpha",
        "threshold": None,
    }


def test_embedding_fingerprint_differs_by_backend_and_threshold() -> None:
    alpha = FakeBackend({}, name="fake:alpha")
    beta = FakeBackend({}, name="fake:beta")
    assert EmbeddingSimilarity(alpha).fingerprint != EmbeddingSimilarity(beta).fingerprint
    assert (
        EmbeddingSimilarity(alpha, threshold=0.7).fingerprint
        != EmbeddingSimilarity(alpha, threshold=0.9).fingerprint
    )
    assert EmbeddingSimilarity(alpha).fingerprint == EmbeddingSimilarity(alpha).fingerprint


def test_embedding_requires_reference() -> None:
    backend = FakeBackend({})
    assert EmbeddingSimilarity.requires_reference is True
    with pytest.raises(ValueError, match="requires a reference"):
        Eval(
            name="needs-refs",
            cases=[Case(input="q")],  # no reference
            scorers=[EmbeddingSimilarity(backend)],
        )


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(FakeBackend({}), EmbeddingBackend)
