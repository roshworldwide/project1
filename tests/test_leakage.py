"""Tests for holdout.leakage: contamination, duplicates, and the ledger."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.leakage import (
    HoldoutLedger,
    check_contamination,
    check_contamination_embeddings,
    find_near_duplicates,
)
from holdout.leakage.ngram import containment, jaccard, normalize, tokens, word_ngrams
from holdout.providers.ollama import Ollama
from holdout.scorers import ExactMatch, RegexMatch
from holdout.testing import assert_no_leakage


def make_eval(cases: list[Case]) -> Eval:
    return Eval("leak-test", cases, [ExactMatch()])


def make_eval_noref(cases: list[Case]) -> Eval:
    """Eval for reference-less cases (RegexMatch needs no reference)."""
    return Eval("leak-test", cases, [RegexMatch("x")])


# ---------------------------------------------------------------------------
# n-gram primitives
# ---------------------------------------------------------------------------


def test_normalize_casefolds_and_collapses_whitespace() -> None:
    assert normalize("  Hello   WORLD \n") == "hello world"


def test_tokens_strip_punctuation() -> None:
    assert tokens("Reset, please—now!") == ["reset", "please", "now"]


def test_word_ngrams_basics() -> None:
    grams = word_ngrams("a b c d", 3)
    assert grams == {("a", "b", "c"), ("b", "c", "d")}
    assert word_ngrams("a b", 5) == {("a", "b")}  # short text: one full-token gram
    assert word_ngrams("", 3) == set()
    with pytest.raises(ValueError, match="n must be >= 1"):
        word_ngrams("a b", 0)


def test_containment_and_jaccard() -> None:
    a = word_ngrams("the cat sat on the mat", 3)
    assert containment(a, a) == 1.0
    assert containment(set(), a) == 0.0
    assert jaccard(a, a) == 1.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard(a, set()) == 0.0


# ---------------------------------------------------------------------------
# contamination (n-gram)
# ---------------------------------------------------------------------------

PROMPT = (
    "You are a support assistant. Example: customers often ask how do I reset "
    "my password on the mobile app and the correct answer is go to settings "
    "then security then reset. Always answer politely."
)


def test_clean_eval_reports_clean() -> None:
    ev = make_eval(
        [
            Case(input="What is the refund window for annual plans?", reference="30 days"),
            Case(input="Which regions have data residency?", reference="EU and US"),
        ]
    )
    report = check_contamination(ev, PROMPT)
    assert report.clean
    assert report.n_cases == 2
    assert "no contamination detected" in report.summary()


def test_exact_substring_is_caught() -> None:
    leaked = Case(input="How do I reset my password on the mobile app", reference="42", id="leak1")
    report = check_contamination(make_eval([leaked]), PROMPT)
    assert not report.clean
    (finding,) = report.findings
    assert finding.case_id == "leak1"
    assert finding.field == "input"
    assert finding.kind == "exact-substring"
    assert finding.score == 1.0


def test_reference_leakage_is_caught() -> None:
    leaked = Case(
        input="Where do I change my password?",
        reference="go to settings then security then reset",
        id="leak2",
    )
    report = check_contamination(make_eval([leaked]), PROMPT)
    assert [f.field for f in report.findings] == ["reference"]


def test_ngram_overlap_catches_near_verbatim() -> None:
    # Same sentence as the prompt with the tail changed: not an exact
    # substring, but most 5-grams survive.
    near = Case(
        input="customers often ask how do I reset my password on the desktop site",
        reference="x",
        id="near1",
    )
    report = check_contamination(make_eval([near]), PROMPT, threshold=0.5)
    assert not report.clean
    (finding,) = report.findings
    assert finding.kind == "ngram-overlap"
    assert 0.5 <= finding.score < 1.0
    assert "5-grams" in finding.detail


def test_short_fields_skip_ngram_but_not_substring() -> None:
    short_clean = Case(input="Reset how?", reference="security reset", id="s1")
    report = check_contamination(make_eval([short_clean]), PROMPT)
    assert report.clean  # too short for n-grams, not a substring

    short_leaked = Case(input="Reset how?", reference="then security then reset", id="s2")
    report = check_contamination(make_eval([short_leaked]), PROMPT)
    assert [f.kind for f in report.findings] == ["exact-substring"]


def test_corpus_list_and_index_in_detail() -> None:
    leaked = Case(input="alpha beta gamma delta epsilon zeta", reference="x", id="c1")
    report = check_contamination(
        make_eval([leaked]), ["nothing here", "alpha beta gamma delta epsilon zeta and more"]
    )
    (finding,) = report.findings
    assert "#1" in finding.detail


def test_contaminated_case_ids_dedupe_and_order() -> None:
    both_leaked = Case(
        input="how do I reset my password on the mobile app",
        reference="go to settings then security then reset",
        id="b1",
    )
    report = check_contamination(make_eval([both_leaked]), PROMPT)
    assert len(report.findings) == 2
    assert report.contaminated_case_ids == ("b1",)


def test_threshold_validation_and_to_dict() -> None:
    ev = make_eval([Case(input="hello world example", reference="x")])
    with pytest.raises(ValueError, match="threshold"):
        check_contamination(ev, PROMPT, threshold=0.0)
    payload = json.loads(json.dumps(check_contamination(ev, PROMPT).to_dict()))
    assert payload["clean"] is True


# ---------------------------------------------------------------------------
# contamination (embeddings)
# ---------------------------------------------------------------------------


class FakeBackend:
    """Maps known texts to fixed vectors; unknown texts to an orthogonal one."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    @property
    def name(self) -> str:
        return "fake:embed"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._table.get(t, [0.0, 0.0, 1.0]) for t in texts]


async def test_embedding_contamination_flags_paraphrase() -> None:
    paraphrase = Case(input="password reset walkthrough", reference="ref", id="p1")
    clean = Case(input="data residency question", reference="ref2", id="p2")
    backend = FakeBackend(
        {
            "password reset walkthrough": [1.0, 0.0, 0.0],
            "how to reset a password": [0.99, 0.14, 0.0],
            "data residency question": [0.0, 1.0, 0.0],
        }
    )
    report = await check_contamination_embeddings(
        make_eval([paraphrase, clean]), "how to reset a password", backend, threshold=0.9
    )
    assert [f.case_id for f in report.findings] == ["p1"]
    assert report.findings[0].kind == "embedding-similarity"
    assert "fake:embed" in report.findings[0].detail


async def test_embedding_contamination_validation() -> None:
    ev = make_eval([Case(input="x y z", reference="r")])
    backend = FakeBackend({})
    with pytest.raises(ValueError, match="threshold"):
        await check_contamination_embeddings(ev, "c", backend, threshold=1.5)
    with pytest.raises(ValueError, match="corpus is empty"):
        await check_contamination_embeddings(ev, [], backend)


# ---------------------------------------------------------------------------
# near-duplicates
# ---------------------------------------------------------------------------


def test_identical_inputs_with_distinct_ids_score_one() -> None:
    ev = make_eval(
        [
            Case(input="What is the refund window?", reference="a", id="d1"),
            Case(input="what is THE refund   window?", reference="b", id="d2"),
        ]
    )
    (pair,) = find_near_duplicates(ev)
    assert (pair.case_a, pair.case_b, pair.similarity) == ("d1", "d2", 1.0)


def test_near_duplicates_sorted_and_thresholded() -> None:
    ev = make_eval_noref(
        [
            Case(input="please summarize the quarterly revenue report for europe", id="n1"),
            Case(input="please summarize the quarterly revenue report for asia", id="n2"),
            Case(input="translate this sentence into french", id="n3"),
        ]
    )
    pairs = find_near_duplicates(ev, threshold=0.5)
    assert [(p.case_a, p.case_b) for p in pairs] == [("n1", "n2")]
    assert 0.5 <= pairs[0].similarity < 1.0
    assert find_near_duplicates(ev, threshold=0.99) == []
    with pytest.raises(ValueError, match="threshold"):
        find_near_duplicates(ev, threshold=0.0)


# ---------------------------------------------------------------------------
# holdout ledger
# ---------------------------------------------------------------------------


def test_ledger_counts_and_persists(tmp_path: Path) -> None:
    ledger = HoldoutLedger(tmp_path)
    assert ledger.uses("fp1") == 0
    assert ledger.record_use("fp1", "qa", context="PR #12") == 1
    assert ledger.record_use("fp1", "qa") == 2
    assert ledger.record_use("fp2", "other") == 1  # fingerprints are isolated
    # Persists across instances on the same root.
    assert HoldoutLedger(tmp_path).uses("fp1") == 2


def test_ledger_levels(tmp_path: Path) -> None:
    ledger = HoldoutLedger(tmp_path)
    assert ledger.check("fp", "qa", budget=4).level == "ok"
    ledger.record_use("fp", "qa")
    assert ledger.check("fp", "qa", budget=4).level == "ok"
    ledger.record_use("fp", "qa")
    report = ledger.check("fp", "qa", budget=4)
    assert report.level == "caution"
    assert "fresh holdout" in str(report)
    ledger.record_use("fp", "qa")
    ledger.record_use("fp", "qa")
    report = ledger.check("fp", "qa", budget=4)
    assert report.level == "overfit-risk"
    assert "Dwork" in str(report)
    assert json.loads(json.dumps(report.to_dict()))["uses"] == 4
    with pytest.raises(ValueError, match="budget"):
        ledger.check("fp", "qa", budget=0)


# ---------------------------------------------------------------------------
# assert_no_leakage
# ---------------------------------------------------------------------------


def test_assert_no_leakage_passes_on_clean_eval() -> None:
    ev = make_eval(
        [
            Case(input="What is the refund window for annual plans?", reference="30 days"),
            Case(input="Which regions have data residency?", reference="EU and US"),
        ]
    )
    assert_no_leakage(ev, PROMPT)


def test_assert_no_leakage_raises_on_contamination() -> None:
    ev = make_eval(
        [Case(input="how do I reset my password on the mobile app", reference="x", id="l1")]
    )
    with pytest.raises(AssertionError, match="eval leakage detected"):
        assert_no_leakage(ev, PROMPT)


def test_assert_no_leakage_raises_on_duplicates_and_can_skip() -> None:
    ev = make_eval_noref(
        [
            Case(input="summarize the quarterly revenue report for europe", id="a"),
            Case(input="summarize the quarterly revenue report for asia", id="b"),
        ]
    )
    # This pair shares 4 of 6 distinct 3-grams (Jaccard 0.67): flagged at
    # 0.6, ignored at the stricter default of 0.8, skippable with None.
    with pytest.raises(AssertionError, match="inflate the effective sample size"):
        assert_no_leakage(ev, "an unrelated system prompt", duplicate_threshold=0.6)
    assert_no_leakage(ev, "an unrelated system prompt")
    assert_no_leakage(ev, "an unrelated system prompt", duplicate_threshold=None)


def test_assert_no_leakage_extracts_target_system_prompt() -> None:
    ev = make_eval(
        [Case(input="how do I reset my password on the mobile app", reference="x", id="t1")]
    )
    target = Ollama("llama3.2", system=PROMPT)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(ev, target)

    promptless = Ollama("llama3.2")
    with pytest.raises(ValueError, match="exposes no system prompt"):
        assert_no_leakage(ev, promptless)
