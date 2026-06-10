"""Unit tests for the core models: hashing, Case, Eval, Score, and Scorer."""

import json
from pathlib import Path

import pytest

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.hashing import canonical_json, fingerprint, short_id
from holdout.core.scoring import Score, Scorer
from holdout.scorers.exact import ExactMatch


class NullScorer(Scorer):
    """Minimal scorer used to vary scorer sets without changing case data."""

    @property
    def name(self) -> str:
        return "null"

    async def score(self, case: Case, output: str) -> Score:
        return Score(value=1.0, kind="binary")


def make_cases() -> list[Case]:
    return [Case(input="q1", reference="a1"), Case(input="q2", reference="a2")]


def make_eval(name: str = "smoke") -> Eval:
    return Eval(name=name, cases=make_cases(), scorers=[ExactMatch()])


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def test_canonical_json_sorts_keys_and_strips_whitespace() -> None:
    out = canonical_json({"b": 1, "a": {"d": 2, "c": 3}})
    assert out == '{"a":{"c":3,"d":2},"b":1}'
    assert " " not in out
    assert "\n" not in out


def test_canonical_json_key_order_invariant() -> None:
    assert canonical_json({"x": 1, "y": 2}) == canonical_json({"y": 2, "x": 1})


def test_fingerprint_stable_and_content_sensitive() -> None:
    a = fingerprint({"k": "v", "n": 1})
    b = fingerprint({"n": 1, "k": "v"})
    assert a == b
    assert len(a) == 64
    assert all(ch in "0123456789abcdef" for ch in a)
    assert fingerprint({"k": "v", "n": 2}) != a


def test_short_id_prefix() -> None:
    full = fingerprint({"k": "v"})
    assert short_id(full) == full[:12]
    assert short_id(full, length=8) == full[:8]


# ---------------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------------


def test_case_content_id_stable_across_identical_constructions() -> None:
    a = Case(input="q", reference="a", metadata={"k": "v"})
    b = Case(input="q", reference="a", metadata={"k": "v"})
    assert a.content_id() == b.content_id()


def test_case_content_id_changes_with_content() -> None:
    base = Case(input="q", reference="a", metadata={"k": "v"})
    assert Case(input="Q", reference="a", metadata={"k": "v"}).content_id() != base.content_id()
    assert Case(input="q", reference="b", metadata={"k": "v"}).content_id() != base.content_id()
    assert Case(input="q", reference=None, metadata={"k": "v"}).content_id() != base.content_id()
    assert Case(input="q", reference="a", metadata={"k": "w"}).content_id() != base.content_id()


def test_case_content_id_shape() -> None:
    cid = Case(input="q").content_id()
    assert cid.startswith("c")
    assert len(cid) == 12


def test_case_to_dict() -> None:
    case = Case(input="q", reference="a", id="x1", metadata={"k": "v"})
    assert case.to_dict() == {"id": "x1", "input": "q", "reference": "a", "metadata": {"k": "v"}}


def test_case_to_dict_defaults() -> None:
    assert Case(input="q").to_dict() == {
        "id": None,
        "input": "q",
        "reference": None,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Eval construction and validation
# ---------------------------------------------------------------------------


def test_eval_auto_assigns_content_ids() -> None:
    cases = make_cases()
    ev = Eval(name="e", cases=cases, scorers=[ExactMatch()])
    assert all(c.id is not None for c in ev.cases)
    assert [c.id for c in ev.cases] == [c.content_id() for c in cases]


def test_eval_preserves_explicit_ids() -> None:
    cases = [Case(input="q1", reference="a1", id="one"), Case(input="q2", reference="a2")]
    ev = Eval(name="e", cases=cases, scorers=[ExactMatch()])
    assert ev.cases[0].id == "one"
    assert ev.cases[1].id == cases[1].content_id()


def test_eval_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name must be non-empty"):
        Eval(name="", cases=make_cases(), scorers=[ExactMatch()])


def test_eval_rejects_no_cases() -> None:
    with pytest.raises(ValueError, match="has no cases"):
        Eval(name="e", cases=[], scorers=[ExactMatch()])


def test_eval_rejects_no_scorers() -> None:
    with pytest.raises(ValueError, match="has no scorers"):
        Eval(name="e", cases=make_cases(), scorers=[])


def test_eval_rejects_duplicate_identical_cases_with_helpful_message() -> None:
    twin = [Case(input="q", reference="a"), Case(input="q", reference="a")]
    with pytest.raises(ValueError) as excinfo:
        Eval(name="e", cases=twin, scorers=[ExactMatch()])
    message = str(excinfo.value)
    assert "duplicate case ids" in message
    assert twin[0].content_id() in message
    assert "deduplicate them or assign explicit distinct ids" in message


def test_eval_rejects_duplicate_explicit_ids() -> None:
    cases = [Case(input="q1", reference="a1", id="x"), Case(input="q2", reference="a2", id="x")]
    with pytest.raises(ValueError, match="duplicate case ids"):
        Eval(name="e", cases=cases, scorers=[ExactMatch()])


def test_eval_rejects_duplicate_scorer_names() -> None:
    with pytest.raises(ValueError, match="duplicate scorer names"):
        Eval(
            name="e",
            cases=make_cases(),
            scorers=[ExactMatch(normalize=True), ExactMatch(normalize=False)],
        )


def test_eval_requires_reference_message_names_scorer() -> None:
    cases = [Case(input="q1", reference="a1"), Case(input="q2")]
    with pytest.raises(ValueError) as excinfo:
        Eval(name="e", cases=cases, scorers=[ExactMatch()])
    message = str(excinfo.value)
    assert "'exact_match'" in message
    assert "requires a reference" in message


# ---------------------------------------------------------------------------
# Eval fingerprint
# ---------------------------------------------------------------------------


def test_eval_fingerprint_stable_across_constructions() -> None:
    assert make_eval().fingerprint == make_eval().fingerprint


def test_eval_fingerprint_independent_of_scorers() -> None:
    with_exact = Eval(name="e", cases=make_cases(), scorers=[ExactMatch()])
    with_null = Eval(name="e", cases=make_cases(), scorers=[NullScorer()])
    assert with_exact.fingerprint == with_null.fingerprint


def test_eval_fingerprint_changes_when_a_case_changes() -> None:
    base = Eval(name="e", cases=make_cases(), scorers=[ExactMatch()])
    changed_cases = make_cases()
    changed_cases[1] = Case(input="q2", reference="DIFFERENT")
    changed = Eval(name="e", cases=changed_cases, scorers=[ExactMatch()])
    assert base.fingerprint != changed.fingerprint


def test_eval_len_and_repr() -> None:
    ev = make_eval(name="smoke")
    assert len(ev) == 2
    assert repr(ev) == "Eval(name='smoke', cases=2, scorers=['exact_match'])"


# ---------------------------------------------------------------------------
# Eval.from_jsonl
# ---------------------------------------------------------------------------


def test_from_jsonl_happy_path(tmp_path: Path) -> None:
    path = tmp_path / "support_qa.jsonl"
    rows: list[dict[str, object]] = [
        {"input": "q1", "reference": "a1", "id": "one", "metadata": {"k": "v"}},
        {"input": "q2", "reference": "a2"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ev = Eval.from_jsonl(path, scorers=[ExactMatch()])
    assert len(ev) == 2
    first, second = ev.cases
    assert (first.input, first.reference, first.id) == ("q1", "a1", "one")
    assert dict(first.metadata) == {"k": "v"}
    assert (second.input, second.reference) == ("q2", "a2")
    assert second.id == Case(input="q2", reference="a2").content_id()
    assert dict(second.metadata) == {}


def test_from_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blanks.jsonl"
    path.write_text(
        '{"input": "q1", "reference": "a1"}\n\n   \n{"input": "q2", "reference": "a2"}\n',
        encoding="utf-8",
    )
    ev = Eval.from_jsonl(path, scorers=[ExactMatch()])
    assert [c.input for c in ev.cases] == ["q1", "q2"]


def test_from_jsonl_invalid_json_reports_path_and_lineno(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": "q1", "reference": "a1"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        Eval.from_jsonl(path, scorers=[ExactMatch()])
    message = str(excinfo.value)
    assert message.startswith(f"{path}:2:")
    assert "invalid JSON" in message


def test_from_jsonl_missing_input_raises(tmp_path: Path) -> None:
    path = tmp_path / "noinput.jsonl"
    path.write_text('{"reference": "a1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field 'input'") as excinfo:
        Eval.from_jsonl(path, scorers=[ExactMatch()])
    assert str(excinfo.value).startswith(f"{path}:1:")


def test_from_jsonl_default_name_is_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "my_eval.jsonl"
    path.write_text('{"input": "q1", "reference": "a1"}\n', encoding="utf-8")
    ev = Eval.from_jsonl(path, scorers=[ExactMatch()])
    assert ev.name == "my_eval"


def test_from_jsonl_explicit_name_wins(tmp_path: Path) -> None:
    path = tmp_path / "my_eval.jsonl"
    path.write_text('{"input": "q1", "reference": "a1"}\n', encoding="utf-8")
    ev = Eval.from_jsonl(path, scorers=[ExactMatch()], name="custom")
    assert ev.name == "custom"


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def test_binary_score_accepts_zero_and_one() -> None:
    assert Score(value=1.0, kind="binary").value == 1.0
    assert Score(value=0.0, kind="binary").value == 0.0


def test_binary_score_rejects_fractional_value() -> None:
    with pytest.raises(ValueError, match=r"binary scores must be 0\.0 or 1\.0"):
        Score(value=0.5, kind="binary")


def test_continuous_score_accepts_fractional_value() -> None:
    assert Score(value=0.5, kind="continuous").value == 0.5


def test_score_to_dict() -> None:
    score = Score(value=0.5, kind="continuous", detail="cosine=0.5")
    assert score.to_dict() == {"value": 0.5, "kind": "continuous", "detail": "cosine=0.5"}
    assert Score(value=1.0, kind="binary").to_dict() == {
        "value": 1.0,
        "kind": "binary",
        "detail": None,
    }


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def test_scorer_fingerprint_sensitive_to_config() -> None:
    normalized = ExactMatch(normalize=True)
    raw = ExactMatch(normalize=False)
    assert normalized.fingerprint != raw.fingerprint
    assert normalized.fingerprint == ExactMatch(normalize=True).fingerprint


def test_scorer_repr() -> None:
    assert repr(ExactMatch(normalize=True)) == "ExactMatch({'normalize': True})"
    assert repr(ExactMatch(normalize=False)) == "ExactMatch({'normalize': False})"
