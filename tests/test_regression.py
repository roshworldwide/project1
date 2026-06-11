"""Tests for the regression engine (holdout.regression.compare)."""

import json
from collections.abc import Mapping

import pytest

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.run import Run
from holdout.core.runner import run
from holdout.core.scoring import Score, Scorer
from holdout.providers.static import StaticTarget
from holdout.regression import RunComparison, compare
from holdout.scorers import ExactMatch

N = 40


def make_eval(n: int = N, scorers: list[Scorer] | None = None) -> Eval:
    cases = [Case(input=f"q{i}", reference="yes", id=f"c{i:03d}") for i in range(n)]
    return Eval("reg-test", cases, scorers or [ExactMatch()])


def make_target(wrong: set[int], n: int = N, name: str = "t") -> StaticTarget:
    return StaticTarget({f"q{i}": ("no" if i in wrong else "yes") for i in range(n)}, name=name)


def run_pair(
    wrong_a: set[int], wrong_b: set[int], n: int = N, scorers: list[Scorer] | None = None
) -> tuple[Run, Run]:
    ev = make_eval(n, scorers)
    a = run(ev, target=make_target(wrong_a, n, "baseline"), seed=7)
    b = run(ev, target=make_target(wrong_b, n, "candidate"), seed=7)
    return a, b


class NamedRegex(Scorer):
    """Binary scorer with a configurable name (distinct metrics in one eval)."""

    def __init__(self, metric_name: str, needle: str) -> None:
        self._metric_name = metric_name
        self._needle = needle

    @property
    def name(self) -> str:
        return self._metric_name

    def config(self) -> Mapping[str, object]:
        return {"needle": self._needle}

    async def score(self, case: Case, output: str) -> Score:
        return Score(value=1.0 if self._needle in output else 0.0, kind="binary")


class LengthScore(Scorer):
    """Continuous scorer: output length scaled into [0, 1]."""

    @property
    def name(self) -> str:
        return "length_score"

    async def score(self, case: Case, output: str) -> Score:
        return Score(value=min(len(output) / 10.0, 1.0), kind="continuous")


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_binary_regression_detected_with_mcnemar() -> None:
    a, b = run_pair(set(), set(range(10)))
    cmp = compare(a, b, seed=0)
    (mc,) = cmp.comparisons
    assert cmp.verdict == "regressed"
    assert mc.verdict == "regressed"
    assert mc.result is not None
    assert mc.result.test == "mcnemar-exact"
    assert mc.result.effect == pytest.approx(-0.25)
    assert mc.p_adjusted is not None and mc.p_adjusted <= 0.05
    assert cmp.regressed == (mc,)
    assert not cmp.improved


def test_binary_improvement_detected() -> None:
    a, b = run_pair(set(range(10)), set())
    cmp = compare(a, b, seed=0)
    assert cmp.verdict == "improved"
    assert cmp.comparisons[0].result is not None
    assert cmp.comparisons[0].result.effect == pytest.approx(+0.25)


def test_identical_runs_no_significant_change() -> None:
    a, b = run_pair(set(), set())
    cmp = compare(a, b, seed=0)
    assert cmp.verdict == "no_significant_change"
    assert cmp.comparisons[0].p_adjusted == 1.0


def test_continuous_metric_uses_paired_bootstrap_under_auto() -> None:
    ev = make_eval(scorers=[LengthScore()])
    a = run(
        ev, target=StaticTarget({f"q{i}": "y" * (i % 7 + 1) for i in range(N)}, name="a"), seed=1
    )
    b = run(
        ev, target=StaticTarget({f"q{i}": "y" * (i % 5 + 3) for i in range(N)}, name="b"), seed=1
    )
    cmp = compare(a, b, seed=0)
    assert cmp.comparisons[0].result is not None
    assert cmp.comparisons[0].result.test == "paired-bootstrap"


@pytest.mark.parametrize(
    ("forced", "expected_prefix"),
    [("permutation", "permutation"), ("paired-bootstrap", "paired-bootstrap")],
)
def test_forced_test_is_honored(forced: str, expected_prefix: str) -> None:
    a, b = run_pair(set(), set(range(10)))
    cmp = compare(a, b, test=forced, seed=0)  # type: ignore[arg-type]
    assert cmp.comparisons[0].result is not None
    assert cmp.comparisons[0].result.test.startswith(expected_prefix)


def test_insufficient_data_when_one_paired_case() -> None:
    ev = make_eval(4)
    a = run(ev, target=make_target(set(), 4, "a"), seed=1)
    # Candidate answers only q0; all other cases error (no default).
    b = run(ev, target=StaticTarget({"q0": "yes"}, name="b"), seed=1)
    cmp = compare(a, b, seed=0)
    (mc,) = cmp.comparisons
    assert mc.verdict == "insufficient_data"
    assert mc.n_pairs == 1
    assert mc.note is not None and "at least 2" in mc.note
    assert mc.result is None and mc.p_adjusted is None
    assert cmp.verdict == "insufficient_data"


def test_verdict_precedence_regression_beats_improvement() -> None:
    # candidate: 15 cases switch "yes" -> "yes 1": exact_match regresses,
    # digit-regex improves. Worst news wins.
    scorers: list[Scorer] = [ExactMatch(), NamedRegex("has_digit", "1")]
    ev = make_eval(scorers=scorers)
    a = run(ev, target=StaticTarget({f"q{i}": "yes" for i in range(N)}, name="a"), seed=1)
    b_out = {f"q{i}": ("yes 1" if i < 15 else "yes") for i in range(N)}
    b = run(ev, target=StaticTarget(b_out, name="b"), seed=1)
    cmp = compare(a, b, seed=0)
    by_metric = {c.metric: c.verdict for c in cmp.comparisons}
    assert by_metric == {"exact_match": "regressed", "has_digit": "improved"}
    assert cmp.verdict == "regressed"


def test_verdict_improvement_with_no_change_elsewhere() -> None:
    scorers: list[Scorer] = [NamedRegex("says_yes", "yes"), NamedRegex("has_digit", "1")]
    ev = make_eval(scorers=scorers)
    a = run(ev, target=StaticTarget({f"q{i}": "yes" for i in range(N)}, name="a"), seed=1)
    b_out = {f"q{i}": ("yes 1" if i < 20 else "yes") for i in range(N)}
    b = run(ev, target=StaticTarget(b_out, name="b"), seed=1)
    cmp = compare(a, b, seed=0)
    by_metric = {c.metric: c.verdict for c in cmp.comparisons}
    assert by_metric == {"says_yes": "no_significant_change", "has_digit": "improved"}
    assert cmp.verdict == "improved"


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def _three_metric_runs() -> tuple[Run, Run]:
    # has_a improves on 6 cases (McNemar exact p = 2/64 = 0.03125 < 0.05);
    # the other two metrics never change (p = 1).
    scorers: list[Scorer] = [
        NamedRegex("has_a", "A"),
        NamedRegex("has_b", "B"),
        NamedRegex("has_c", "C"),
    ]
    ev = make_eval(scorers=scorers)
    a = run(ev, target=StaticTarget({f"q{i}": "x" for i in range(N)}, name="a"), seed=1)
    b_out = {f"q{i}": ("A" if i < 6 else "x") for i in range(N)}
    b = run(ev, target=StaticTarget(b_out, name="b"), seed=1)
    return a, b


def test_bh_correction_can_flip_borderline_significance() -> None:
    a, b = _three_metric_runs()
    uncorrected = compare(a, b, correction="none", seed=0)
    corrected = compare(a, b, correction="benjamini-hochberg", seed=0)

    raw = {c.metric: c for c in uncorrected.comparisons}
    adj = {c.metric: c for c in corrected.comparisons}
    assert raw["has_a"].verdict == "improved"  # raw p = 0.03125 <= 0.05
    assert raw["has_a"].p_adjusted == pytest.approx(0.03125)
    # BH over {0.03125, 1, 1}: q = 0.09375 > 0.05 — the fluke is absorbed.
    assert adj["has_a"].p_adjusted == pytest.approx(0.09375)
    assert adj["has_a"].verdict == "no_significant_change"
    assert corrected.verdict == "no_significant_change"

    for metric in ("has_a", "has_b", "has_c"):
        r, c = raw[metric], adj[metric]
        assert r.result is not None and c.p_adjusted is not None
        assert c.p_adjusted >= r.result.p_value  # correction never lowers p


def test_holm_correction_runs() -> None:
    a, b = _three_metric_runs()
    cmp = compare(a, b, correction="holm", seed=0)
    adj = {c.metric: c.p_adjusted for c in cmp.comparisons}
    assert adj["has_a"] == pytest.approx(0.09375)  # 3 * 0.03125


# ---------------------------------------------------------------------------
# Warnings and validation
# ---------------------------------------------------------------------------


def test_fingerprint_mismatch_warns_but_compares_shared_ids() -> None:
    scorers: list[Scorer] = [ExactMatch()]
    ev_a = make_eval(scorers=scorers)
    cases_b = [Case(input=f"q{i}", reference="yes", id=f"c{i:03d}") for i in range(N - 5)]
    ev_b = Eval("reg-test", cases_b, scorers)
    a = run(ev_a, target=make_target(set(), name="a"), seed=1)
    b = run(ev_b, target=make_target(set(range(10)), N - 5, "b"), seed=1)
    cmp = compare(a, b, seed=0)
    assert any("fingerprint mismatch" in w for w in cmp.warnings)
    assert cmp.comparisons[0].n_pairs == N - 5
    assert cmp.verdict == "regressed"


def test_errored_cases_warn_as_dropped() -> None:
    ev = make_eval(10)
    a = run(ev, target=make_target(set(), 10, "a"), seed=1)
    incomplete = {f"q{i}": "yes" for i in range(8)}  # q8, q9 error
    b = run(ev, target=StaticTarget(incomplete, name="b"), seed=1)
    cmp = compare(a, b, seed=0)
    assert any("dropped" in w and "unpaired" in w for w in cmp.warnings)
    assert cmp.comparisons[0].n_pairs == 8


def test_asymmetric_metrics_warn_as_skipped() -> None:
    cases = [Case(input=f"q{i}", reference="yes", id=f"c{i:03d}") for i in range(10)]
    ev_a = Eval("reg-test", cases, [ExactMatch()])
    ev_b = Eval("reg-test", cases, [ExactMatch(), NamedRegex("has_digit", "1")])
    a = run(ev_a, target=make_target(set(), 10, "a"), seed=1)
    b = run(ev_b, target=make_target(set(), 10, "b"), seed=1)
    cmp = compare(a, b, seed=0)
    assert any("skipped" in w and "has_digit" in w for w in cmp.warnings)
    assert [c.metric for c in cmp.comparisons] == ["exact_match"]


def test_no_shared_metrics_raises() -> None:
    cases = [Case(input=f"q{i}", reference="yes", id=f"c{i:03d}") for i in range(10)]
    ev_a = Eval("reg-test", cases, [NamedRegex("only_a", "A")])
    ev_b = Eval("reg-test", cases, [NamedRegex("only_b", "B")])
    a = run(ev_a, target=make_target(set(), 10, "a"), seed=1)
    b = run(ev_b, target=make_target(set(), 10, "b"), seed=1)
    with pytest.raises(ValueError, match="share no metrics"):
        compare(a, b, seed=0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_alpha_validation(alpha: float) -> None:
    a, b = run_pair(set(), set())
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        compare(a, b, alpha=alpha)


# ---------------------------------------------------------------------------
# Rendering, serialization, determinism
# ---------------------------------------------------------------------------


def test_summary_contents() -> None:
    a, b = run_pair(set(), set(range(10)))
    cmp = compare(a, b, seed=0)
    text = cmp.summary()
    assert "reg-test" in text
    assert "baseline" in text and "candidate" in text
    assert a.run_id[:12] in text and b.run_id[:12] in text
    assert "REGRESSED" in text
    assert "[95% CI" in text
    assert "alpha=0.05" in text


def test_summary_renders_warnings_and_insufficient_metrics() -> None:
    ev = make_eval(4)
    a = run(ev, target=make_target(set(), 4, "a"), seed=1)
    b = run(ev, target=StaticTarget({"q0": "yes"}, name="b"), seed=1)
    text = compare(a, b, seed=0).summary()
    assert "insufficient data" in text
    assert "warning:" in text


def test_to_dict_is_json_serializable() -> None:
    a, b = run_pair(set(), set(range(10)))
    payload = compare(a, b, seed=0).to_dict()
    parsed = json.loads(json.dumps(payload))
    assert parsed["verdict"] == "regressed"
    assert parsed["comparisons"][0]["metric"] == "exact_match"


def test_compare_is_deterministic() -> None:
    a, b = run_pair(set(), set(range(10)))
    c1: RunComparison = compare(a, b, seed=5)
    c2: RunComparison = compare(a, b, seed=5)
    assert c1 == c2


def test_effect_sign_convention_candidate_minus_baseline() -> None:
    a, b = run_pair(set(), set(range(10)))
    forward = compare(a, b, seed=0).comparisons[0]
    backward = compare(b, a, seed=0).comparisons[0]
    assert forward.result is not None and backward.result is not None
    assert forward.result.effect == pytest.approx(-backward.result.effect)
