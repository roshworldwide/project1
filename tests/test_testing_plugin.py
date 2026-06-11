"""Tests for holdout.testing assertions and the pytest plugin."""

import inspect
from pathlib import Path

import pytest

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.run import Run
from holdout.core.runner import run
from holdout.providers.static import StaticTarget
from holdout.regression import RunComparison
from holdout.scorers import ExactMatch
from holdout.store import RunStore
from holdout.testing import (
    assert_adequately_powered,
    assert_no_regression,
    assert_significant_improvement,
    llm_eval,
)

pytest_plugins = ["pytester"]


def make_eval(n: int = 40) -> Eval:
    cases = [Case(input=f"q{i}", reference="yes", id=f"c{i:03d}") for i in range(n)]
    return Eval("plugin-test", cases, [ExactMatch()])


def make_run(wrong: set[int], n: int = 40, name: str = "t") -> Run:
    ev = make_eval(n)
    target = StaticTarget({f"q{i}": ("no" if i in wrong else "yes") for i in range(n)}, name=name)
    return run(ev, target=target, seed=7)


# ---------------------------------------------------------------------------
# assert_no_regression
# ---------------------------------------------------------------------------


def test_no_regression_passes_on_identical_runs() -> None:
    a, b = make_run(set(), name="a"), make_run(set(), name="b")
    cmp = assert_no_regression(a, b, seed=0)
    assert isinstance(cmp, RunComparison)
    assert cmp.verdict == "no_significant_change"


def test_no_regression_passes_on_improvement() -> None:
    a, b = make_run(set(range(10)), name="a"), make_run(set(), name="b")
    assert assert_no_regression(a, b, seed=0).verdict == "improved"


def test_no_regression_fails_with_table_in_message() -> None:
    a, b = make_run(set(), name="a"), make_run(set(range(10)), name="b")
    with pytest.raises(AssertionError) as exc:
        assert_no_regression(a, b, seed=0)
    msg = str(exc.value)
    assert "regression detected on: exact_match" in msg
    assert "Δ=" in msg and "CI" in msg  # the full comparison table travels along


def test_no_regression_refuses_to_certify_insufficient_data() -> None:
    ev = make_eval(4)
    a = run(ev, target=StaticTarget({f"q{i}": "yes" for i in range(4)}, name="a"), seed=1)
    b = run(ev, target=StaticTarget({"q0": "yes"}, name="b"), seed=1)
    with pytest.raises(AssertionError, match="refusing to certify"):
        assert_no_regression(a, b, seed=0)


# ---------------------------------------------------------------------------
# assert_significant_improvement
# ---------------------------------------------------------------------------


def test_improvement_passes_for_any_and_named_metric() -> None:
    a, b = make_run(set(range(10)), name="a"), make_run(set(), name="b")
    assert assert_significant_improvement(a, b, seed=0).verdict == "improved"
    assert_significant_improvement(a, b, metric="exact_match", seed=0)


def test_improvement_fails_when_nothing_improved() -> None:
    a, b = make_run(set(), name="a"), make_run(set(), name="b")
    with pytest.raises(AssertionError, match="no metric significantly improved"):
        assert_significant_improvement(a, b, seed=0)


def test_improvement_fails_on_unknown_metric() -> None:
    a, b = make_run(set(range(10)), name="a"), make_run(set(), name="b")
    with pytest.raises(AssertionError, match="not in comparison"):
        assert_significant_improvement(a, b, metric="nope", seed=0)


def test_improvement_fails_when_a_regression_coexists() -> None:
    # exact_match regresses while no other metric improves: never "clean".
    a, b = make_run(set(), name="a"), make_run(set(range(10)), name="b")
    with pytest.raises(AssertionError, match="not a clean improvement"):
        assert_significant_improvement(a, b, seed=0)


# ---------------------------------------------------------------------------
# assert_adequately_powered
# ---------------------------------------------------------------------------


def test_power_passes_on_zero_variance_and_returns_empty() -> None:
    a, b = make_run(set(), name="a"), make_run(set(), name="b")
    assert assert_adequately_powered(a, b, mde=0.05) == {}


def test_power_passes_when_n_is_ample() -> None:
    a = make_run(set(), n=200, name="a")
    b = make_run({0, 1, 2, 3, 4, 5}, n=200, name="b")  # 3% discordance
    analyses = assert_adequately_powered(a, b, mde=0.2)
    assert "exact_match" in analyses
    assert analyses["exact_match"].n <= 200


def test_power_fails_when_underpowered() -> None:
    a = make_run(set(), n=10, name="a")
    b = make_run({0, 2, 4, 6, 8}, n=10, name="b")  # 50% discordance
    with pytest.raises(AssertionError, match=r"have 10 pairs, need \d+"):
        assert_adequately_powered(a, b, mde=0.02)


def test_power_metric_variant_and_no_shared_metrics() -> None:
    a = make_run(set(), n=10, name="a")
    b = make_run({0, 2, 4, 6, 8}, n=10, name="b")
    with pytest.raises(AssertionError, match="underpowered"):
        assert_adequately_powered(a, b, mde=0.02, metric="exact_match")


# ---------------------------------------------------------------------------
# llm_eval decorator
# ---------------------------------------------------------------------------


def test_llm_eval_injects_run_and_hides_parameter(tmp_path: Path) -> None:
    ev = make_eval(6)
    target = StaticTarget({f"q{i}": "yes" for i in range(6)}, name="static")
    seen: list[Run] = []

    @llm_eval(ev, target=target, seed=3, store=str(tmp_path))
    def check(run: Run) -> None:
        seen.append(run)

    assert "run" not in inspect.signature(check).parameters
    marks = [m.name for m in getattr(check, "pytestmark", [])]
    assert "llm_eval" in marks

    check()
    assert seen[0].eval_name == "plugin-test"
    assert seen[0].seed == 3
    # The run was persisted before the test body executed.
    assert RunStore(tmp_path).load(seen[0].run_id).run_id == seen[0].run_id


# ---------------------------------------------------------------------------
# pytest plugin (exercised in a fresh pytester project)
# ---------------------------------------------------------------------------


def test_marker_is_registered(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--markers")
    result.stdout.fnmatch_lines(["*llm_eval*LLM evaluation*"])


def test_fixtures_read_cli_options(pytester: pytest.Pytester, tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    pytester.makepyfile(
        f"""
        from holdout.store import RunStore

        def test_fixtures(holdout_store, holdout_seed):
            assert isinstance(holdout_store, RunStore)
            assert str(holdout_store.root) == {str(store_dir)!r}
            assert holdout_seed == 42
        """
    )
    result = pytester.runpytest(f"--holdout-store={store_dir}", "--holdout-seed=42")
    result.assert_outcomes(passed=1)


def test_llm_eval_marker_deselection(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        from holdout import Case, Eval
        from holdout.providers.static import StaticTarget
        from holdout.scorers import ExactMatch
        from holdout.testing import llm_eval

        ev = Eval("d", [Case(input="q", reference="y", id="c1"),
                        Case(input="r", reference="y", id="c2")], [ExactMatch()])

        @llm_eval(ev, target=StaticTarget({"q": "y", "r": "y"}))
        def test_real_model(run):
            assert run.n_errors == 0

        def test_plain():
            pass
        """
    )
    result = pytester.runpytest("-m", "not llm_eval")
    result.assert_outcomes(passed=1, deselected=1)
