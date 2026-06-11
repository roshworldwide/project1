"""Statistically honest assertions — the "pytest for LLMs" surface.

These functions raise ``AssertionError`` with a full comparison table in
the message, so a failing CI run tells you *which* metric moved, by how
much, with what confidence — not just that a number changed.

A note on semantics: :func:`assert_no_regression` fails on
``insufficient_data`` as well as on a regression. If nothing could be
tested, certifying "no regression" would be dishonest.
"""

import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from holdout.core.evalset import Eval
from holdout.core.run import Run
from holdout.core.runner import run as _run_eval
from holdout.core.target import Target
from holdout.regression.compare import Correction, PairedTest, RunComparison, compare
from holdout.stats.power import PowerAnalysis, required_sample_size, sd_diff_from_scores

if TYPE_CHECKING:
    from holdout.store.run_store import RunStore

P = ParamSpec("P")
R = TypeVar("R")


def assert_no_regression(
    baseline: Run,
    candidate: Run,
    *,
    alpha: float = 0.05,
    correction: Correction = "benjamini-hochberg",
    test: PairedTest = "auto",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> RunComparison:
    """Assert that ``candidate`` did not significantly regress on any metric.

    Fails when any metric regressed at the corrected significance level —
    and also when *nothing could be tested* (``insufficient_data``),
    because certifying "no regression" without evidence would be dishonest.
    Passes on improvement or no significant change.

    Returns the full :class:`RunComparison` so callers can log or store it.
    """
    cmp = compare(
        baseline,
        candidate,
        alpha=alpha,
        correction=correction,
        test=test,
        n_resamples=n_resamples,
        seed=seed,
    )
    if cmp.verdict == "regressed":
        names = ", ".join(c.metric for c in cmp.regressed)
        raise AssertionError(f"regression detected on: {names}\n{cmp.summary()}")
    if cmp.verdict == "insufficient_data":
        raise AssertionError(
            "refusing to certify 'no regression': no metric had enough paired data "
            f"to test\n{cmp.summary()}"
        )
    return cmp


def assert_significant_improvement(
    baseline: Run,
    candidate: Run,
    *,
    metric: str | None = None,
    alpha: float = 0.05,
    correction: Correction = "benjamini-hochberg",
    test: PairedTest = "auto",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> RunComparison:
    """Assert that ``candidate`` significantly improved — and broke nothing.

    With ``metric`` set, that specific metric must have improved at the
    corrected level. Without it, at least one metric must have improved.
    Either way, any significant regression on *any* metric fails the
    assertion: an improvement that breaks something else is not shippable.

    Returns the full :class:`RunComparison`.
    """
    cmp = compare(
        baseline,
        candidate,
        alpha=alpha,
        correction=correction,
        test=test,
        n_resamples=n_resamples,
        seed=seed,
    )
    if cmp.regressed:
        names = ", ".join(c.metric for c in cmp.regressed)
        raise AssertionError(
            f"candidate regressed on {names} — not a clean improvement\n{cmp.summary()}"
        )
    if metric is not None:
        match = next((c for c in cmp.comparisons if c.metric == metric), None)
        if match is None:
            known = [c.metric for c in cmp.comparisons]
            raise AssertionError(f"metric {metric!r} not in comparison (has {known})")
        if match.verdict != "improved":
            raise AssertionError(
                f"{metric} did not significantly improve (verdict: {match.verdict})\n"
                f"{cmp.summary()}"
            )
    elif not cmp.improved:
        raise AssertionError(
            f"no metric significantly improved (verdict: {cmp.verdict})\n{cmp.summary()}"
        )
    return cmp


def assert_adequately_powered(
    baseline: Run,
    candidate: Run,
    *,
    mde: float,
    metric: str | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, PowerAnalysis]:
    """Assert the paired comparison can detect effects of size ``mde``.

    For each shared metric (or just ``metric``), measures the observed SD
    of per-pair differences and computes the sample size required to detect
    ``mde`` at the stated alpha and power. Fails if the runs have fewer
    paired cases than required — i.e. if a "no significant change" verdict
    from this comparison would be statistically meaningless for effects of
    the size you care about.

    Metrics whose paired differences have zero variance are trivially
    powered (any real effect would show) and pass.

    Returns the per-metric :class:`PowerAnalysis` for reporting.
    """
    metrics = (
        [metric]
        if metric is not None
        else [m for m in baseline.scorer_names if m in candidate.scorer_names]
    )
    if not metrics:
        raise AssertionError("runs share no metrics to power-check")

    analyses: dict[str, PowerAnalysis] = {}
    underpowered: list[str] = []
    for m in metrics:
        a_scores = baseline.case_scores(m)
        b_scores = candidate.case_scores(m)
        ids = sorted(set(a_scores) & set(b_scores))
        if len(ids) < 2:
            underpowered.append(f"{m}: only {len(ids)} paired case(s)")
            continue
        a = [a_scores[i] for i in ids]
        b = [b_scores[i] for i in ids]
        sd = sd_diff_from_scores(a, b)
        if sd == 0.0:
            continue  # zero observed variance: trivially powered for any mde > 0
        analysis = required_sample_size(mde, sd, alpha=alpha, power=power)
        analyses[m] = analysis
        if len(ids) < analysis.n:
            underpowered.append(
                f"{m}: have {len(ids)} pairs, need {analysis.n} to detect "
                f"|Δ| >= {mde:g} (sd_diff={sd:.4f}, alpha={alpha:g}, power={power:g})"
            )
    if underpowered:
        raise AssertionError(
            "comparison is underpowered — a null result here would be meaningless:\n  "
            + "\n  ".join(underpowered)
        )
    return analyses


def llm_eval(
    ev: Eval,
    *,
    target: Target,
    seed: int = 0,
    store: "RunStore | str | None" = None,
    max_concurrency: int = 8,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Decorate a test to receive a completed :class:`Run` as ``run``.

    The eval executes against ``target`` when the test runs (once per
    decorated test), is optionally persisted to a store, and is passed to
    the test as the ``run`` keyword argument::

        @llm_eval(support_qa, target=Ollama("llama3.2"), seed=7)
        def test_quality(run: Run) -> None:
            assert run.n_errors == 0

    Under pytest the decorated test is also marked ``llm_eval``, so real
    model calls can be deselected with ``-m "not llm_eval"``.

    Parameters
    ----------
    ev
        The eval to run.
    target
        The system under evaluation.
    seed
        Run seed (deterministic hash guarantee applies).
    store
        A :class:`~holdout.store.RunStore` or a path to one; when given,
        the run is saved before the test body executes.
    max_concurrency
        Concurrent cases in flight.
    """

    def decorate(fn: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            result = _run_eval(ev, target=target, seed=seed, max_concurrency=max_concurrency)
            if store is not None:
                from holdout.store.run_store import RunStore

                (store if isinstance(store, RunStore) else RunStore(store)).save(result)
            return fn(*args, run=result, **kwargs)

        # Hide the injected ``run`` parameter from pytest's fixture
        # resolution: the wrapper's visible signature must not request it.
        sig = inspect.signature(fn)
        params = [p for name, p in sig.parameters.items() if name != "run"]
        wrapper.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        try:
            import pytest

            return cast("Callable[..., R]", pytest.mark.llm_eval(wrapper))
        except ImportError:  # pragma: no cover - pytest is present in every dev env
            return wrapper

    return decorate
