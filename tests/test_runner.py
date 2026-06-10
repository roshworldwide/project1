"""End-to-end tests for the runner (run/arun) and the Run record.

All tests are offline: they use StaticTarget / small local Target and Scorer
implementations, never the network.
"""

import asyncio
import json
from dataclasses import replace

import pytest

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.run import Run
from holdout.core.runner import arun, run
from holdout.core.scoring import Score, Scorer
from holdout.core.target import Completion, Target
from holdout.providers.static import StaticTarget
from holdout.scorers.exact import ExactMatch
from holdout.stats.estimate import Estimate

# Question -> correct answer for the reference eval.
QA: dict[str, str] = {
    "capital of France?": "Paris",
    "2+2?": "4",
    "color of the sky?": "blue",
    "opposite of hot?": "cold",
}

# A target that gets exactly one answer wrong, so exact_match is 0.75 and the
# bootstrap interval is non-degenerate.
RESPONSES: dict[str, str] = {**QA, "2+2?": "5"}


def make_eval(name: str = "smoke") -> Eval:
    cases = [Case(input=q, reference=a) for q, a in QA.items()]
    return Eval(name=name, cases=cases, scorers=[ExactMatch()])


def make_target(responses: dict[str, str] | None = None) -> StaticTarget:
    return StaticTarget(responses if responses is not None else RESPONSES)


class ExplodingScorer(Scorer):
    """A scorer that always raises, to exercise per-scorer error recording."""

    @property
    def name(self) -> str:
        return "exploding"

    async def score(self, case: Case, output: str) -> Score:
        raise RuntimeError("kaboom")


class CountingTarget:
    """Implements the Target protocol while tracking max in-flight generate calls."""

    def __init__(self, delay_s: float = 0.02) -> None:
        self._delay_s = delay_s
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "counting"

    @property
    def fingerprint(self) -> str:
        return "counting-target-fingerprint"

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        async with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self._delay_s)
        finally:
            async with self._lock:
                self._in_flight -= 1
        return Completion(text=prompt)


# --- shape of a run -------------------------------------------------------


def test_results_align_one_to_one_with_cases() -> None:
    ev = make_eval()
    r = run(ev, target=make_target(), seed=0)
    assert isinstance(r, Run)
    assert len(r.results) == len(ev.cases)
    assert [cr.case_id for cr in r.results] == [c.id for c in ev.cases]


def test_metrics_returns_dict_of_estimates() -> None:
    ev = make_eval()
    r = run(ev, target=make_target(), seed=0)
    metrics = r.metrics(n_resamples=500)
    assert isinstance(metrics, dict)
    assert set(metrics) == {"exact_match"}
    est = metrics["exact_match"]
    assert isinstance(est, Estimate)
    assert est.n == len(ev.cases)
    assert est.value == pytest.approx(0.75)
    assert est.ci_low <= est.value <= est.ci_high


def test_summary_contains_ci_and_short_run_id() -> None:
    r = run(make_eval(), target=make_target(), seed=0)
    s = r.summary()
    assert "CI" in s
    assert r.short_run_id in s
    assert len(r.short_run_id) == 12
    assert r.run_id.startswith(r.short_run_id)


# --- determinism: the flagship guarantee ----------------------------------


def test_same_eval_target_seed_gives_identical_run_id_and_metrics() -> None:
    r1 = run(make_eval(), target=make_target(), seed=7)
    r2 = run(make_eval(), target=make_target(), seed=7)
    assert r1.run_id == r2.run_id
    assert r1.metrics() == r2.metrics()


def test_different_seed_gives_different_run_id() -> None:
    r1 = run(make_eval(), target=make_target(), seed=7)
    r2 = run(make_eval(), target=make_target(), seed=8)
    assert r1.run_id != r2.run_id


def test_different_target_mapping_gives_different_run_id() -> None:
    r1 = run(make_eval(), target=make_target(), seed=7)
    r2 = run(make_eval(), target=make_target({**RESPONSES, "2+2?": "4"}), seed=7)
    assert r1.run_id != r2.run_id


def test_created_at_and_latency_do_not_affect_run_id() -> None:
    r1 = run(make_eval(), target=make_target(), seed=0)
    shifted = tuple(replace(cr, latency_s=cr.latency_s + 123.0) for cr in r1.results)
    r2 = replace(r1, created_at="1999-12-31T23:59:59+00:00", results=shifted)
    assert r2.created_at != r1.created_at
    assert [cr.latency_s for cr in r2.results] != [cr.latency_s for cr in r1.results]
    assert r2.run_id == r1.run_id


# --- error handling --------------------------------------------------------


def test_generation_failure_is_recorded_per_case_and_excluded_from_metrics() -> None:
    cases = [
        Case(input="a", reference="A", id="case-a"),
        Case(input="b", reference="B", id="case-b"),
        Case(input="missing", reference="M", id="case-missing"),
    ]
    ev = Eval(name="gen-errors", cases=cases, scorers=[ExactMatch()])
    target = StaticTarget({"a": "A", "b": "wrong"})  # default=None: unknown input raises

    r = run(ev, target=target, seed=1)

    failed = r.results[2]
    assert failed.case_id == "case-missing"
    assert failed.output is None
    assert failed.error is not None
    assert "generation failed" in failed.error
    assert dict(failed.scores) == {}
    assert r.n_errors == 1

    scores = r.case_scores("exact_match")
    assert scores == {"case-a": 1.0, "case-b": 0.0}  # failed case absent

    est = r.metrics(n_resamples=200)["exact_match"]
    assert est.n == 2
    assert est.value == pytest.approx(0.5)


def test_failing_scorer_does_not_block_other_scorers() -> None:
    cases = [Case(input="a", reference="A", id="only")]
    ev = Eval(name="scorer-errors", cases=cases, scorers=[ExactMatch(), ExplodingScorer()])

    r = run(ev, target=StaticTarget({"a": "A"}), seed=0)

    res = r.results[0]
    assert res.output == "A"
    assert res.error is not None
    assert "scorer 'exploding' failed" in res.error
    assert "kaboom" in res.error
    assert "exploding" not in res.scores
    assert res.scores["exact_match"].value == 1.0
    assert r.case_scores("exact_match") == {"only": 1.0}
    assert r.case_scores("exploding") == {}
    assert "exploding" not in r.metrics(n_resamples=100)
    assert "no data" in r.summary()


# --- concurrency ------------------------------------------------------------


def test_in_flight_generate_calls_never_exceed_max_concurrency() -> None:
    cases = [Case(input=f"p{i}", reference=f"p{i}") for i in range(12)]
    ev = Eval(name="concurrency", cases=cases, scorers=[ExactMatch()])
    target = CountingTarget()
    assert isinstance(target, Target)

    r = run(ev, target=target, seed=0, max_concurrency=3)

    assert len(r.results) == 12
    assert r.n_errors == 0
    assert target.max_in_flight <= 3
    assert target.max_in_flight >= 2  # the bound was actually exercised


def test_max_concurrency_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        run(make_eval(), target=make_target(), max_concurrency=0)


# --- event-loop discipline ---------------------------------------------------


async def test_run_inside_event_loop_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="running event loop"):
        run(make_eval(), target=make_target(), seed=0)


async def test_arun_works_inside_event_loop() -> None:
    r = await arun(make_eval(), target=make_target(), seed=0)
    assert len(r.results) == len(QA)
    assert r.n_errors == 0


# --- serialization and accessors ---------------------------------------------


def test_to_dict_from_dict_round_trip_preserves_everything() -> None:
    cases = [
        Case(input="a", reference="A", id="case-a"),
        Case(input="b", reference="B", id="case-b"),
        Case(input="missing", reference="M", id="case-missing"),
    ]
    ev = Eval(name="round-trip", cases=cases, scorers=[ExactMatch()])
    r = run(ev, target=StaticTarget({"a": "A", "b": "wrong"}), seed=42)

    payload = json.loads(json.dumps(r.to_dict()))  # prove JSON serializability
    restored = Run.from_dict(payload)

    assert restored.run_id == r.run_id
    assert payload["run_id"] == r.run_id
    assert restored.metrics(n_resamples=300) == r.metrics(n_resamples=300)
    assert restored.n_errors == r.n_errors == 1
    assert [cr.error for cr in restored.results] == [cr.error for cr in r.results]
    assert [cr.latency_s for cr in restored.results] == [cr.latency_s for cr in r.results]
    assert restored.created_at == r.created_at
    assert restored.seed == r.seed


def test_case_scores_unknown_scorer_raises_key_error() -> None:
    r = run(make_eval(), target=make_target(), seed=0)
    with pytest.raises(KeyError, match="unknown scorer"):
        r.case_scores("nope")


def test_score_kind_binary_and_never_scored_raises() -> None:
    r = run(make_eval(), target=make_target(), seed=0)
    assert r.score_kind("exact_match") == "binary"
    with pytest.raises(KeyError, match="no scores recorded"):
        r.score_kind("never_ran")
