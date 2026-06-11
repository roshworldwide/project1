"""Measure holdout's framework overhead with real numbers.

Run: ``python benchmarks/overhead.py``

Honesty rules: nothing here is hardcoded; every number is measured on the
machine running the script. The questions answered:

1. How much time does the framework add per case when the target is
   instant? (pure overhead)
2. Is a 1,000-case eval I/O-bound? (wall time vs the theoretical floor for
   a target with simulated latency under bounded concurrency)
3. How long do the statistics take? (BCa bootstrap, full compare())
"""

import asyncio
import platform
import sys
from time import perf_counter

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.runner import arun
from holdout.core.target import Completion
from holdout.providers.static import StaticTarget
from holdout.regression.compare import compare
from holdout.scorers.exact import ExactMatch
from holdout.stats.bootstrap import bootstrap_ci

N_CASES = 1_000
CONCURRENCY = 64
SIM_LATENCY_S = 0.05


class DelayTarget:
    """A target that simulates provider latency with asyncio.sleep."""

    def __init__(self, answers: dict[str, str], delay_s: float) -> None:
        self._inner = StaticTarget(answers, name=f"delay-{delay_s * 1000:g}ms")
        self._delay_s = delay_s

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        await asyncio.sleep(self._delay_s)
        return await self._inner.generate(prompt, seed=seed)


def main() -> None:
    """Run the benchmark suite and print measured numbers."""
    answers = {f"q{i}": "yes" for i in range(N_CASES)}
    wrong = {f"q{i}": ("no" if i % 10 == 0 else "yes") for i in range(N_CASES)}
    cases = [Case(input=f"q{i}", reference="yes", id=f"c{i:04d}") for i in range(N_CASES)]
    ev = Eval("bench", cases, [ExactMatch()])

    print(f"machine: {platform.machine()} · {platform.platform()}")
    print(f"python:  {sys.version.split()[0]}")
    print(f"eval:    {N_CASES} cases, ExactMatch, max_concurrency={CONCURRENCY}")
    print()

    # 1 — pure framework overhead against an instant target.
    t0 = perf_counter()
    run_a = asyncio.run(
        arun(ev, target=StaticTarget(answers, name="instant"), seed=7, max_concurrency=CONCURRENCY)
    )
    instant_s = perf_counter() - t0
    print(f"instant target:   {instant_s:.3f}s wall  ->  {instant_s / N_CASES * 1e6:.0f} µs/case")

    # 2 — I/O-bound check against a simulated-latency target.
    floor_s = N_CASES / CONCURRENCY * SIM_LATENCY_S
    t0 = perf_counter()
    asyncio.run(
        arun(
            ev,
            target=DelayTarget(answers, SIM_LATENCY_S),
            seed=7,
            max_concurrency=CONCURRENCY,
        )
    )
    delay_s = perf_counter() - t0
    print(
        f"{SIM_LATENCY_S * 1000:g}ms-latency target: {delay_s:.3f}s wall vs {floor_s:.3f}s "
        f"theoretical floor  ->  {delay_s / floor_s:.2f}x (1.0x = perfectly I/O-bound)"
    )

    # 3 — statistics timing.
    values = list(run_a.case_scores("exact_match").values())
    t0 = perf_counter()
    bootstrap_ci(values, n_resamples=10_000, seed=0)
    boot_s = perf_counter() - t0
    print(f"BCa bootstrap:    {boot_s * 1000:.1f}ms (n={len(values)}, B=10,000)")

    run_b = asyncio.run(
        arun(ev, target=StaticTarget(wrong, name="wrong"), seed=7, max_concurrency=CONCURRENCY)
    )
    t0 = perf_counter()
    cmp = compare(run_a, run_b, seed=0)
    cmp_s = perf_counter() - t0
    print(f"compare():        {cmp_s * 1000:.1f}ms ({N_CASES} paired cases, verdict {cmp.verdict})")


if __name__ == "__main__":
    main()
