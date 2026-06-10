"""The runner: executes an Eval against a Target, concurrently and seeded.

Cases run under bounded async concurrency so a 1,000-case eval is I/O-bound,
not framework-bound. Failures are recorded per case — one flaky provider
call never aborts a run; it shows up honestly in the error count instead.
"""

import asyncio
from datetime import UTC, datetime
from time import perf_counter

import holdout
from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.run import CaseResult, Run
from holdout.core.scoring import Score, Scorer
from holdout.core.target import Target


async def _run_case(
    case: Case,
    target: Target,
    scorers: tuple[Scorer, ...],
    seed: int | None,
    sem: asyncio.Semaphore,
) -> CaseResult:
    """Generate and score one case, recording any failure on the result."""
    assert case.id is not None  # Eval normalization guarantees ids
    async with sem:
        t0 = perf_counter()
        try:
            completion = await target.generate(case.input, seed=seed)
        except Exception as exc:
            return CaseResult(
                case_id=case.id,
                output=None,
                error=f"generation failed: {type(exc).__name__}: {exc}",
                latency_s=perf_counter() - t0,
            )
        latency = perf_counter() - t0

        scores: dict[str, Score] = {}
        errors: list[str] = []
        for scorer in scorers:
            try:
                scores[scorer.name] = await scorer.score(case, completion.text)
            except Exception as exc:
                errors.append(f"scorer {scorer.name!r} failed: {type(exc).__name__}: {exc}")
        return CaseResult(
            case_id=case.id,
            output=completion.text,
            scores=scores,
            error="; ".join(errors) if errors else None,
            latency_s=latency,
        )


async def arun(
    ev: Eval,
    *,
    target: Target,
    seed: int | None = None,
    max_concurrency: int = 8,
) -> Run:
    """Run ``ev`` against ``target`` and return an immutable :class:`Run`.

    Parameters
    ----------
    ev
        The eval to run.
    target
        The system under evaluation.
    seed
        Seed threaded to the target (where the backend supports it) and into
        the run's identity. Same seed + same inputs => identical run hash.
    max_concurrency
        Maximum number of cases in flight at once.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    sem = asyncio.Semaphore(max_concurrency)
    results = await asyncio.gather(
        *(_run_case(case, target, ev.scorers, seed, sem) for case in ev.cases)
    )
    return Run(
        eval_name=ev.name,
        eval_fingerprint=ev.fingerprint,
        target_name=target.name,
        target_fingerprint=target.fingerprint,
        scorer_names=tuple(s.name for s in ev.scorers),
        scorer_fingerprints=tuple(s.fingerprint for s in ev.scorers),
        seed=seed,
        results=tuple(results),
        created_at=datetime.now(UTC).isoformat(),
        holdout_version=holdout.__version__,
    )


def run(
    ev: Eval,
    *,
    target: Target,
    seed: int | None = None,
    max_concurrency: int = 8,
) -> Run:
    """Run ``ev`` against ``target`` synchronously (facade over :func:`arun`).

    Raises
    ------
    RuntimeError
        If called from inside a running event loop — use :func:`arun` there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun(ev, target=target, seed=seed, max_concurrency=max_concurrency))
    raise RuntimeError("run() cannot be called from a running event loop; use arun() instead")
