"""The Run: an immutable, content-addressed record of one evaluation.

A Run's identity (``run_id``) is the hash of its semantic content — eval
fingerprint, target fingerprint, scorer fingerprints, seed, and per-case
results. Wall-clock fields (timestamps, latencies) are recorded but excluded
from the hash, so the determinism guarantee holds: same seed + same inputs
=> identical run hash.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

from holdout.core.hashing import fingerprint, short_id
from holdout.core.scoring import Score, ScoreKind
from holdout.stats.bootstrap import bootstrap_ci
from holdout.stats.estimate import Estimate


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The outcome of one case within a run.

    Parameters
    ----------
    case_id
        Id of the case this result belongs to (pairs results across runs).
    output
        The target's output text, or ``None`` if generation failed.
    scores
        Scores keyed by scorer name. May be partial if a scorer errored.
    error
        Error description if generation or any scorer failed.
    latency_s
        Wall-clock generation latency (excluded from the run hash).
    """

    case_id: str
    output: str | None
    scores: Mapping[str, Score] = field(default_factory=dict)
    error: str | None = None
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "case_id": self.case_id,
            "output": self.output,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "error": self.error,
            "latency_s": self.latency_s,
        }


@dataclass(frozen=True)
class Run:
    """An immutable record of one evaluation run.

    Parameters
    ----------
    eval_name, eval_fingerprint
        Identity of the dataset that was evaluated.
    target_name, target_fingerprint
        Identity of the system under evaluation.
    scorer_names, scorer_fingerprints
        Identity of the measurements applied.
    seed
        The seed threaded through generation and aggregation.
    results
        Per-case results, in eval case order.
    created_at
        ISO-8601 UTC timestamp (excluded from the run hash).
    holdout_version
        Version of holdout that produced the run (excluded from the hash).
    """

    eval_name: str
    eval_fingerprint: str
    target_name: str
    target_fingerprint: str
    scorer_names: tuple[str, ...]
    scorer_fingerprints: tuple[str, ...]
    seed: int | None
    results: tuple[CaseResult, ...]
    created_at: str
    holdout_version: str

    @cached_property
    def run_id(self) -> str:
        """Content hash of the run's semantic fields (full SHA-256 hex)."""
        return fingerprint(
            {
                "eval_name": self.eval_name,
                "eval": self.eval_fingerprint,
                "target": self.target_fingerprint,
                "scorers": list(self.scorer_fingerprints),
                "seed": self.seed,
                "results": [
                    {
                        "case_id": r.case_id,
                        "output": r.output,
                        "scores": {k: v.to_dict() for k, v in r.scores.items()},
                        "error": r.error,
                    }
                    for r in self.results
                ],
            }
        )

    @property
    def short_run_id(self) -> str:
        """Twelve-character display prefix of :attr:`run_id`."""
        return short_id(self.run_id)

    @property
    def n_errors(self) -> int:
        """Number of cases that failed generation or scoring."""
        return sum(1 for r in self.results if r.error is not None)

    def case_scores(self, scorer_name: str) -> dict[str, float]:
        """Return per-case score values for one scorer, keyed by case id.

        This is raw paired data (the input to the statistics engine), not a
        reported metric — cases that errored for this scorer are absent.
        """
        if scorer_name not in self.scorer_names:
            raise KeyError(
                f"unknown scorer {scorer_name!r}; this run has {list(self.scorer_names)}"
            )
        return {
            r.case_id: r.scores[scorer_name].value for r in self.results if scorer_name in r.scores
        }

    def score_kind(self, scorer_name: str) -> ScoreKind:
        """Return the score kind (``"binary"``/``"continuous"``) for a scorer."""
        for r in self.results:
            if scorer_name in r.scores:
                return r.scores[scorer_name].kind
        raise KeyError(f"no scores recorded for scorer {scorer_name!r}")

    def metrics(self, *, level: float = 0.95, n_resamples: int = 10_000) -> dict[str, Estimate]:
        """Aggregate each scorer's per-case scores into an :class:`Estimate`.

        There is no API that returns a bare aggregate float: every metric
        carries a bootstrap confidence interval. The bootstrap RNG is seeded
        from the run hash, so the same run always reports identical
        intervals.
        """
        agg_seed = int(self.run_id[:8], 16)
        out: dict[str, Estimate] = {}
        for name in self.scorer_names:
            values = list(self.case_scores(name).values())
            if not values:
                continue
            out[name] = bootstrap_ci(values, level=level, n_resamples=n_resamples, seed=agg_seed)
        return out

    def summary(self, *, level: float = 0.95) -> str:
        """Render a human-readable summary — every metric with its interval."""
        lines = [
            f"{self.eval_name}  n={len(self.results)}  target={self.target_name}  "
            f"run={self.short_run_id}"
        ]
        metrics = self.metrics(level=level)
        width = max((len(n) for n in metrics), default=0)
        for name, est in metrics.items():
            lines.append(f"  {name:<{width}}  {est}")
        for name in self.scorer_names:
            if name not in metrics:
                lines.append(f"  {name:<{width}}  no data (all cases errored)")
        if self.n_errors:
            lines.append(f"  errors: {self.n_errors} case(s) failed (excluded from aggregates)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation (used by the run store)."""
        return {
            "run_id": self.run_id,
            "eval_name": self.eval_name,
            "eval_fingerprint": self.eval_fingerprint,
            "target_name": self.target_name,
            "target_fingerprint": self.target_fingerprint,
            "scorer_names": list(self.scorer_names),
            "scorer_fingerprints": list(self.scorer_fingerprints),
            "seed": self.seed,
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at,
            "holdout_version": self.holdout_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Run":
        """Reconstruct a Run from :meth:`to_dict` output.

        Raises
        ------
        ValueError
            If ``data`` does not have the expected structure, with context
            naming the offending field.
        """

        def _str_field(key: str) -> str:
            value = data.get(key)
            if not isinstance(value, str):
                raise ValueError(f"{key!r} must be a string, got {type(value).__name__}")
            return value

        def _str_tuple(key: str) -> tuple[str, ...]:
            value = data.get(key)
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError(f"{key!r} must be a list of strings")
            return tuple(value)

        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            raise ValueError(f"'results' must be a list, got {type(raw_results).__name__}")

        results: list[CaseResult] = []
        for i, r in enumerate(raw_results):
            if not isinstance(r, Mapping):
                raise ValueError(f"results[{i}] must be an object, got {type(r).__name__}")
            if "case_id" not in r or not isinstance(r["case_id"], str):
                raise ValueError(f"results[{i}] is missing a string 'case_id'")
            output = r.get("output")
            if output is not None and not isinstance(output, str):
                raise ValueError(f"results[{i}]['output'] must be a string or null")
            raw_scores = r.get("scores")
            if not isinstance(raw_scores, Mapping):
                raise ValueError(f"results[{i}]['scores'] must be an object")
            scores: dict[str, Score] = {}
            for k, v in raw_scores.items():
                if not isinstance(v, Mapping) or "value" not in v or "kind" not in v:
                    raise ValueError(
                        f"results[{i}] score {k!r} must be an object with 'value' and 'kind'"
                    )
                kind = v["kind"]
                if kind not in ("binary", "continuous"):
                    raise ValueError(f"results[{i}] score {k!r} has invalid kind {kind!r}")
                scores[str(k)] = Score(value=float(v["value"]), kind=kind, detail=v.get("detail"))
            error = r.get("error")
            if error is not None and not isinstance(error, str):
                raise ValueError(f"results[{i}]['error'] must be a string or null")
            results.append(
                CaseResult(
                    case_id=r["case_id"],
                    output=output,
                    scores=scores,
                    error=error,
                    latency_s=float(r.get("latency_s", 0.0)),
                )
            )

        seed = data.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError(f"'seed' must be an int or null, got {type(seed).__name__}")

        return cls(
            eval_name=_str_field("eval_name"),
            eval_fingerprint=_str_field("eval_fingerprint"),
            target_name=_str_field("target_name"),
            target_fingerprint=_str_field("target_fingerprint"),
            scorer_names=_str_tuple("scorer_names"),
            scorer_fingerprints=_str_tuple("scorer_fingerprints"),
            seed=seed,
            results=tuple(results),
            created_at=_str_field("created_at"),
            holdout_version=_str_field("holdout_version"),
        )
