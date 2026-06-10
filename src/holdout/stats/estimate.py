"""The Estimate type: point estimates that never travel without uncertainty.

holdout's cultural rule is that no metric is ever reported as a naked number.
``Estimate`` is the type that enforces it — every aggregate in the public API
(run summaries, comparisons, reports) is an ``Estimate``, and its string form
always renders the confidence interval.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with its confidence interval.

    Parameters
    ----------
    value
        The point estimate (e.g. a mean score).
    ci_low, ci_high
        Lower and upper bounds of the confidence interval.
    n
        Number of observations the estimate is computed from.
    level
        Confidence level of the interval (default 0.95).
    method
        The method that produced the interval (e.g. ``"bootstrap-bca"``).
    """

    value: float
    ci_low: float
    ci_high: float
    n: int
    level: float = 0.95
    method: str = "bootstrap-percentile"

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if not self.ci_low <= self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_high - self.ci_low

    def __str__(self) -> str:
        pct = f"{self.level * 100:g}"
        return (
            f"{self.value:.3f} [{pct}% CI {self.ci_low:.3f}, {self.ci_high:.3f}] "
            f"(n={self.n}, {self.method})"
        )

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable representation."""
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "level": self.level,
            "method": self.method,
        }
