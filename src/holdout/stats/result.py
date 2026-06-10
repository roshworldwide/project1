"""The TestResult type: a significance test's full, honest output.

A test never returns a bare p-value: it returns the effect size, the
confidence interval on the effect, the number of pairs, and the name of
the test that produced it — everything needed to judge the claim.
"""

from dataclasses import dataclass

from holdout.stats.estimate import Estimate


@dataclass(frozen=True, slots=True)
class TestResult:
    """The result of a paired significance test.

    Parameters
    ----------
    test
        The test that produced this result (e.g. ``"paired-bootstrap"``).
    p_value
        Two-sided p-value for H0: no difference.
    effect
        Point estimate of the effect — the mean of per-pair differences,
        ``mean(b) - mean(a)``. Positive means ``b`` scored higher.
    ci
        Confidence interval on the effect.
    n
        Number of pairs the test used.
    detail
        Optional test-specific context (e.g. discordant-pair counts).
    """

    test: str
    p_value: float
    effect: float
    ci: Estimate
    n: int
    detail: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    def __str__(self) -> str:
        pct = f"{self.ci.level * 100:g}"
        return (
            f"Δ={self.effect:+.3f} [{pct}% CI {self.ci.ci_low:+.3f}, "
            f"{self.ci.ci_high:+.3f}], p={self.p_value:.4g} ({self.test}, n={self.n})"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "test": self.test,
            "p_value": self.p_value,
            "effect": self.effect,
            "ci": self.ci.to_dict(),
            "n": self.n,
            "detail": self.detail,
        }
