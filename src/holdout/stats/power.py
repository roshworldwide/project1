"""Power analysis — whether your eval set is big enough to detect what you care about.

The silent failure mode of LLM evals is running 50 cases, seeing "no
significant change", and concluding safety — when the eval never had the
power to detect the regression being asked about. These functions make the
detectability of an effect explicit *before* trusting a null result.

All formulas use the normal approximation for a two-sided paired test on
the mean of per-pair differences.

References
----------
Chow, S.-C., Shao, J. & Wang, H. (2008). *Sample Size Calculations in
Clinical Research*, 2nd ed. Chapman & Hall/CRC. Ch. 3 (paired designs):
``n = ((z_{1-alpha/2} + z_{power}) * sigma_d / delta)^2``.

Miettinen, O. S. (1968). "The matched pairs design in the case of
all-or-none responses". *Biometrics*, 24(2), 339-352. (Variance of the
paired difference in proportions used by :func:`paired_binary_sd`.)
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from holdout.stats.paired import paired_diffs

_NORMAL = NormalDist()


@dataclass(frozen=True, slots=True)
class PowerAnalysis:
    """A power calculation with all of its assumptions on display.

    Parameters
    ----------
    n
        Number of paired cases.
    mde
        Minimum detectable effect: the smallest |mean difference| the
        design detects with the stated power.
    sd_diff
        Assumed standard deviation of per-pair differences.
    alpha
        Two-sided significance level.
    power
        Probability of detecting an effect of size ``mde``.
    """

    n: int
    mde: float
    sd_diff: float
    alpha: float
    power: float

    def __str__(self) -> str:
        return (
            f"n={self.n} pairs detects |Δ| >= {self.mde:.4f} at alpha={self.alpha:g} "
            f"with power {self.power:g} (sd_diff={self.sd_diff:.4f})"
        )

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""
        return {
            "n": self.n,
            "mde": self.mde,
            "sd_diff": self.sd_diff,
            "alpha": self.alpha,
            "power": self.power,
        }


def _validate_design(sd_diff: float, alpha: float, power: float) -> tuple[float, float]:
    """Validate shared design parameters; return the two z quantiles."""
    if sd_diff <= 0.0:
        raise ValueError(f"sd_diff must be > 0, got {sd_diff}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    return _NORMAL.inv_cdf(1.0 - alpha / 2.0), _NORMAL.inv_cdf(power)


def minimum_detectable_effect(
    n: int, sd_diff: float, *, alpha: float = 0.05, power: float = 0.80
) -> PowerAnalysis:
    """Smallest |mean difference| detectable with ``n`` paired cases.

    ``mde = (z_{1-alpha/2} + z_{power}) * sd_diff / sqrt(n)``
    (Chow, Shao & Wang 2008, ch. 3).

    Parameters
    ----------
    n
        Number of paired cases (>= 2).
    sd_diff
        Standard deviation of per-pair differences — measure it with
        :func:`sd_diff_from_scores` or assume it with
        :func:`paired_binary_sd`.
    alpha
        Two-sided significance level (default 0.05).
    power
        Desired detection probability (default 0.80).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    z_alpha, z_power = _validate_design(sd_diff, alpha, power)
    mde = (z_alpha + z_power) * sd_diff / math.sqrt(n)
    return PowerAnalysis(n=n, mde=mde, sd_diff=sd_diff, alpha=alpha, power=power)


def required_sample_size(
    mde: float, sd_diff: float, *, alpha: float = 0.05, power: float = 0.80
) -> PowerAnalysis:
    """Paired cases needed to detect a mean difference of ``mde``.

    ``n = ceil(((z_{1-alpha/2} + z_{power}) * sd_diff / mde)^2)``
    (Chow, Shao & Wang 2008, ch. 3), floored at 2.

    Parameters
    ----------
    mde
        The smallest |mean difference| that matters to you.
    sd_diff
        Standard deviation of per-pair differences.
    alpha
        Two-sided significance level (default 0.05).
    power
        Desired detection probability (default 0.80).
    """
    if mde <= 0.0:
        raise ValueError(f"mde must be > 0, got {mde}")
    z_alpha, z_power = _validate_design(sd_diff, alpha, power)
    n = max(2, math.ceil(((z_alpha + z_power) * sd_diff / mde) ** 2))
    return PowerAnalysis(n=n, mde=mde, sd_diff=sd_diff, alpha=alpha, power=power)


def paired_binary_sd(p01: float, p10: float) -> float:
    """SD of per-pair differences for paired binary outcomes.

    For binary scores, the per-pair difference is +1 with probability
    ``p01`` (case improved), -1 with probability ``p10`` (case regressed),
    else 0 — so ``Var(d) = p01 + p10 - (p01 - p10)^2`` (Miettinen 1968).
    Use the discordance you expect; 10-20% total discordance is typical for
    prompt changes on a stable eval.

    Parameters
    ----------
    p01
        Expected fraction of cases the candidate fixes (0 -> 1).
    p10
        Expected fraction of cases the candidate breaks (1 -> 0).
    """
    if p01 < 0.0 or p10 < 0.0 or p01 + p10 > 1.0:
        raise ValueError(f"p01 and p10 must be >= 0 with p01 + p10 <= 1, got {p01} and {p10}")
    var = p01 + p10 - (p01 - p10) ** 2
    return math.sqrt(max(var, 0.0))


def sd_diff_from_scores(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Sample SD (ddof=1) of observed per-pair differences.

    Measure ``sd_diff`` from a pilot comparison, then plug it into
    :func:`minimum_detectable_effect` or :func:`required_sample_size`.
    """
    d = paired_diffs(scores_a, scores_b)
    return float(d.std(ddof=1))
