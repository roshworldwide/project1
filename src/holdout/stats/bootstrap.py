"""Bootstrap confidence intervals: BCa (default) and percentile.

References
----------
Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
Chapman & Hall. Ch. 13 (percentile intervals), ch. 14 (BCa intervals).

Efron, B. (1987). "Better Bootstrap Confidence Intervals". *Journal of the
American Statistical Association*, 82(397), 171-185. (The BCa construction:
bias-correction z0 and acceleration a via the jackknife.)
"""

from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from holdout.stats.estimate import Estimate

Statistic = Callable[[NDArray[np.float64]], float]

_NORMAL = NormalDist()


def _validate_sample(values: "Sequence[float] | NDArray[np.float64]") -> NDArray[np.float64]:
    """Convert to a float array, rejecting empty or non-finite samples."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not np.isfinite(arr).all():
        raise ValueError("values contain NaN or infinity")
    return arr


def _resampled_statistics(
    arr: NDArray[np.float64],
    statistic: Statistic | None,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, NDArray[np.float64]]:
    """Return (point estimate on the original sample, bootstrap distribution)."""
    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resamples = arr[indices]
    if statistic is None:
        return float(arr.mean()), resamples.mean(axis=1)
    point = float(statistic(arr))
    boot = np.asarray([statistic(row) for row in resamples], dtype=np.float64)
    return point, boot


def _jackknife_statistics(
    arr: NDArray[np.float64], statistic: Statistic | None
) -> NDArray[np.float64]:
    """Leave-one-out statistics (vectorized for the mean fast path)."""
    n = arr.size
    if statistic is None:
        return np.asarray((arr.sum() - arr) / (n - 1), dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = statistic(np.delete(arr, i))
    return out


def _bca_adjusted_level(q: float, z0: float, accel: float) -> float:
    """Map a nominal quantile level through the BCa transformation.

    Implements Efron & Tibshirani (1993) eq. 14.10. When the denominator
    ``1 - a*(z0 + z)`` is non-positive the transformation degenerates
    (possible only with adversarially skewed custom statistics); the level
    saturates to 0 or 1, which np.quantile maps to the sample extremes.
    """
    z = _NORMAL.inv_cdf(q)
    num = z0 + z
    d = 1.0 - accel * num
    if d <= 0.0:
        return 1.0 if num > 0 else 0.0
    return _NORMAL.cdf(z0 + num / d)


def bootstrap_ci(
    values: "Sequence[float] | NDArray[np.float64]",
    *,
    statistic: Statistic | None = None,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    method: Literal["bca", "percentile"] = "bca",
) -> Estimate:
    """Compute a bootstrap confidence interval for a statistic of ``values``.

    The default method is BCa — bias-corrected and accelerated (Efron 1987;
    Efron & Tibshirani 1993, ch. 14) — which corrects the percentile
    interval for median bias (via ``z0``, the normal quantile of the
    fraction of bootstrap statistics below the point estimate) and for
    skewness (via the acceleration ``a``, estimated from the jackknife).
    BCa is second-order accurate where the percentile method is only
    first-order accurate.

    Implementation notes, in the interest of honesty:

    - Ties between bootstrap statistics and the point estimate count half
      toward the bias-correction fraction (a mid-rank convention). With
      heavily discrete data (e.g. accuracy on a small eval) a strictly-less
      convention makes ``z0`` lurch; mid-rank degrades gracefully and
      reduces BCa to the percentile interval for degenerate distributions.
    - If every bootstrap statistic falls on one side of the point estimate,
      ``z0`` is undefined; we fall back to the percentile interval and say
      so in the Estimate's ``method``.
    - ``n == 1`` has no resampling distribution; the result is a degenerate
      interval labeled as such, never presented as a real CI.

    Parameters
    ----------
    values
        Observed per-case values. Must be non-empty, 1-D, and finite.
    statistic
        Statistic to bootstrap. Defaults to the mean (vectorized fast path).
    level
        Confidence level, in (0, 1). Default 0.95.
    n_resamples
        Number of bootstrap resamples. Default 10,000.
    seed
        Seed for the resampling RNG; same seed + same values => identical
        interval.
    method
        ``"bca"`` (default) or ``"percentile"``.

    Returns
    -------
    Estimate
        The point estimate on the original sample with its interval.
    """
    arr = _validate_sample(values)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    if arr.size == 1:
        v = float(arr[0]) if statistic is None else float(statistic(arr))
        return Estimate(value=v, ci_low=v, ci_high=v, n=1, level=level, method="degenerate (n=1)")

    rng = np.random.default_rng(seed)
    point, boot = _resampled_statistics(arr, statistic, n_resamples, rng)
    alpha = 1.0 - level
    n = int(arr.size)

    if method == "percentile":
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return Estimate(
            value=point,
            ci_low=float(lo),
            ci_high=float(hi),
            n=n,
            level=level,
            method="bootstrap-percentile",
        )

    # --- BCa (Efron & Tibshirani 1993, eq. 14.9-14.10) ---
    below = float((boot < point).sum())
    ties = float((boot == point).sum())
    prop = (below + 0.5 * ties) / float(boot.size)
    if prop <= 0.0 or prop >= 1.0:
        # z0 undefined: the whole bootstrap mass is on one side of the
        # estimate. Report the percentile interval and disclose the fallback.
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return Estimate(
            value=point,
            ci_low=float(lo),
            ci_high=float(hi),
            n=n,
            level=level,
            method="bootstrap-percentile (bca z0 undefined)",
        )
    z0 = _NORMAL.inv_cdf(prop)

    jack = _jackknife_statistics(arr, statistic)
    diffs = jack.mean() - jack
    denom = float((diffs**2).sum()) ** 1.5
    accel = 0.0 if denom == 0.0 else float((diffs**3).sum()) / (6.0 * denom)

    a1 = _bca_adjusted_level(alpha / 2.0, z0, accel)
    a2 = _bca_adjusted_level(1.0 - alpha / 2.0, z0, accel)
    lo = np.quantile(boot, min(a1, a2))
    hi = np.quantile(boot, max(a1, a2))
    return Estimate(
        value=point,
        ci_low=float(lo),
        ci_high=float(hi),
        n=n,
        level=level,
        method="bootstrap-bca",
    )
