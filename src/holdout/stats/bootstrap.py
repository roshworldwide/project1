"""Bootstrap confidence intervals.

M1 ships the percentile bootstrap as the interim default; M2 upgrades the
default to bias-corrected and accelerated (BCa) intervals. Both follow
Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman &
Hall — ch. 13 (percentile), ch. 14 (BCa).
"""

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from holdout.stats.estimate import Estimate

Statistic = Callable[[NDArray[np.float64]], float]


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Statistic | None = None,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    method: Literal["percentile"] = "percentile",
) -> Estimate:
    """Compute a bootstrap confidence interval for a statistic of ``values``.

    Resamples ``values`` with replacement ``n_resamples`` times, computes the
    statistic on each resample, and takes the alpha/2 and 1-alpha/2 quantiles
    of the resulting bootstrap distribution as the interval (the percentile
    method; Efron & Tibshirani 1993, ch. 13).

    Parameters
    ----------
    values
        Observed per-case values. Must be non-empty.
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
        Interval method. M1 implements ``"percentile"``; ``"bca"`` arrives
        with the M2 statistics engine.

    Returns
    -------
    Estimate
        The point estimate on the original sample with its interval.

    Raises
    ------
    ValueError
        If ``values`` is empty or ``level`` is not in (0, 1).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    rng = np.random.default_rng(seed)
    n = int(arr.size)
    indices = rng.integers(0, n, size=(n_resamples, n))
    resamples = arr[indices]

    if statistic is None:
        point = float(arr.mean())
        boot: NDArray[np.float64] = resamples.mean(axis=1)
    else:
        point = float(statistic(arr))
        boot = np.asarray([statistic(row) for row in resamples], dtype=np.float64)

    alpha = 1.0 - level
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Estimate(
        value=point,
        ci_low=float(lo),
        ci_high=float(hi),
        n=n,
        level=level,
        method=f"bootstrap-{method}",
    )
