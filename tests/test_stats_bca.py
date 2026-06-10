"""Tests for the BCa path of holdout.stats.bootstrap.

The percentile path is covered by tests/test_stats_basic.py; everything here
exercises method="bca": the scipy cross-validation, the skew correction, the
z0-undefined fallback, the mid-tie convention on discrete data, the accel==0
symmetric case, and the jackknife loop for custom statistics.
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from holdout.stats.bootstrap import bootstrap_ci


def _median(sample: NDArray[np.float64]) -> float:
    return float(np.median(sample))


def _unique_count(sample: NDArray[np.float64]) -> float:
    return float(np.unique(sample).size)


def _as_floats(arr: NDArray[np.float64]) -> list[float]:
    return [float(v) for v in arr]


# ---------------------------------------------------------------------------
# Cross-validation against scipy's BCa
# ---------------------------------------------------------------------------


def test_bca_matches_scipy_bca_on_continuous_normal_data() -> None:
    # Continuous data => no bootstrap statistic ever ties the point estimate,
    # so our mid-tie z0 convention and scipy's strictly-less convention agree.
    rng = np.random.default_rng(42)
    data: NDArray[np.float64] = rng.normal(loc=0.7, scale=1.0, size=100)
    scale = float(np.std(data))

    est = bootstrap_ci(_as_floats(data), n_resamples=20_000, seed=123, method="bca")
    assert est.method == "bootstrap-bca"
    assert est.value == pytest.approx(float(np.mean(data)))

    res = scipy_stats.bootstrap(
        (data,),
        np.mean,
        confidence_level=0.95,
        n_resamples=20_000,
        method="BCa",
        rng=np.random.default_rng(7),
    )
    scipy_low = float(res.confidence_interval.low)
    scipy_high = float(res.confidence_interval.high)

    # Two independent 20k-resample BCa bootstraps: a ~0.03 * scale tolerance
    # is over 10x the Monte-Carlo standard error of either endpoint.
    assert est.ci_low == pytest.approx(scipy_low, abs=0.03 * scale)
    assert est.ci_high == pytest.approx(scipy_high, abs=0.03 * scale)


# ---------------------------------------------------------------------------
# Skew correction direction
# ---------------------------------------------------------------------------


def test_bca_shifts_interval_right_for_right_skewed_mean() -> None:
    # Strongly right-skewed lognormal data: BCa's z0 (bootstrap means are
    # right-skewed, so the median resample sits below the point estimate)
    # and accel (positive jackknife skewness) both push the quantile levels
    # up, shifting the interval toward the right tail.
    rng = np.random.default_rng(8)
    vals = _as_floats(rng.lognormal(mean=0.0, sigma=1.0, size=80))

    # Same seed => identical bootstrap draws; only the quantile levels differ.
    bca = bootstrap_ci(vals, n_resamples=4_000, seed=17, method="bca")
    pct = bootstrap_ci(vals, n_resamples=4_000, seed=17, method="percentile")

    assert bca.method == "bootstrap-bca"
    assert (bca.ci_low, bca.ci_high) != (pct.ci_low, pct.ci_high)
    assert bca.ci_high > pct.ci_high
    assert bca.ci_low > pct.ci_low  # both endpoints weakly larger for right-skew
    bca_mid = (bca.ci_low + bca.ci_high) / 2.0
    pct_mid = (pct.ci_low + pct.ci_high) / 2.0
    assert bca_mid > pct_mid


# ---------------------------------------------------------------------------
# z0-undefined fallback
# ---------------------------------------------------------------------------


def test_z0_undefined_falls_back_to_percentile_and_discloses() -> None:
    # statistic = number of distinct values. On 20 distinct points the point
    # estimate is 20, but a with-replacement resample has fewer than 20
    # distinct values unless it is a full permutation (probability ~2e-8),
    # so every bootstrap statistic falls strictly below the estimate and the
    # bias-correction fraction is exactly 1.0 => z0 undefined.
    vals = [float(i) for i in range(20)]
    est = bootstrap_ci(vals, statistic=_unique_count, n_resamples=500, seed=2, method="bca")

    assert "bca z0 undefined" in est.method
    assert est.method.startswith("bootstrap-percentile")
    assert est.value == pytest.approx(20.0)
    assert math.isfinite(est.ci_low)
    assert math.isfinite(est.ci_high)
    assert est.ci_low <= est.ci_high
    # The interval is the percentile interval of the bootstrap distribution,
    # which lives strictly below the (unattainable) point estimate.
    assert est.ci_high < est.value


# ---------------------------------------------------------------------------
# Mid-tie convention on heavily discrete data
# ---------------------------------------------------------------------------


def test_discrete_binary_data_never_crashes_or_returns_nan() -> None:
    # 0/1 accuracy with n=10: bootstrap means tie the point estimate en
    # masse. The mid-rank tie convention keeps z0 finite, so this must
    # return a finite interval -- either genuine BCa or the disclosed
    # fallback, never a crash and never NaN bounds.
    vals = [1.0] * 7 + [0.0] * 3
    est = bootstrap_ci(vals, n_resamples=2_000, seed=5, method="bca")

    assert est.method in {"bootstrap-bca", "bootstrap-percentile (bca z0 undefined)"}
    assert math.isfinite(est.ci_low)
    assert math.isfinite(est.ci_high)
    assert 0.0 <= est.ci_low <= est.ci_high <= 1.0
    assert est.value == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# accel == 0 symmetric case
# ---------------------------------------------------------------------------


def test_symmetric_data_bca_is_close_to_percentile() -> None:
    # Perfectly symmetric data: the jackknife third moment vanishes, so
    # accel == 0, and the bootstrap distribution of the mean is symmetric
    # around the estimate, so z0 ~ 0. With the same seed the bootstrap
    # draws are identical, so the BCa and percentile intervals should be
    # nearly the same (only the quantile levels can differ, and barely).
    vals = [-2.0, -1.0, 0.0, 1.0, 2.0] * 6
    bca = bootstrap_ci(vals, n_resamples=4_000, seed=3, method="bca")
    pct = bootstrap_ci(vals, n_resamples=4_000, seed=3, method="percentile")

    assert bca.method == "bootstrap-bca"
    # Bootstrap SE of the mean here is ~0.26; the intervals agree far tighter.
    assert bca.ci_low == pytest.approx(pct.ci_low, abs=0.06)
    assert bca.ci_high == pytest.approx(pct.ci_high, abs=0.06)
    assert bca.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Custom statistic: the jackknife loop path
# ---------------------------------------------------------------------------


def test_median_statistic_goes_through_jackknife_loop_and_brackets_median() -> None:
    rng = np.random.default_rng(13)
    data: NDArray[np.float64] = rng.normal(loc=5.0, scale=1.0, size=101)
    vals = _as_floats(data)
    sample_median = float(np.median(data))

    est = bootstrap_ci(vals, statistic=_median, n_resamples=800, seed=9, method="bca")

    assert est.method == "bootstrap-bca"
    assert est.value == pytest.approx(sample_median)
    assert est.ci_low <= sample_median <= est.ci_high
    assert math.isfinite(est.ci_low)
    assert math.isfinite(est.ci_high)
    assert 0.0 < est.width < 2.0  # sane width for n=101 normal data


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_bca_estimate() -> None:
    rng = np.random.default_rng(0)
    vals = _as_floats(rng.normal(size=60))
    a = bootstrap_ci(vals, n_resamples=2_000, seed=123, method="bca")
    b = bootstrap_ci(vals, n_resamples=2_000, seed=123, method="bca")
    assert a == b
    assert a.method == "bootstrap-bca"


# ---------------------------------------------------------------------------
# Input validation and degenerate cases
# ---------------------------------------------------------------------------


def test_nan_values_raise() -> None:
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_ci([1.0, float("nan"), 2.0], method="bca")


def test_two_dimensional_input_raises() -> None:
    grid: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_ci(grid, method="bca")


def test_n_equals_one_returns_degenerate_method() -> None:
    est = bootstrap_ci([2.5], n_resamples=500, seed=0, method="bca")
    assert est.method == "degenerate (n=1)"
    assert est.value == est.ci_low == est.ci_high == pytest.approx(2.5)
    assert est.n == 1
