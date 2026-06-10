"""Tests for holdout.stats: the Estimate type and the percentile bootstrap."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from holdout.stats.bootstrap import bootstrap_ci
from holdout.stats.estimate import Estimate


def _median(sample: NDArray[np.float64]) -> float:
    return float(np.median(sample))


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


def test_str_renders_value_ci_n_and_method() -> None:
    est = Estimate(value=0.75, ci_low=0.7, ci_high=0.8, n=4, method="bootstrap-percentile")
    rendered = str(est)
    assert rendered == "0.750 [95% CI 0.700, 0.800] (n=4, bootstrap-percentile)"
    assert "[95% CI" in rendered
    assert "bootstrap-percentile" in rendered
    assert "n=4" in rendered


@pytest.mark.parametrize(
    ("level", "expected"),
    [(0.9, "[90% CI"), (0.95, "[95% CI"), (0.99, "[99% CI")],
)
def test_str_level_rendering(level: float, expected: str) -> None:
    est = Estimate(value=0.5, ci_low=0.4, ci_high=0.6, n=10, level=level)
    assert expected in str(est)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.2, 1.5])
def test_level_out_of_range_raises(level: float) -> None:
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        Estimate(value=0.5, ci_low=0.4, ci_high=0.6, n=10, level=level)


@pytest.mark.parametrize("n", [0, -3])
def test_n_below_one_raises(n: int) -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        Estimate(value=0.5, ci_low=0.4, ci_high=0.6, n=n)


def test_ci_low_above_ci_high_raises() -> None:
    with pytest.raises(ValueError, match="must be <="):
        Estimate(value=0.5, ci_low=0.7, ci_high=0.6, n=10)


def test_degenerate_interval_is_allowed() -> None:
    est = Estimate(value=1.0, ci_low=1.0, ci_high=1.0, n=1)
    assert est.width == 0.0


def test_width() -> None:
    est = Estimate(value=0.5, ci_low=0.4, ci_high=0.7, n=10)
    assert est.width == pytest.approx(0.3)


def test_to_dict_contents() -> None:
    est = Estimate(
        value=0.5, ci_low=0.4, ci_high=0.6, n=12, level=0.9, method="bootstrap-percentile"
    )
    assert est.to_dict() == {
        "value": 0.5,
        "ci_low": 0.4,
        "ci_high": 0.6,
        "n": 12,
        "level": 0.9,
        "method": "bootstrap-percentile",
    }


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_same_seed_and_data_gives_identical_estimate() -> None:
    rng = np.random.default_rng(0)
    vals = [float(v) for v in rng.normal(size=60)]
    a = bootstrap_ci(vals, n_resamples=2_000, seed=123)
    b = bootstrap_ci(vals, n_resamples=2_000, seed=123)
    assert a == b
    assert str(a) == str(b)


def test_different_seed_gives_different_interval() -> None:
    rng = np.random.default_rng(1)
    vals = [float(v) for v in rng.normal(size=60)]
    a = bootstrap_ci(vals, n_resamples=2_000, seed=0)
    b = bootstrap_ci(vals, n_resamples=2_000, seed=1)
    # The point estimate is seed-independent; the resampled interval is not.
    assert a.value == b.value
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


@pytest.mark.parametrize("level", [0.0, 1.0, -1.0, 2.0])
def test_bootstrap_level_out_of_range_raises(level: float) -> None:
    with pytest.raises(ValueError, match=r"level must be in \(0, 1\)"):
        bootstrap_ci([1.0, 2.0, 3.0], level=level)


@pytest.mark.parametrize("n_resamples", [0, -1])
def test_n_resamples_below_one_raises(n_resamples: int) -> None:
    with pytest.raises(ValueError, match="n_resamples must be >= 1"):
        bootstrap_ci([1.0, 2.0, 3.0], n_resamples=n_resamples)


def test_constant_data_gives_zero_width_ci_at_constant() -> None:
    est = bootstrap_ci([0.42] * 25, n_resamples=500, seed=3)
    assert est.value == pytest.approx(0.42)
    assert est.ci_low == pytest.approx(0.42)
    assert est.ci_high == pytest.approx(0.42)
    assert est.width == pytest.approx(0.0)
    assert est.n == 25


def test_single_observation_gives_degenerate_ci() -> None:
    est = bootstrap_ci([3.14], n_resamples=500, seed=4)
    assert est.value == pytest.approx(3.14)
    assert est.ci_low == pytest.approx(3.14)
    assert est.ci_high == pytest.approx(3.14)
    assert est.n == 1


def test_custom_statistic_median_is_honored() -> None:
    rng = np.random.default_rng(7)
    data = rng.lognormal(mean=0.0, sigma=1.5, size=101)
    vals = [float(v) for v in data]
    sample_median = float(np.median(data))
    sample_mean = float(np.mean(data))

    est = bootstrap_ci(vals, statistic=_median, n_resamples=4_000, seed=11)
    assert est.value == pytest.approx(sample_median)
    assert est.ci_low <= sample_median <= est.ci_high
    # Heavily right-skewed data: a median interval sits well below the mean.
    assert est.ci_high < sample_mean

    est_mean = bootstrap_ci(vals, n_resamples=4_000, seed=11)
    assert est.value != est_mean.value
    assert (est.ci_low, est.ci_high) != (est_mean.ci_low, est_mean.ci_high)


def test_binary_data_ci_within_unit_interval() -> None:
    rng = np.random.default_rng(5)
    vals = [float(v) for v in rng.integers(0, 2, size=40)]
    assert 0.0 < float(np.mean(vals)) < 1.0  # non-degenerate sample
    est = bootstrap_ci(vals, n_resamples=5_000, seed=6)
    assert 0.0 <= est.ci_low <= est.value <= est.ci_high <= 1.0


def test_metadata_fields_are_recorded() -> None:
    est = bootstrap_ci([1.0, 2.0, 3.0, 4.0], level=0.9, n_resamples=500, seed=0)
    assert est.n == 4
    assert est.level == 0.9
    assert est.method == "bootstrap-percentile"


@settings(max_examples=60, deadline=None)
@given(
    xs=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_mean_ci_stays_within_data_range_and_is_deterministic(xs: list[float], seed: int) -> None:
    est = bootstrap_ci(xs, n_resamples=200, seed=seed)
    again = bootstrap_ci(xs, n_resamples=200, seed=seed)
    assert est == again

    tol = 1e-9 * max(1.0, max(abs(x) for x in xs))
    lo, hi = min(xs), max(xs)
    assert lo - tol <= est.ci_low <= est.ci_high <= hi + tol
    assert lo - tol <= est.value <= hi + tol


# ---------------------------------------------------------------------------
# Cross-validation against scipy
# ---------------------------------------------------------------------------


def test_matches_scipy_percentile_bootstrap() -> None:
    rng = np.random.default_rng(42)
    data: NDArray[np.float64] = rng.normal(loc=0.7, scale=1.0, size=100)
    vals = [float(v) for v in data]
    scale = float(np.std(data))

    est = bootstrap_ci(vals, n_resamples=20_000, seed=123)

    res = scipy_stats.bootstrap(
        (data,),
        np.mean,
        confidence_level=0.95,
        n_resamples=20_000,
        method="percentile",
        rng=np.random.default_rng(7),
    )
    scipy_low = float(res.confidence_interval.low)
    scipy_high = float(res.confidence_interval.high)

    # Loose Monte-Carlo tolerance: two independent 20k-resample percentile
    # bootstraps agree to well within 0.02 of the data scale.
    assert est.ci_low == pytest.approx(scipy_low, abs=0.02 * scale)
    assert est.ci_high == pytest.approx(scipy_high, abs=0.02 * scale)
    assert est.value == pytest.approx(float(np.mean(data)))


def test_ci_brackets_sample_mean_for_well_behaved_data() -> None:
    rng = np.random.default_rng(9)
    vals = [float(v) for v in rng.normal(loc=2.0, scale=0.5, size=100)]
    est = bootstrap_ci(vals, n_resamples=10_000, seed=21)
    assert est.ci_low <= float(np.mean(vals)) <= est.ci_high
    assert est.width > 0.0
