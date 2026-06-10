"""Tests for holdout.stats.paired (paired significance tests) and holdout.stats.result."""

from collections.abc import Callable
from itertools import product

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from holdout.stats.estimate import Estimate
from holdout.stats.paired import (
    mcnemar_test,
    paired_bootstrap_test,
    paired_diffs,
    permutation_test,
)

# Aliased so pytest does not try to collect the class as a test suite.
from holdout.stats.result import TestResult as Result

PairedTest = Callable[..., Result]


def _floats(arr: NDArray[np.float64]) -> list[float]:
    return [float(v) for v in arr]


def _estimate(value: float = 0.5, lo: float = 0.4, hi: float = 0.6, n: int = 10) -> Estimate:
    return Estimate(value=value, ci_low=lo, ci_high=hi, n=n)


def _brute_force_signflip_p(d: NDArray[np.float64]) -> float:
    """Independent exact sign-flip p-value: enumerate all 2^n assignments."""
    obs = float(d.mean())
    tol = 1e-12 * max(1.0, abs(obs))
    n = int(d.size)
    extreme = 0
    for signs in product((1.0, -1.0), repeat=n):
        perm_mean = float(np.mean(np.asarray(signs, dtype=np.float64) * d))
        if abs(perm_mean) >= abs(obs) - tol:
            extreme += 1
    return extreme / float(2**n)


def _shifted_pair(
    n: int = 60, shift: float = 0.5, noise: float = 0.01, seed: int = 3
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.5, 0.1, size=n)
    b = a + shift + rng.normal(0.0, noise, size=n)
    return _floats(a), _floats(b)


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------


def test_result_str_renders_signed_effect_ci_p_test_and_n() -> None:
    res = Result(
        test="paired-bootstrap",
        p_value=0.0123,
        effect=0.5,
        ci=Estimate(value=0.5, ci_low=0.412, ci_high=0.588, n=24),
        n=24,
    )
    rendered = str(res)
    assert rendered == "Δ=+0.500 [95% CI +0.412, +0.588], p=0.0123 (paired-bootstrap, n=24)"
    assert "+0.500" in rendered  # effect carries an explicit sign
    assert "[95% CI" in rendered
    assert "p=0.0123" in rendered
    assert "paired-bootstrap" in rendered
    assert "n=24" in rendered


def test_result_str_negative_effect_is_signed() -> None:
    res = Result(
        test="permutation-exact",
        p_value=0.5,
        effect=-0.25,
        ci=Estimate(value=-0.25, ci_low=-0.31, ci_high=-0.19, n=8),
        n=8,
    )
    assert str(res) == "Δ=-0.250 [95% CI -0.310, -0.190], p=0.5 (permutation-exact, n=8)"


@pytest.mark.parametrize("p", [-0.01, -1.0, 1.0001, 2.0])
def test_result_p_value_out_of_range_raises(p: float) -> None:
    with pytest.raises(ValueError, match=r"p_value must be in \[0, 1\]"):
        Result(test="t", p_value=p, effect=0.0, ci=_estimate(), n=10)


@pytest.mark.parametrize("n", [0, -5])
def test_result_n_below_one_raises(n: int) -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        Result(test="t", p_value=0.5, effect=0.0, ci=_estimate(), n=n)


def test_result_to_dict_round_content() -> None:
    ci = Estimate(value=0.1, ci_low=0.05, ci_high=0.15, n=12, level=0.9, method="bootstrap-bca")
    res = Result(test="mcnemar-exact", p_value=0.25, effect=0.1, ci=ci, n=12, detail="d")
    assert res.to_dict() == {
        "test": "mcnemar-exact",
        "p_value": 0.25,
        "effect": 0.1,
        "ci": {
            "value": 0.1,
            "ci_low": 0.05,
            "ci_high": 0.15,
            "n": 12,
            "level": 0.9,
            "method": "bootstrap-bca",
        },
        "n": 12,
        "detail": "d",
    }


# ---------------------------------------------------------------------------
# paired_diffs
# ---------------------------------------------------------------------------


def test_paired_diffs_returns_b_minus_a() -> None:
    d = paired_diffs([1.0, 2.0, 3.0], [1.5, 1.0, 4.0])
    assert d.tolist() == pytest.approx([0.5, -1.0, 1.0])


def test_paired_diffs_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length, got 3 and 2"):
        paired_diffs([1.0, 2.0, 3.0], [1.0, 2.0])


@pytest.mark.parametrize(("a", "b"), [([1.0], [2.0]), ([], [])])
def test_paired_diffs_fewer_than_two_pairs_raises(a: list[float], b: list[float]) -> None:
    with pytest.raises(ValueError, match="at least 2 pairs"):
        paired_diffs(a, b)


@pytest.mark.parametrize(
    ("a", "b"),
    [([1.0, float("nan")], [1.0, 2.0]), ([1.0, 2.0], [float("nan"), 2.0])],
)
def test_paired_diffs_nan_raises(a: list[float], b: list[float]) -> None:
    with pytest.raises(ValueError, match="NaN"):
        paired_diffs(a, b)


def test_paired_diffs_two_dimensional_raises() -> None:
    square = np.zeros((2, 2), dtype=np.float64)
    flat = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_diffs(square, square)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_diffs(square, flat)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------


def test_bootstrap_null_case_p_well_above_alpha() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0.5, 0.1, size=40)
    b = a + rng.normal(0.0, 0.05, size=40)  # small symmetric noise, no real shift
    res = paired_bootstrap_test(_floats(a), _floats(b), n_resamples=1_000, seed=7)
    assert res.p_value > 0.5
    assert res.effect == pytest.approx(0.0, abs=0.05)


def test_bootstrap_strong_shift_is_detected() -> None:
    a, b = _shifted_pair(n=60, shift=0.5, noise=0.01, seed=3)
    res = paired_bootstrap_test(a, b, n_resamples=2_000, seed=0)
    assert res.p_value <= 0.0011  # at the 1/(B+1) floor for this effect size
    assert res.ci.ci_low > 0.0  # CI excludes zero
    assert res.effect == pytest.approx(0.5, abs=0.05)
    assert res.test == "paired-bootstrap"


def test_bootstrap_sign_convention_swapping_a_and_b_flips_effect() -> None:
    a, b = _shifted_pair()
    forward = paired_bootstrap_test(a, b, n_resamples=500, seed=0)
    backward = paired_bootstrap_test(b, a, n_resamples=500, seed=0)
    assert forward.effect > 0.0
    assert backward.effect == pytest.approx(-forward.effect, rel=1e-12)
    assert backward.p_value == forward.p_value  # two-sided: same evidence either way


def test_bootstrap_p_is_never_zero_floor_is_one_over_b_plus_one() -> None:
    a = _floats(np.linspace(0.0, 1.0, 30))
    b = [x + 10.0 for x in a]  # huge shift: no resample can be as extreme
    res = paired_bootstrap_test(a, b, n_resamples=999, seed=1)
    assert res.p_value == pytest.approx(1.0 / (999 + 1))
    assert res.p_value > 0.0


def test_bootstrap_same_seed_is_deterministic() -> None:
    a, b = _shifted_pair(n=30, shift=0.05, noise=0.1, seed=9)
    first = paired_bootstrap_test(a, b, n_resamples=800, seed=21)
    second = paired_bootstrap_test(a, b, n_resamples=800, seed=21)
    assert first == second
    assert str(first) == str(second)


def test_bootstrap_different_seed_p_may_differ_slightly() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 0.2, size=20)
    b = a + 0.08 + rng.normal(0.0, 0.15, size=20)
    one = paired_bootstrap_test(_floats(a), _floats(b), n_resamples=400, seed=0)
    two = paired_bootstrap_test(_floats(a), _floats(b), n_resamples=400, seed=1)
    assert one.effect == two.effect  # the observed effect is seed-independent
    assert one.p_value != two.p_value  # the Monte-Carlo p-value is not
    assert abs(one.p_value - two.p_value) < 0.05  # ... but only slightly


def test_bootstrap_n_is_pair_count_and_detail_says_two_sided() -> None:
    a, b = _shifted_pair(n=37)
    res = paired_bootstrap_test(a, b, n_resamples=500, seed=0)
    assert res.n == 37
    assert res.detail is not None
    assert "two-sided" in res.detail


# ---------------------------------------------------------------------------
# mcnemar_test
# ---------------------------------------------------------------------------


def _discordant_binary(
    n01: int, n10: int, both0: int, both1: int
) -> tuple[list[float], list[float]]:
    a = [0.0] * n01 + [1.0] * n10 + [0.0] * both0 + [1.0] * both1
    b = [1.0] * n01 + [0.0] * n10 + [0.0] * both0 + [1.0] * both1
    return a, b


def test_mcnemar_hand_computed_exact_p_8_improvements_2_regressions() -> None:
    a, b = _discordant_binary(n01=8, n10=2, both0=20, both1=20)  # n = 50
    res = mcnemar_test(a, b, n_resamples=500, seed=0)
    # Exact conditional: 2 * sum_{i=0..2} C(10, i) / 2^10 = 2 * (1 + 10 + 45) / 1024 = 7/64.
    assert res.p_value == pytest.approx(7.0 / 64.0, rel=1e-12)
    assert res.test == "mcnemar-exact"
    assert res.n == 50
    assert res.effect == pytest.approx((8 - 2) / 50, rel=1e-12)
    assert res.effect == pytest.approx(float(np.mean(b) - np.mean(a)), rel=1e-12)
    assert res.detail is not None
    assert "improved=8" in res.detail
    assert "regressed=2" in res.detail


def test_mcnemar_hand_computed_exact_p_1_improvement_9_regressions() -> None:
    a, b = _discordant_binary(n01=1, n10=9, both0=10, both1=10)  # n = 30
    res = mcnemar_test(a, b, n_resamples=500, seed=0)
    # 2 * sum_{i=0..1} C(10, i) / 2^10 = 2 * (1 + 10) / 1024 = 11/512.
    assert res.p_value == pytest.approx(11.0 / 512.0, rel=1e-12)
    assert res.effect == pytest.approx((1 - 9) / 30, rel=1e-12)
    assert res.detail is not None
    assert "improved=1" in res.detail
    assert "regressed=9" in res.detail


def test_mcnemar_equal_discordant_counts_p_capped_at_one() -> None:
    a, b = _discordant_binary(n01=3, n10=3, both0=2, both1=2)
    res = mcnemar_test(a, b, n_resamples=500, seed=0)
    assert res.p_value == 1.0  # 2 * P(X <= 3 | Binom(6, 1/2)) > 1, capped
    assert res.effect == pytest.approx(0.0)


def test_mcnemar_zero_discordant_pairs_p_is_one() -> None:
    a = [0.0, 1.0, 1.0, 0.0, 1.0]
    res = mcnemar_test(a, list(a), n_resamples=500, seed=0)
    assert res.p_value == 1.0
    assert res.effect == 0.0


def test_mcnemar_non_binary_scores_a_raises_naming_argument() -> None:
    with pytest.raises(ValueError, match="scores_a"):
        mcnemar_test([0.0, 0.5, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0])


def test_mcnemar_non_binary_scores_b_raises_naming_argument() -> None:
    with pytest.raises(ValueError, match="scores_b"):
        mcnemar_test([0.0, 1.0, 0.0, 1.0], [1.0, 0.25, 0.0, 1.0])


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

_SMALL_A = [0.10, 0.40, 0.35, 0.80, 0.20, 0.60, 0.55, 0.30]
_SMALL_B = [0.25, 0.42, 0.50, 0.78, 0.33, 0.71, 0.60, 0.41]


def test_permutation_exact_branch_matches_brute_force_enumeration() -> None:
    res = permutation_test(_SMALL_A, _SMALL_B, n_resamples=2_000, seed=0)  # 2^8 = 256 <= 2000
    expected = _brute_force_signflip_p(paired_diffs(_SMALL_A, _SMALL_B))
    assert res.test == "permutation-exact"
    assert res.p_value == pytest.approx(expected, rel=1e-12)
    assert res.p_value > 0.0  # the observed assignment is in the enumeration
    assert res.p_value >= 1.0 / 256.0


def test_permutation_exact_branch_matches_scipy() -> None:
    def mean_diff(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        return float(np.mean(y) - np.mean(x))

    res = permutation_test(_SMALL_A, _SMALL_B, n_resamples=2_000, seed=0)
    scipy_res = scipy_stats.permutation_test(
        (np.asarray(_SMALL_A, dtype=np.float64), np.asarray(_SMALL_B, dtype=np.float64)),
        mean_diff,
        permutation_type="samples",  # flip within pairs; exact since n_resamples >= 2^8
        n_resamples=100_000,
        alternative="two-sided",
        vectorized=False,
    )
    assert res.effect == pytest.approx(float(scipy_res.statistic), rel=1e-12)
    assert res.p_value == pytest.approx(float(scipy_res.pvalue), rel=1e-9)


def test_permutation_all_zero_differences_gives_p_one() -> None:
    a = [0.2, 0.4, 0.6, 0.8]
    res = permutation_test(a, list(a), n_resamples=500, seed=0)
    assert res.test == "permutation-exact"
    assert res.p_value == 1.0
    assert res.effect == 0.0


def test_permutation_mc_branch_is_deterministic_and_named() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 0.2, size=20)
    b = a + 0.08 + rng.normal(0.0, 0.15, size=20)
    first = permutation_test(_floats(a), _floats(b), n_resamples=500, seed=5)  # 2^20 > 500
    second = permutation_test(_floats(a), _floats(b), n_resamples=500, seed=5)
    assert first.test == "permutation-mc"
    assert first == second
    other_seed = permutation_test(_floats(a), _floats(b), n_resamples=500, seed=6)
    assert abs(first.p_value - other_seed.p_value) < 0.05


def test_permutation_mc_p_floor_for_huge_shift() -> None:
    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 1.0, size=20)
    b = a + 7.0
    res = permutation_test(_floats(a), _floats(b), n_resamples=500, seed=5)
    assert res.test == "permutation-mc"
    assert res.p_value == pytest.approx(1.0 / (500 + 1))
    assert res.p_value > 0.0


# ---------------------------------------------------------------------------
# Shared contracts across all three tests
# ---------------------------------------------------------------------------

_ALL_TESTS: list[PairedTest] = [paired_bootstrap_test, mcnemar_test, permutation_test]


@pytest.mark.parametrize("test_fn", _ALL_TESTS, ids=lambda f: str(f.__name__))
def test_ci_is_bootstrap_estimate_with_pair_count(test_fn: PairedTest) -> None:
    # Binary scores so the same data is valid for mcnemar_test too.
    a = [0.0, 1.0] * 10
    b = [1.0, 1.0, 0.0, 1.0] + [0.0, 1.0] * 8
    res = test_fn(a, b, n_resamples=400, seed=0)
    assert isinstance(res, Result)
    assert isinstance(res.ci, Estimate)
    assert "bootstrap" in res.ci.method
    assert res.n == 20
    assert res.ci.n == 20
    assert res.ci.level == 0.95


@pytest.mark.parametrize("test_fn", _ALL_TESTS, ids=lambda f: str(f.__name__))
def test_single_pair_raises_for_all(test_fn: PairedTest) -> None:
    with pytest.raises(ValueError, match="at least 2 pairs"):
        test_fn([1.0], [1.0], n_resamples=100, seed=0)
