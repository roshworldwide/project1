"""Tests for holdout.stats.correction and holdout.stats.power."""

import math
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from holdout.stats.correction import benjamini_hochberg, holm_bonferroni
from holdout.stats.power import (
    PowerAnalysis,
    minimum_detectable_effect,
    paired_binary_sd,
    required_sample_size,
    sd_diff_from_scores,
)

# z_{0.975} + z_{0.80} for the default alpha=0.05 / power=0.80 design.
_Z_SUM = 1.9599639845 + 0.8416212336


def _random_p_vectors(seed: int, count: int, max_len: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    vectors: list[list[float]] = []
    for _ in range(count):
        size = int(rng.integers(2, max_len + 1))
        vectors.append([float(x) for x in rng.uniform(0.0, 1.0, size=size)])
    return vectors


def _assert_elementwise_close(
    actual: Sequence[float], expected: Sequence[float], tol: float = 1e-12
) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=tol)


def _scipy_bh(p_values: Sequence[float]) -> list[float]:
    ref = np.asarray(false_discovery_control(np.asarray(p_values, dtype=np.float64), method="bh"))
    return [float(x) for x in ref]


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_values",
    [
        [0.01, 0.02, 0.03, 0.04, 0.05],
        [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205],
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.25],
    ],
)
def test_bh_matches_scipy_worked_examples(p_values: list[float]) -> None:
    _assert_elementwise_close(benjamini_hochberg(p_values), _scipy_bh(p_values))


@pytest.mark.parametrize("p_values", _random_p_vectors(seed=42, count=6, max_len=15))
def test_bh_matches_scipy_random_vectors(p_values: list[float]) -> None:
    _assert_elementwise_close(benjamini_hochberg(p_values), _scipy_bh(p_values))


def test_bh_worked_example_values() -> None:
    # p_(i) * m / i for [0.01..0.05], m=5: [0.05, 0.05, 0.05, 0.05, 0.05]
    # (each raw q_i = 0.01*i*5/i = 0.05; cumulative min leaves them equal).
    q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    _assert_elementwise_close(q, [0.05, 0.05, 0.05, 0.05, 0.05])


def test_bh_preserves_input_order_under_shuffle() -> None:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    q = benjamini_hochberg(p_values)
    rng = np.random.default_rng(7)
    perm = [int(i) for i in rng.permutation(len(p_values))]
    shuffled = [p_values[i] for i in perm]
    q_shuffled = benjamini_hochberg(shuffled)
    for out_pos, src_pos in enumerate(perm):
        assert q_shuffled[out_pos] == pytest.approx(q[src_pos], abs=1e-12)


def test_bh_empty_returns_empty_list() -> None:
    assert benjamini_hochberg([]) == []


@pytest.mark.parametrize("p", [0.0, 0.013, 0.5, 1.0])
def test_bh_single_p_is_unchanged(p: float) -> None:
    assert benjamini_hochberg([p]) == [p]


@pytest.mark.parametrize("p", [0.04, 0.2, 1.0])
def test_bh_all_equal_ps_give_q_equal_p(p: float) -> None:
    q = benjamini_hochberg([p] * 6)
    _assert_elementwise_close(q, [p] * 6, tol=1e-15)


@pytest.mark.parametrize(
    "p_values",
    [[-0.01], [1.01], [0.02, -1e-9, 0.5], [0.02, 1.0 + 1e-9]],
)
def test_bh_out_of_range_raises(p_values: list[float]) -> None:
    with pytest.raises(ValueError, match=r"p_values must all be in \[0, 1\]"):
        benjamini_hochberg(p_values)


@pytest.mark.parametrize("p_values", [[math.nan], [0.01, math.nan, 0.5]])
def test_bh_nan_raises(p_values: list[float]) -> None:
    with pytest.raises(ValueError, match=r"p_values must all be in \[0, 1\]"):
        benjamini_hochberg(p_values)


@pytest.mark.parametrize("p_values", _random_p_vectors(seed=99, count=5, max_len=20))
def test_bh_monotone_dominates_p_and_capped(p_values: list[float]) -> None:
    q = benjamini_hochberg(p_values)
    # q sorted by ascending p must be non-decreasing.
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    q_by_p = [q[i] for i in order]
    assert all(lo <= hi + 1e-15 for lo, hi in pairwise(q_by_p))
    # Adjustment never shrinks a p-value (up to a 1-ulp float artifact:
    # correction.py computes (p * m) / i, which for i == m can round one ulp
    # below p; scipy multiplies by m/i == 1.0 exactly), and is capped at 1.
    assert all(qi >= pi - 1e-12 for qi, pi in zip(q, p_values, strict=True))
    assert all(qi <= 1.0 for qi in q)


def test_bh_caps_at_one() -> None:
    q = benjamini_hochberg([0.9, 0.95, 1.0])
    assert max(q) == 1.0
    assert all(qi <= 1.0 for qi in q)


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_hand_worked_example() -> None:
    # p=[0.01, 0.04, 0.03], m=3. Sorted: [0.01, 0.03, 0.04].
    # Raw: [3*0.01, 2*0.03, 1*0.04] = [0.03, 0.06, 0.04].
    # Cumulative max: [0.03, 0.06, 0.06]. Back to input order: [0.03, 0.06, 0.06].
    q = holm_bonferroni([0.01, 0.04, 0.03])
    _assert_elementwise_close(q, [0.03, 0.06, 0.06], tol=1e-15)


@pytest.mark.parametrize("p_values", _random_p_vectors(seed=11, count=5, max_len=15))
def test_holm_q_geq_p_capped_and_monotone(p_values: list[float]) -> None:
    q = holm_bonferroni(p_values)
    assert all(qi >= pi for qi, pi in zip(q, p_values, strict=True))
    assert all(qi <= 1.0 for qi in q)
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    q_by_p = [q[i] for i in order]
    assert all(lo <= hi + 1e-15 for lo, hi in pairwise(q_by_p))


def test_holm_caps_at_one() -> None:
    # Raw sorted values [3*0.5, 2*0.6, 1*0.9] = [1.5, 1.2, 0.9];
    # cummax then clip => all 1.0.
    assert holm_bonferroni([0.5, 0.6, 0.9]) == [1.0, 1.0, 1.0]


def test_holm_preserves_input_order_under_shuffle() -> None:
    p_values = [0.002, 0.01, 0.04, 0.03, 0.3, 0.7, 0.011]
    q = holm_bonferroni(p_values)
    rng = np.random.default_rng(3)
    perm = [int(i) for i in rng.permutation(len(p_values))]
    shuffled = [p_values[i] for i in perm]
    q_shuffled = holm_bonferroni(shuffled)
    for out_pos, src_pos in enumerate(perm):
        assert q_shuffled[out_pos] == pytest.approx(q[src_pos], abs=1e-12)


def test_holm_empty_returns_empty_list() -> None:
    assert holm_bonferroni([]) == []


def test_holm_is_at_least_as_strict_as_bh() -> None:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    q_holm = holm_bonferroni(p_values)
    q_bh = benjamini_hochberg(p_values)
    assert all(h >= b - 1e-15 for h, b in zip(q_holm, q_bh, strict=True))


# ---------------------------------------------------------------------------
# minimum_detectable_effect / required_sample_size
# ---------------------------------------------------------------------------


def test_mde_hand_check_n100_sd_half() -> None:
    pa = minimum_detectable_effect(100, 0.5)
    assert pa.mde == pytest.approx(_Z_SUM * 0.5 / math.sqrt(100), rel=1e-9)
    assert pa.mde == pytest.approx(0.14007926090, rel=1e-9)
    assert pa.n == 100
    assert pa.sd_diff == 0.5
    assert pa.alpha == 0.05
    assert pa.power == 0.80


def test_mde_scales_inversely_with_sqrt_n() -> None:
    mde_100 = minimum_detectable_effect(100, 0.5).mde
    mde_400 = minimum_detectable_effect(400, 0.5).mde
    assert mde_400 == pytest.approx(mde_100 / 2.0, rel=1e-12)


def test_required_sample_size_hand_check() -> None:
    # n = ceil((2.8015852181 * 0.35 / 0.05)^2) = ceil(384.595...) = 385
    pa = required_sample_size(0.05, 0.35)
    assert pa.n == 385
    assert pa.mde == 0.05
    assert pa.sd_diff == 0.35


def test_required_sample_size_inverts_mde() -> None:
    original_n = 100
    pa = minimum_detectable_effect(original_n, 0.5)
    back = required_sample_size(pa.mde, 0.5)
    assert back.n <= original_n  # ceil effects only ever round n up to the exact design


def test_mde_at_required_n_is_detectable() -> None:
    n = required_sample_size(0.05, 0.35).n
    achieved = minimum_detectable_effect(n, 0.35).mde
    assert achieved <= 0.05 + 1e-12


@pytest.mark.parametrize("n", [1, 0, -3])
def test_mde_n_below_two_raises(n: int) -> None:
    with pytest.raises(ValueError, match="n must be >= 2"):
        minimum_detectable_effect(n, 0.5)


@pytest.mark.parametrize("sd_diff", [0.0, -0.5])
def test_sd_diff_nonpositive_raises(sd_diff: float) -> None:
    with pytest.raises(ValueError, match="sd_diff must be > 0"):
        minimum_detectable_effect(100, sd_diff)
    with pytest.raises(ValueError, match="sd_diff must be > 0"):
        required_sample_size(0.05, sd_diff)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5])
def test_alpha_out_of_range_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        minimum_detectable_effect(100, 0.5, alpha=alpha)
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        required_sample_size(0.05, 0.5, alpha=alpha)


@pytest.mark.parametrize("power", [0.0, 1.0, -0.2, 2.0])
def test_power_out_of_range_raises(power: float) -> None:
    with pytest.raises(ValueError, match=r"power must be in \(0, 1\)"):
        minimum_detectable_effect(100, 0.5, power=power)
    with pytest.raises(ValueError, match=r"power must be in \(0, 1\)"):
        required_sample_size(0.05, 0.5, power=power)


@pytest.mark.parametrize("mde", [0.0, -0.1])
def test_required_sample_size_nonpositive_mde_raises(mde: float) -> None:
    with pytest.raises(ValueError, match="mde must be > 0"):
        required_sample_size(mde, 0.5)


def test_required_sample_size_floored_at_two() -> None:
    pa = required_sample_size(100.0, 0.1)
    assert pa.n == 2


# ---------------------------------------------------------------------------
# paired_binary_sd
# ---------------------------------------------------------------------------


def test_paired_binary_sd_symmetric_discordance() -> None:
    # Var = 0.1 + 0.1 - 0^2 = 0.2
    assert paired_binary_sd(0.1, 0.1) == pytest.approx(math.sqrt(0.2), rel=1e-12)


def test_paired_binary_sd_one_sided_discordance() -> None:
    # Var = 0.2 + 0 - 0.2^2 = 0.16 => sd = 0.4
    assert paired_binary_sd(0.2, 0.0) == pytest.approx(0.4, rel=1e-12)


def test_paired_binary_sd_degenerate_zero() -> None:
    assert paired_binary_sd(0.0, 0.0) == 0.0


@pytest.mark.parametrize(("p01", "p10"), [(0.6, 0.5), (1.0, 0.1), (-0.1, 0.2), (0.2, -0.1)])
def test_paired_binary_sd_invalid_raises(p01: float, p10: float) -> None:
    with pytest.raises(ValueError, match="p01 and p10 must be >= 0"):
        paired_binary_sd(p01, p10)


# ---------------------------------------------------------------------------
# sd_diff_from_scores
# ---------------------------------------------------------------------------


def test_sd_diff_from_scores_matches_numpy() -> None:
    rng = np.random.default_rng(5)
    a = [float(x) for x in rng.uniform(0.0, 1.0, size=25)]
    b = [float(x) for x in rng.uniform(0.0, 1.0, size=25)]
    diffs = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    expected = float(np.std(diffs, ddof=1))
    assert sd_diff_from_scores(a, b) == pytest.approx(expected, rel=1e-12)


def test_sd_diff_from_scores_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        sd_diff_from_scores([0.1, 0.2, 0.3], [0.1, 0.2])


def test_sd_diff_from_scores_too_few_pairs_raises() -> None:
    with pytest.raises(ValueError, match="at least 2 pairs"):
        sd_diff_from_scores([0.5], [0.7])


# ---------------------------------------------------------------------------
# PowerAnalysis
# ---------------------------------------------------------------------------


def test_power_analysis_str_contains_fields() -> None:
    pa = minimum_detectable_effect(100, 0.5)
    rendered = str(pa)
    assert "n=100" in rendered
    assert f"{pa.mde:.4f}" in rendered  # 0.1401
    assert "0.1401" in rendered
    assert "alpha=0.05" in rendered
    assert "power 0.8" in rendered
    assert "sd_diff=0.5000" in rendered


def test_power_analysis_to_dict_round_content() -> None:
    pa = PowerAnalysis(n=64, mde=0.125, sd_diff=0.5, alpha=0.05, power=0.8)
    assert pa.to_dict() == {
        "n": 64,
        "mde": 0.125,
        "sd_diff": 0.5,
        "alpha": 0.05,
        "power": 0.8,
    }


def test_power_analysis_to_dict_matches_computed_fields() -> None:
    pa = required_sample_size(0.05, 0.35)
    d = pa.to_dict()
    assert d["n"] == pa.n
    assert d["mde"] == pa.mde
    assert d["sd_diff"] == pa.sd_diff
    assert d["alpha"] == pa.alpha
    assert d["power"] == pa.power
