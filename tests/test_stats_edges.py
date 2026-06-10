"""Edge-branch tests for holdout.stats internals (defensive guards)."""

import pytest

from holdout.stats.bootstrap import _bca_adjusted_level
from holdout.stats.correction import benjamini_hochberg
from holdout.stats.paired import _binom_cdf_half


def test_bca_adjusted_level_saturates_when_denominator_degenerates() -> None:
    # accel large enough that 1 - a*(z0 + z) <= 0: positive num saturates to 1.
    assert _bca_adjusted_level(0.975, z0=1.0, accel=0.5) == 1.0
    # Mirrored case: negative num saturates to 0.
    assert _bca_adjusted_level(0.025, z0=-1.0, accel=-0.5) == 0.0


def test_bca_adjusted_level_normal_path_is_monotone() -> None:
    lo = _bca_adjusted_level(0.025, z0=0.1, accel=0.05)
    hi = _bca_adjusted_level(0.975, z0=0.1, accel=0.05)
    assert 0.0 < lo < hi < 1.0


def test_binom_cdf_half_boundary_guards() -> None:
    assert _binom_cdf_half(-1, 10) == 0.0
    assert _binom_cdf_half(10, 10) == 1.0
    assert _binom_cdf_half(12, 10) == 1.0
    # Interior value: P(X <= 2 | Bin(10, 1/2)) = (1 + 10 + 45) / 1024.
    assert _binom_cdf_half(2, 10) == pytest.approx(56 / 1024, rel=1e-12)


def test_bh_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        benjamini_hochberg([[0.01, 0.02], [0.03, 0.04]])  # type: ignore[list-item]
