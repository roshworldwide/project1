"""Paired significance tests — did B actually score differently from A, or is it noise.

All tests here are *paired*: they operate on per-case differences between
two runs over the same cases. Pairing removes between-case variance from
the comparison, which is why a paired test detects effects an unpaired
comparison of two aggregate scores cannot.

Sign convention throughout: differences are ``b - a``, so a positive
effect means ``b`` (conventionally the candidate) scored higher than ``a``
(the baseline).

References
----------
Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*,
ch. 16 (bootstrap hypothesis testing via the shifted null).

McNemar, Q. (1947). "Note on the sampling error of the difference between
correlated proportions or percentages". *Psychometrika*, 12(2), 153-157.

Fagerland, M. W., Lydersen, S. & Laake, P. (2013). "The McNemar test for
binary matched-pairs data: mid-p and asymptotic are better than exact
conditional". *BMC Medical Research Methodology*, 13:91. (We default to the
exact-conditional test — conservative, never anti-conservative.)

Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of
Hypotheses*, 3rd ed. Springer. (Sign-flip permutation test for paired data.)

Phipson, B. & Smyth, G. K. (2010). "Permutation p-values should never be
zero". *Statistical Applications in Genetics and Molecular Biology*, 9(1).
(The (r+1)/(B+1) estimator used for all Monte-Carlo p-values here.)
"""

import math
from collections.abc import Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray

from holdout.stats.bootstrap import bootstrap_ci
from holdout.stats.result import TestResult


def paired_diffs(scores_a: Sequence[float], scores_b: Sequence[float]) -> NDArray[np.float64]:
    """Validate two paired score sequences and return ``b - a`` differences.

    Raises
    ------
    ValueError
        If the sequences differ in length, have fewer than 2 pairs, or
        contain NaN or infinity.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("paired scores must be one-dimensional sequences")
    if a.size != b.size:
        raise ValueError(f"paired scores must have equal length, got {a.size} and {b.size}")
    if a.size < 2:
        raise ValueError(f"paired tests need at least 2 pairs, got {a.size}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores contain NaN or infinity")
    return b - a


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TestResult:
    """Paired bootstrap test for a difference in means.

    Computes the observed mean difference, then bootstraps the differences
    *centered under H0* (shifted to mean zero; Efron & Tibshirani 1993,
    ch. 16) and reports the two-sided Monte-Carlo p-value with the
    Phipson-Smyth (r+1)/(B+1) estimator, so a p-value of exactly zero is
    impossible. The CI on the effect is the BCa interval on the raw
    differences.

    Parameters
    ----------
    scores_a, scores_b
        Per-case scores, paired by position (``b - a`` is the effect).
    level
        Confidence level for the effect's CI.
    n_resamples
        Bootstrap resamples for both the p-value and the CI.
    seed
        RNG seed; identical inputs + seed => identical result.
    """
    d = paired_diffs(scores_a, scores_b)
    n = int(d.size)
    effect = float(d.mean())
    ci = bootstrap_ci(d, level=level, n_resamples=n_resamples, seed=seed)

    rng = np.random.default_rng([seed, 0x01])
    centered = d - d.mean()
    indices = rng.integers(0, n, size=(n_resamples, n))
    null_means = centered[indices].mean(axis=1)
    extreme = int((np.abs(null_means) >= abs(effect)).sum())
    p = (extreme + 1) / (n_resamples + 1)

    return TestResult(
        test="paired-bootstrap",
        p_value=min(1.0, float(p)),
        effect=effect,
        ci=ci,
        n=n,
        detail="H0: mean(b - a) = 0, two-sided, shifted-null bootstrap",
    )


def _binom_cdf_half(k: int, m: int) -> float:
    """P(X <= k) for X ~ Binomial(m, 1/2), via log-space terms (no SciPy)."""
    if k < 0:
        return 0.0
    if k >= m:
        return 1.0
    log_half_m = m * math.log(0.5)
    total = 0.0
    for i in range(k + 1):
        log_term = math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) + log_half_m
        total += math.exp(log_term)
    return min(1.0, total)


def mcnemar_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TestResult:
    """Exact McNemar test for paired binary outcomes.

    Only discordant pairs carry information about a difference (McNemar
    1947): cases where exactly one of the two runs scored 1. Under H0 the
    ``n01`` improvements and ``n10`` regressions are Binomial(m, 1/2) among
    the ``m = n01 + n10`` discordant pairs; the exact-conditional two-sided
    p-value is ``min(1, 2 * P(X <= min(n01, n10)))``. The exact test is
    conservative but never anti-conservative (Fagerland et al. 2013).

    The effect is the difference in proportions, ``mean(b) - mean(a) =
    (n01 - n10) / n``, with a BCa bootstrap CI on the per-pair differences.

    Parameters
    ----------
    scores_a, scores_b
        Per-case binary scores (each value 0.0 or 1.0), paired by position.
    level, n_resamples, seed
        Govern the effect's bootstrap CI (the p-value is exact, not
        resampled).

    Raises
    ------
    ValueError
        If any score is not 0.0 or 1.0.
    """
    d = paired_diffs(scores_a, scores_b)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    for name, arr in (("scores_a", a), ("scores_b", b)):
        if not np.isin(arr, (0.0, 1.0)).all():
            raise ValueError(f"mcnemar_test requires binary scores; {name} has other values")

    n01 = int(((a == 0.0) & (b == 1.0)).sum())  # b improved the case
    n10 = int(((a == 1.0) & (b == 0.0)).sum())  # b regressed the case
    m = n01 + n10
    # No discordant pairs => no information about a difference => p = 1.
    p = 1.0 if m == 0 else min(1.0, 2.0 * _binom_cdf_half(min(n01, n10), m))

    ci = bootstrap_ci(d, level=level, n_resamples=n_resamples, seed=seed)
    return TestResult(
        test="mcnemar-exact",
        p_value=p,
        effect=float(d.mean()),
        ci=ci,
        n=int(d.size),
        detail=f"discordant pairs: improved={n01}, regressed={n10}",
    )


def permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> TestResult:
    """Sign-flip permutation test for paired data.

    Under H0 the two runs are exchangeable within each pair, so each
    difference's sign is a fair coin (Good 2005; Fisher's randomization
    argument). When the full ``2^n`` sign assignments number at most
    ``n_resamples`` the test enumerates them exactly (the p-value includes
    the observed assignment, so it can never be zero); otherwise it samples
    sign flips and uses the Phipson-Smyth (r+1)/(B+1) estimator.

    Parameters
    ----------
    scores_a, scores_b
        Per-case scores, paired by position (``b - a`` is the effect).
    level
        Confidence level for the effect's BCa bootstrap CI.
    n_resamples
        Monte-Carlo budget; also the threshold below which the exact
        enumeration is used.
    seed
        RNG seed for the Monte-Carlo branch; identical inputs + seed =>
        identical result.
    """
    d = paired_diffs(scores_a, scores_b)
    n = int(d.size)
    effect = float(d.mean())
    ci = bootstrap_ci(d, level=level, n_resamples=n_resamples, seed=seed)
    # Tolerance guards float-association noise when comparing permuted means
    # to the observed mean (the all-flipped assignment is exactly -effect).
    tol = 1e-12 * max(1.0, abs(effect))

    if 2**n <= n_resamples:
        signs = np.asarray(list(product((1.0, -1.0), repeat=n)), dtype=np.float64)
        perm_means = (signs * d).mean(axis=1)
        p = float((np.abs(perm_means) >= abs(effect) - tol).mean())
        return TestResult(
            test="permutation-exact",
            p_value=min(1.0, p),
            effect=effect,
            ci=ci,
            n=n,
            detail=f"exact enumeration of 2^{n} sign assignments",
        )

    rng = np.random.default_rng([seed, 0x02])
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_resamples, n))
    perm_means = (signs * d).mean(axis=1)
    extreme = int((np.abs(perm_means) >= abs(effect) - tol).sum())
    p = (extreme + 1) / (n_resamples + 1)
    return TestResult(
        test="permutation-mc",
        p_value=min(1.0, float(p)),
        effect=effect,
        ci=ci,
        n=n,
        detail=f"{n_resamples} Monte-Carlo sign flips",
    )
