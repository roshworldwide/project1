"""Multiple-comparison correction.

When an eval reports several metrics (or a suite runs several evals), each
extra comparison is an extra chance for a fluke "significant" result. These
procedures adjust p-values for that; the regression engine applies
Benjamini-Hochberg by default and gates on the *adjusted* values.

References
----------
Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery
Rate: A Practical and Powerful Approach to Multiple Testing". *Journal of
the Royal Statistical Society, Series B*, 57(1), 289-300.

Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test
Procedure". *Scandinavian Journal of Statistics*, 6(2), 65-70.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _validate_p_values(p_values: Sequence[float]) -> NDArray[np.float64]:
    """Convert to an array, rejecting values outside [0, 1] or NaN."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional, got shape {p.shape}")
    if p.size and (np.isnan(p).any() or (p < 0.0).any() or (p > 1.0).any()):
        raise ValueError("p_values must all be in [0, 1]")
    return p


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), controlling FDR.

    The step-up procedure (Benjamini & Hochberg 1995): sort the m p-values
    ascending, compute ``p_(i) * m / i``, then enforce monotonicity by a
    cumulative minimum from the largest rank down. Rejecting all hypotheses
    with adjusted p <= alpha controls the false discovery rate at alpha
    (under independence or positive regression dependence).

    Parameters
    ----------
    p_values
        Raw two-sided p-values from the individual tests.

    Returns
    -------
    list[float]
        Adjusted p-values, in the original order.
    """
    p = _validate_p_values(p_values)
    m = int(p.size)
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    # Multiply by (m / i) rather than computing (p * m) / i: at the largest
    # rank m / i is exactly 1.0, so q == p to the last ulp (q >= p must hold).
    q = ranked * (m / np.arange(1, m + 1, dtype=np.float64))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = q
    return [float(x) for x in out]


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, controlling family-wise error.

    The step-down procedure (Holm 1979): sort ascending, compute
    ``(m - i + 1) * p_(i)``, enforce monotonicity by a cumulative maximum
    from the smallest rank up, cap at 1. Stricter than Benjamini-Hochberg;
    use when any single false alarm is unacceptable.

    Parameters
    ----------
    p_values
        Raw two-sided p-values from the individual tests.

    Returns
    -------
    list[float]
        Adjusted p-values, in the original order.
    """
    p = _validate_p_values(p_values)
    m = int(p.size)
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    q = (m - np.arange(m, dtype=np.float64)) * ranked
    q = np.maximum.accumulate(q)
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(m, dtype=np.float64)
    out[order] = q
    return [float(x) for x in out]
