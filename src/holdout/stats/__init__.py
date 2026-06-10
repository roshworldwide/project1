"""Statistical machinery for holdout.

This package is the reason the product exists: bootstrap confidence
intervals, paired significance tests, multiple-comparison correction, and
power analysis — built to quant-library standards, with citations in
docstrings.
"""

from holdout.stats.bootstrap import bootstrap_ci
from holdout.stats.estimate import Estimate

__all__ = ["Estimate", "bootstrap_ci"]
