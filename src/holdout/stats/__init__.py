"""Statistical machinery for holdout.

This package is the reason the product exists: bootstrap confidence
intervals (BCa by default), paired significance tests, multiple-comparison
correction, and power analysis — built to quant-library standards, with
citations in docstrings.
"""

from holdout.stats.bootstrap import bootstrap_ci
from holdout.stats.correction import benjamini_hochberg, holm_bonferroni
from holdout.stats.estimate import Estimate
from holdout.stats.paired import (
    mcnemar_test,
    paired_bootstrap_test,
    paired_diffs,
    permutation_test,
)
from holdout.stats.power import (
    PowerAnalysis,
    minimum_detectable_effect,
    paired_binary_sd,
    required_sample_size,
    sd_diff_from_scores,
)
from holdout.stats.result import TestResult

__all__ = [
    "Estimate",
    "PowerAnalysis",
    "TestResult",
    "benjamini_hochberg",
    "bootstrap_ci",
    "holm_bonferroni",
    "mcnemar_test",
    "minimum_detectable_effect",
    "paired_binary_sd",
    "paired_bootstrap_test",
    "paired_diffs",
    "permutation_test",
    "required_sample_size",
    "sd_diff_from_scores",
]
