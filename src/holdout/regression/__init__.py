"""The regression engine: compares two Runs and issues a statistical verdict."""

from holdout.regression.compare import (
    Correction,
    MetricComparison,
    PairedTest,
    RunComparison,
    Verdict,
    compare,
)

__all__ = [
    "Correction",
    "MetricComparison",
    "PairedTest",
    "RunComparison",
    "Verdict",
    "compare",
]
