"""Leakage detection and holdout discipline.

Three ways an eval lies, three checks:

- :func:`check_contamination` — eval cases hiding in prompt/few-shot text;
- :func:`find_near_duplicates` — near-copies inflating the effective n;
- :class:`HoldoutLedger` — counting adaptive reuses of the same eval set
  (overfitting-to-eval, the silent killer).
"""

from holdout.leakage.contamination import (
    ContaminationFinding,
    ContaminationReport,
    check_contamination,
    check_contamination_embeddings,
)
from holdout.leakage.duplicates import DuplicatePair, find_near_duplicates
from holdout.leakage.ledger import DisciplineReport, HoldoutLedger

__all__ = [
    "ContaminationFinding",
    "ContaminationReport",
    "DisciplineReport",
    "DuplicatePair",
    "HoldoutLedger",
    "check_contamination",
    "check_contamination_embeddings",
    "find_near_duplicates",
]
