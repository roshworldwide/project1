"""holdout — quant-grade LLM evaluation.

Confidence intervals, significance tests, and regression gates — not vanity
numbers. Every metric this library reports carries its uncertainty; there is
no public API that returns a naked point estimate.
"""

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.run import CaseResult, Run
from holdout.core.runner import arun, run
from holdout.core.scoring import Score, Scorer
from holdout.core.target import Completion, Target
from holdout.stats.estimate import Estimate

__version__ = "1.1.0"

__all__ = [
    "Case",
    "CaseResult",
    "Completion",
    "Estimate",
    "Eval",
    "Run",
    "Score",
    "Scorer",
    "Target",
    "__version__",
    "arun",
    "run",
]
