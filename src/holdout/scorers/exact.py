"""Exact-match scoring."""

import re
from collections.abc import Mapping

from holdout.core.case import Case
from holdout.core.scoring import Score, Scorer

_WHITESPACE = re.compile(r"\s+")


class ExactMatch(Scorer):
    """Binary scorer that passes when the output exactly matches the reference.

    Parameters
    ----------
    normalize
        If true (default), compare after stripping, casefolding, and
        collapsing internal whitespace — robust to formatting noise while
        staying strict on content.
    """

    requires_reference = True

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize

    @property
    def name(self) -> str:
        """Metric name: ``exact_match``."""
        return "exact_match"

    def config(self) -> Mapping[str, object]:
        """Return the scorer's configuration for fingerprinting."""
        return {"normalize": self._normalize}

    def _canon(self, text: str) -> str:
        if not self._normalize:
            return text
        return _WHITESPACE.sub(" ", text.strip().casefold())

    async def score(self, case: Case, output: str) -> Score:
        """Return 1.0 if ``output`` matches the reference, else 0.0."""
        assert case.reference is not None  # enforced by Eval validation
        matched = self._canon(output) == self._canon(case.reference)
        return Score(value=1.0 if matched else 0.0, kind="binary")
