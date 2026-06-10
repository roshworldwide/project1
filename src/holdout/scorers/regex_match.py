"""Regex scoring."""

import re
from collections.abc import Mapping

from holdout.core.case import Case
from holdout.core.scoring import Score, Scorer


class RegexMatch(Scorer):
    r"""Binary scorer that passes when the output matches a regular expression.

    Useful for format contracts — "the answer contains a dollar amount",
    "the output is valid ISO-8601" — that need no reference text.

    Parameters
    ----------
    pattern
        The regular expression, searched anywhere in the output
        (anchor with ``\A``/``\Z`` for full-match semantics).
    ignore_case
        Case-insensitive matching (default false).
    """

    def __init__(self, pattern: str, *, ignore_case: bool = False) -> None:
        flags = re.IGNORECASE if ignore_case else 0
        self._pattern = re.compile(pattern, flags)
        self._ignore_case = ignore_case

    @property
    def name(self) -> str:
        """Metric name: ``regex_match``."""
        return "regex_match"

    def config(self) -> Mapping[str, object]:
        """Return the scorer's configuration for fingerprinting."""
        return {"pattern": self._pattern.pattern, "ignore_case": self._ignore_case}

    async def score(self, case: Case, output: str) -> Score:
        """Return 1.0 if the pattern matches anywhere in ``output``."""
        del case  # scorer interface; regex needs no reference
        matched = self._pattern.search(output) is not None
        return Score(
            value=1.0 if matched else 0.0,
            kind="binary",
            detail=f"pattern={self._pattern.pattern!r}",
        )
