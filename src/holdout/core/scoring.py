"""Scores and the Scorer contract.

A Scorer maps (case, model output) to a :class:`Score`. Scores carry their
kind — ``"binary"`` or ``"continuous"`` — because the statistics engine
chooses its test accordingly (McNemar for paired binary outcomes, paired
bootstrap for continuous metrics).
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from holdout.core.hashing import fingerprint

if TYPE_CHECKING:
    from holdout.core.case import Case

ScoreKind = Literal["binary", "continuous"]


@dataclass(frozen=True, slots=True)
class Score:
    """The result of scoring one case.

    Parameters
    ----------
    value
        The score. Binary scores must be 0.0 or 1.0; continuous scores are
        conventionally in [0, 1].
    kind
        ``"binary"`` or ``"continuous"`` — determines which paired test the
        statistics engine applies.
    detail
        Optional human-readable context (e.g. the cosine similarity behind
        a thresholded pass/fail).
    """

    value: float
    kind: ScoreKind
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "binary" and self.value not in (0.0, 1.0):
            raise ValueError(f"binary scores must be 0.0 or 1.0, got {self.value}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {"value": self.value, "kind": self.kind, "detail": self.detail}


class Scorer(ABC):
    """Base class for scorers.

    Subclasses implement :meth:`score` and may override :meth:`config` so
    their configuration participates in run fingerprints (determinism: a
    changed threshold is a changed measurement).
    """

    requires_reference: ClassVar[bool] = False
    """Whether every case must carry a ``reference`` for this scorer."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short snake_case identifier, used as the metric name."""

    def config(self) -> Mapping[str, object]:
        """Return the scorer's configuration for fingerprinting."""
        return {}

    @property
    def fingerprint(self) -> str:
        """Content hash of the scorer's identity and configuration."""
        return fingerprint({"scorer": self.name, "config": dict(self.config())})

    @abstractmethod
    async def score(self, case: "Case", output: str) -> Score:
        """Score the target's ``output`` for ``case``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self.config())!r})"
