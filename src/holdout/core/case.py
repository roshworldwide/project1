"""The Case: one unit of evaluation — an input and an optional reference."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from holdout.core.hashing import fingerprint


@dataclass(frozen=True)
class Case:
    """A single evaluation case.

    Parameters
    ----------
    input
        The prompt/input sent to the target.
    reference
        The expected output, if the scorers need one (e.g. exact match).
    id
        Stable identifier used to pair this case across runs. If omitted,
        a content-derived id is assigned when the case joins an
        :class:`~holdout.core.evalset.Eval`.
    metadata
        Free-form string tags (e.g. ``{"category": "billing"}``).
    """

    input: str
    reference: str | None = None
    id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def content_id(self) -> str:
        """Return the content-derived id for this case."""
        payload = {
            "input": self.input,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }
        return "c" + fingerprint(payload)[:11]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "input": self.input,
            "reference": self.reference,
            "metadata": dict(self.metadata),
        }
