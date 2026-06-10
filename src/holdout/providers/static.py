"""A deterministic in-memory target for tests, examples, and dry runs."""

from collections.abc import Mapping

from holdout.core.hashing import fingerprint
from holdout.core.target import Completion


class StaticTarget:
    """A Target that answers from a fixed input->output mapping.

    Fully deterministic and offline — useful for testing eval plumbing,
    demonstrating the API without credentials, and verifying the
    determinism guarantee (same inputs => same run hash).

    Parameters
    ----------
    responses
        Mapping from exact input text to output text.
    name
        Display name (default ``"static"``).
    default
        Output for inputs not in the mapping; if ``None``, unknown inputs
        raise ``KeyError`` (which the runner records as a case error).
    """

    def __init__(
        self,
        responses: Mapping[str, str],
        *,
        name: str = "static",
        default: str | None = None,
    ) -> None:
        self._responses = dict(responses)
        self._name = name
        self._default = default

    @property
    def name(self) -> str:
        """Display name of the target."""
        return self._name

    @property
    def fingerprint(self) -> str:
        """Content hash of the full response mapping."""
        return fingerprint(
            {"static": self._name, "responses": self._responses, "default": self._default}
        )

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        """Look up ``prompt`` in the response mapping."""
        del seed  # deterministic by construction
        if prompt in self._responses:
            return Completion(text=self._responses[prompt], model=self._name)
        if self._default is not None:
            return Completion(text=self._default, model=self._name)
        raise KeyError(f"no static response for input: {prompt!r}")
