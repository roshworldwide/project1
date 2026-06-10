"""The Target contract: anything that turns a prompt into a completion.

Providers (OpenAI, Anthropic, Ollama, MLX) implement this protocol, but so
can any user object — a RAG pipeline, an agent, a function — as long as it
exposes a stable fingerprint and an async ``generate``.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Completion:
    """One model completion.

    Parameters
    ----------
    text
        The completion text.
    model
        The concrete model that produced it, if known.
    input_tokens, output_tokens
        Token usage, if the backend reports it.
    """

    text: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class Target(Protocol):
    """A system under evaluation."""

    @property
    def name(self) -> str:
        """Human-readable identifier (e.g. ``"anthropic:claude-sonnet-4-6"``)."""
        ...

    @property
    def fingerprint(self) -> str:
        """Content hash of everything that defines the target's behavior.

        Model, system prompt, temperature, decoding parameters — if a change
        could change outputs, it must change the fingerprint.
        """
        ...

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        """Generate a completion for ``prompt``.

        Implementations should honor ``seed`` where the backend supports it
        and document when determinism is best-effort only.
        """
        ...
