"""The provider base class: retry discipline and fingerprinting in one place."""

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import httpx

from holdout.core.hashing import fingerprint
from holdout.core.target import Completion
from holdout.exceptions import ProviderError


class ModelProvider(ABC):
    """Base class for model providers; satisfies the :class:`~holdout.core.target.Target` protocol.

    Subclasses implement :meth:`_generate_once`; this class owns retries
    (exponential backoff with jitter on transient failures) and the
    fingerprint, so every provider behaves identically under failure and
    participates identically in run identity.

    Parameters
    ----------
    model
        Backend model identifier.
    system
        Optional system prompt — part of the target's fingerprint.
    temperature
        Sampling temperature; defaults to 0.0 for determinism.
    max_tokens
        Maximum tokens to generate.
    timeout
        Per-request timeout in seconds.
    max_retries
        Additional attempts after the first failure (transient errors only).
    backoff_base
        Base of the exponential backoff schedule, in seconds.
    """

    provider_id: ClassVar[str]
    """Short backend identifier (e.g. ``"openai"``); set by each subclass."""

    def __init__(
        self,
        model: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        self.model = model
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    @property
    def name(self) -> str:
        """Human-readable identifier, ``"<provider>:<model>"``."""
        return f"{self.provider_id}:{self.model}"

    def config(self) -> Mapping[str, object]:
        """Return everything that defines this target's behavior.

        Subclasses extend via :meth:`_extra_config`. Operational settings
        (timeout, retries) are excluded — they cannot change outputs.
        """
        return {
            "provider": self.provider_id,
            "model": self.model,
            "system": self.system,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self._extra_config(),
        }

    def _extra_config(self) -> Mapping[str, object]:
        """Provider-specific config merged into :meth:`config`."""
        return {}

    @property
    def fingerprint(self) -> str:
        """Content hash of :meth:`config`."""
        return fingerprint(dict(self.config()))

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        """Generate a completion, retrying transient failures with backoff."""
        max_attempts = self.max_retries + 1
        last_exc: Exception | None = None
        attempts_made = 0
        for attempt in range(max_attempts):
            attempts_made = attempt + 1
            try:
                return await self._generate_once(prompt, seed=seed)
            except Exception as exc:
                last_exc = exc
                if attempts_made >= max_attempts or not self._is_retryable(exc):
                    break
                delay = min(self.backoff_base * 2**attempt, 8.0)
                delay += random.random() * 0.1 * delay  # jitter; never affects outputs
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise ProviderError(self.name, attempts_made, last_exc) from last_exc

    def _is_retryable(self, exc: Exception) -> bool:
        """Whether ``exc`` is transient. Subclasses extend for SDK errors."""
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status == 429 or status >= 500
        return False

    @abstractmethod
    async def _generate_once(self, prompt: str, *, seed: int | None) -> Completion:
        """Make exactly one backend call. Implemented by each provider."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
