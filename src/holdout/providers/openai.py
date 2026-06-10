"""OpenAI provider (optional extra: ``pip install 'holdout[openai]'``)."""

from typing import Any

from holdout.core.target import Completion
from holdout.exceptions import MissingDependencyError
from holdout.providers.base import ModelProvider


class OpenAI(ModelProvider):
    """Evaluate against an OpenAI chat model.

    OpenAI accepts a ``seed`` parameter for best-effort determinism; holdout
    passes the run seed through and defaults temperature to 0.0.

    Parameters
    ----------
    model
        Model name (e.g. ``"gpt-4o-mini"``).
    api_key
        API key; falls back to the ``OPENAI_API_KEY`` environment variable.
    base_url
        Optional override for OpenAI-compatible endpoints.

    Other parameters are inherited from :class:`ModelProvider`.
    """

    provider_id = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        super().__init__(
            model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai") from exc
        # SDK retries are disabled: ModelProvider.generate is the single
        # retry authority, so behavior is identical across providers.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )
        self._base_url = base_url

    def _extra_config(self) -> dict[str, object]:
        """Include a non-default base_url in the fingerprint."""
        return {"base_url": self._base_url} if self._base_url else {}

    def _is_retryable(self, exc: Exception) -> bool:
        """Retry SDK connection errors, rate limits, and 5xx responses."""
        import openai

        if isinstance(exc, openai.APIConnectionError | openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", 0)
            return isinstance(status, int) and status >= 500
        return super()._is_retryable(exc)

    async def _generate_once(self, prompt: str, *, seed: int | None) -> Completion:
        """Make one chat-completions call."""
        messages: list[Any] = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
