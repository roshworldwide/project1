"""Anthropic provider (optional extra: ``pip install 'holdout[anthropic]'``)."""

from holdout.core.target import Completion
from holdout.exceptions import MissingDependencyError
from holdout.providers.base import ModelProvider


class Anthropic(ModelProvider):
    """Evaluate against an Anthropic Claude model.

    The Anthropic API has no seed parameter; determinism is best-effort at
    temperature 0.0 (the default). The run seed still participates in the
    run hash, so reruns are honestly distinguishable.

    Parameters
    ----------
    model
        Model name (e.g. ``"claude-sonnet-4-6"``).
    api_key
        API key; falls back to the ``ANTHROPIC_API_KEY`` environment variable.

    Other parameters are inherited from :class:`ModelProvider`.
    """

    provider_id = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
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
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise MissingDependencyError("anthropic", "anthropic") from exc
        # SDK retries are disabled: ModelProvider.generate is the single
        # retry authority, so behavior is identical across providers.
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=0)

    def _is_retryable(self, exc: Exception) -> bool:
        """Retry SDK connection errors, rate limits, and 5xx responses."""
        import anthropic

        if isinstance(exc, anthropic.APIConnectionError | anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", 0)
            return isinstance(status, int) and status >= 500
        return super()._is_retryable(exc)

    async def _generate_once(self, prompt: str, *, seed: int | None) -> Completion:
        """Make one messages call. ``seed`` is unsupported by the API and ignored."""
        del seed  # no seed parameter in the Anthropic API; see class docstring
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system is not None:
            kwargs["system"] = self.system
        resp = await self._client.messages.create(**kwargs)
        text = "".join(block.text for block in resp.content if block.type == "text")
        return Completion(
            text=text,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
