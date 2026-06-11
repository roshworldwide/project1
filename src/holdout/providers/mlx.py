"""Apple MLX provider — fully local inference on Apple silicon.

Zero bytes leave the machine: the model runs in-process via ``mlx-lm``.
Install with ``pip install 'holdout[mlx]'`` (Apple silicon only) and point
at any MLX-community model (e.g. ``mlx-community/Llama-3.2-3B-Instruct-4bit``).
"""

import asyncio
from typing import Any

from holdout.core.target import Completion
from holdout.exceptions import MissingDependencyError
from holdout.providers.base import ModelProvider


class MLX(ModelProvider):
    """Evaluate against a local MLX model (air-gapped, in-process).

    The model loads lazily on the first generation and stays resident.
    MLX seeds its sampler from ``seed`` for reproducibility; at the default
    temperature 0.0 decoding is greedy and deterministic regardless.

    Parameters
    ----------
    model
        MLX model path or Hugging Face repo id (cached locally after the
        first load; fully offline thereafter).

    Other parameters are inherited from :class:`ModelProvider`.
    """

    provider_id = "mlx"

    def __init__(
        self,
        model: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 600.0,
        max_retries: int = 0,
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
            import mlx_lm  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError("mlx-lm", "mlx") from exc
        self._loaded: tuple[Any, Any] | None = None
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> tuple[Any, Any]:
        """Load the model once, guarded against concurrent first calls."""
        async with self._load_lock:
            if self._loaded is None:
                from mlx_lm import load

                loaded = await asyncio.to_thread(load, self.model)
                self._loaded = (loaded[0], loaded[1])
            return self._loaded

    def _generate_sync(self, model: Any, tokenizer: Any, prompt: str, seed: int | None) -> str:
        """Run one greedy/sampled generation on the calling thread."""
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        if seed is not None:
            mx.random.seed(seed)
        messages = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        if getattr(tokenizer, "chat_template", None) is not None:
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prefix = f"{self.system}\n\n" if self.system is not None else ""
            prompt_text = prefix + prompt
        sampler = make_sampler(temp=self.temperature)
        text = generate(
            model,
            tokenizer,
            prompt=prompt_text,
            max_tokens=self.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return str(text)

    async def _generate_once(self, prompt: str, *, seed: int | None) -> Completion:
        """Generate off the event loop (MLX generation is synchronous)."""
        model, tokenizer = await self._ensure_loaded()
        text = await asyncio.to_thread(self._generate_sync, model, tokenizer, prompt, seed)
        return Completion(text=text, model=self.model)
