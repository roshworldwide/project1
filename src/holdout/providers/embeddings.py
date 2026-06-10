"""Embedding backends for the embedding-similarity scorer."""

from collections.abc import Sequence

import httpx

from holdout.exceptions import MissingDependencyError


class OllamaEmbeddings:
    """Local embeddings via Ollama — fully air-gapped.

    Parameters
    ----------
    model
        Embedding model name (default ``"nomic-embed-text"``).
    base_url
        Ollama server URL; defaults to the local daemon.
    transport
        Optional httpx transport override (used by tests).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    @property
    def name(self) -> str:
        """Backend identifier used in scorer fingerprints."""
        return f"ollama:{self.model}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` via the Ollama /api/embed endpoint."""
        resp = await self._client.post(
            "/api/embed", json={"model": self.model, "input": list(texts)}
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        return [[float(x) for x in vec] for vec in embeddings]

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


class OpenAIEmbeddings:
    """Embeddings via the OpenAI API (optional extra: ``holdout[openai]``).

    Parameters
    ----------
    model
        Embedding model name (default ``"text-embedding-3-small"``).
    api_key
        API key; falls back to the ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise MissingDependencyError("openai", "openai") from exc
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    @property
    def name(self) -> str:
        """Backend identifier used in scorer fingerprints."""
        return f"openai:{self.model}"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` via the OpenAI embeddings endpoint."""
        resp = await self._client.embeddings.create(model=self.model, input=list(texts))
        return [[float(x) for x in item.embedding] for item in resp.data]
