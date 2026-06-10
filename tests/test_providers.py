"""Tests for holdout.providers: base, ollama, embeddings, static, and lazy imports."""

import importlib.util
import json
import subprocess
import sys

import httpx
import pytest

from holdout import providers
from holdout.core.target import Completion
from holdout.exceptions import MissingDependencyError, ProviderError
from holdout.providers.base import ModelProvider
from holdout.providers.embeddings import OllamaEmbeddings
from holdout.providers.ollama import Ollama
from holdout.providers.static import StaticTarget

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _chat_body(content: str = "ok") -> dict[str, object]:
    """A complete successful /api/chat response body."""
    return {
        "model": "llama3.2:latest",
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 7,
        "eval_count": 3,
    }


class EchoProvider(ModelProvider):
    """Minimal concrete provider for exercising ModelProvider itself."""

    provider_id = "echo"

    async def _generate_once(self, prompt: str, *, seed: int | None) -> Completion:
        del seed
        return Completion(text=prompt)


# ---------------------------------------------------------------------------
# Ollama: outgoing request shape
# ---------------------------------------------------------------------------


async def test_ollama_request_includes_system_and_seed() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_body())

    provider = Ollama(
        "llama3.2",
        system="be terse",
        temperature=0.25,
        max_tokens=64,
        transport=httpx.MockTransport(handler),
    )
    try:
        await provider.generate("hi", seed=42)
    finally:
        await provider.aclose()

    assert len(requests) == 1
    assert requests[0].url.path == "/api/chat"
    payload = json.loads(requests[0].content)
    assert payload == {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
        "options": {"temperature": 0.25, "num_predict": 64, "seed": 42},
    }


async def test_ollama_request_omits_system_and_seed_when_absent() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_chat_body())

    provider = Ollama("llama3.2", transport=httpx.MockTransport(handler))
    try:
        await provider.generate("hi")
    finally:
        await provider.aclose()

    payload = json.loads(requests[0].content)
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.0, "num_predict": 1024}
    assert "seed" not in payload["options"]


# ---------------------------------------------------------------------------
# Ollama: response parsing
# ---------------------------------------------------------------------------


async def test_ollama_parses_completion_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body("hello there"))

    provider = Ollama("llama3.2", transport=httpx.MockTransport(handler))
    try:
        completion = await provider.generate("hi")
    finally:
        await provider.aclose()

    assert completion == Completion(
        text="hello there",
        model="llama3.2:latest",
        input_tokens=7,
        output_tokens=3,
    )


async def test_ollama_completion_defaults_when_optional_fields_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})

    provider = Ollama("llama3.2", transport=httpx.MockTransport(handler))
    try:
        completion = await provider.generate("hi")
    finally:
        await provider.aclose()

    assert completion.text == "x"
    assert completion.model == "llama3.2"  # falls back to the configured model
    assert completion.input_tokens is None
    assert completion.output_tokens is None


# ---------------------------------------------------------------------------
# Ollama: retry behavior (backoff_base=0.0 so tests are instant)
# ---------------------------------------------------------------------------


async def test_ollama_retries_500_then_succeeds_in_two_attempts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=_chat_body("recovered"))

    provider = Ollama(
        "llama3.2",
        max_retries=3,
        backoff_base=0.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        completion = await provider.generate("hi")
    finally:
        await provider.aclose()

    assert completion.text == "recovered"
    assert len(requests) == 2


async def test_ollama_all_500_raises_provider_error_with_attempts_and_name() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={"error": "boom"})

    provider = Ollama(
        "llama3.2",
        max_retries=2,
        backoff_base=0.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ProviderError) as excinfo:
            await provider.generate("hi")
    finally:
        await provider.aclose()

    err = excinfo.value
    assert err.attempts == provider.max_retries + 1 == 3
    assert len(requests) == 3
    assert "ollama:llama3.2" in str(err)
    assert "3 attempt" in str(err)
    assert isinstance(err.cause, httpx.HTTPStatusError)


async def test_ollama_400_fails_immediately_without_retry() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400, json={"error": "bad request"})

    provider = Ollama(
        "llama3.2",
        max_retries=3,
        backoff_base=0.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ProviderError) as excinfo:
            await provider.generate("hi")
    finally:
        await provider.aclose()

    assert len(requests) == 1  # client errors are not retried
    cause = excinfo.value.cause
    assert isinstance(cause, httpx.HTTPStatusError)
    assert cause.response.status_code == 400


async def test_ollama_aclose_closes_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body())

    provider = Ollama("llama3.2", transport=httpx.MockTransport(handler))
    await provider.aclose()
    assert provider._client.is_closed


# ---------------------------------------------------------------------------
# OllamaEmbeddings
# ---------------------------------------------------------------------------


async def test_embeddings_request_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"embeddings": [[0.0], [1.0]]})

    backend = OllamaEmbeddings("nomic-embed-text", transport=httpx.MockTransport(handler))
    try:
        await backend.embed(("alpha", "beta"))  # a tuple must be sent as a JSON list
    finally:
        await backend.aclose()

    assert len(requests) == 1
    assert requests[0].url.path == "/api/embed"
    payload = json.loads(requests[0].content)
    assert payload == {"model": "nomic-embed-text", "input": ["alpha", "beta"]}


async def test_embeddings_parse_to_lists_of_floats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1, 2.5], [3, 4]]})

    backend = OllamaEmbeddings(transport=httpx.MockTransport(handler))
    try:
        vectors = await backend.embed(["a", "b"])
    finally:
        await backend.aclose()

    assert vectors == [[1.0, 2.5], [3.0, 4.0]]
    assert all(type(x) is float for row in vectors for x in row)


async def test_embeddings_name_property() -> None:
    default_backend = OllamaEmbeddings()
    custom_backend = OllamaEmbeddings("custom-model")
    try:
        assert default_backend.name == "ollama:nomic-embed-text"
        assert custom_backend.name == "ollama:custom-model"
    finally:
        await default_backend.aclose()
        await custom_backend.aclose()


# ---------------------------------------------------------------------------
# ModelProvider base behavior
# ---------------------------------------------------------------------------


def test_provider_name_is_provider_colon_model() -> None:
    assert EchoProvider("my-model").name == "echo:my-model"
    assert (
        Ollama("llama3.2", transport=httpx.MockTransport(lambda r: httpx.Response(200))).name
        == "ollama:llama3.2"
    )


def test_empty_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        EchoProvider("")


def test_fingerprint_is_stable_across_identical_instances() -> None:
    a = EchoProvider("m", system="s", temperature=0.2, max_tokens=10)
    b = EchoProvider("m", system="s", temperature=0.2, max_tokens=10)
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint == a.fingerprint  # property is deterministic


def test_fingerprint_changes_with_behavioral_settings() -> None:
    base = EchoProvider("m")
    variants = [
        EchoProvider("m2"),
        EchoProvider("m", system="you are terse"),
        EchoProvider("m", temperature=0.7),
        EchoProvider("m", max_tokens=2048),
    ]
    fingerprints = [base.fingerprint, *[v.fingerprint for v in variants]]
    assert len(set(fingerprints)) == len(fingerprints)


def test_fingerprint_ignores_operational_settings() -> None:
    a = EchoProvider("m", timeout=1.0, max_retries=0, backoff_base=0.0)
    b = EchoProvider("m", timeout=300.0, max_retries=9, backoff_base=4.0)
    assert a.fingerprint == b.fingerprint


# ---------------------------------------------------------------------------
# StaticTarget
# ---------------------------------------------------------------------------


async def test_static_target_returns_mapped_responses() -> None:
    target = StaticTarget({"q1": "a1", "q2": "a2"}, name="fixture")
    first = await target.generate("q1")
    second = await target.generate("q2")
    assert first == Completion(text="a1", model="fixture")
    assert second == Completion(text="a2", model="fixture")
    assert target.name == "fixture"


async def test_static_target_default_fallback() -> None:
    target = StaticTarget({"q": "a"}, default="dunno")
    result = await target.generate("unmapped")
    assert result.text == "dunno"


async def test_static_target_without_default_raises_key_error() -> None:
    target = StaticTarget({"q": "a"})
    with pytest.raises(KeyError, match="no static response"):
        await target.generate("unmapped")


def test_static_target_fingerprint_changes_with_mapping() -> None:
    a = StaticTarget({"q": "a"})
    b = StaticTarget({"q": "b"})
    c = StaticTarget({"q": "a"})
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == c.fingerprint


async def test_static_target_ignores_seed() -> None:
    target = StaticTarget({"q": "a"})
    assert await target.generate("q", seed=1) == await target.generate("q", seed=999)
    assert await target.generate("q") == await target.generate("q", seed=0)


# ---------------------------------------------------------------------------
# Lazy imports in holdout.providers
# ---------------------------------------------------------------------------


def test_lazy_import_exposes_sdk_free_providers() -> None:
    from holdout.providers import Ollama as LazyOllama
    from holdout.providers import StaticTarget as LazyStaticTarget

    assert LazyOllama is Ollama
    assert LazyStaticTarget is StaticTarget


def test_unknown_attribute_raises_attribute_error_naming_module() -> None:
    missing = "NoSuchProvider"
    with pytest.raises(AttributeError, match=r"holdout\.providers"):
        getattr(providers, missing)


def test_dir_includes_every_public_name() -> None:
    assert set(providers.__all__) <= set(dir(providers))


def test_importing_providers_does_not_import_sdk_modules() -> None:
    # Run in a fresh interpreter so this test cannot be polluted by other
    # tests (or pass vacuously because something imported the SDKs earlier).
    code = (
        "import sys\n"
        "import holdout.providers\n"
        "from holdout.providers import Ollama, StaticTarget\n"
        "leaked = [m for m in ('openai', 'anthropic') if m in sys.modules]\n"
        "if leaked:\n"
        "    raise SystemExit(f'SDK modules imported eagerly: {leaked}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# MissingDependencyError for optional SDK providers
# ---------------------------------------------------------------------------

_openai_installed = importlib.util.find_spec("openai") is not None
_anthropic_installed = importlib.util.find_spec("anthropic") is not None


@pytest.mark.skipif(
    _openai_installed, reason="openai SDK installed; missing-dependency path unreachable"
)
def test_openai_without_sdk_raises_missing_dependency() -> None:
    from holdout.providers import OpenAI

    with pytest.raises(MissingDependencyError) as excinfo:
        OpenAI("x")
    assert "pip install 'holdout[openai]'" in str(excinfo.value)
    assert excinfo.value.package == "openai"
    assert excinfo.value.extra == "openai"


@pytest.mark.skipif(
    _openai_installed, reason="openai SDK installed; missing-dependency path unreachable"
)
def test_openai_embeddings_without_sdk_raises_missing_dependency() -> None:
    from holdout.providers import OpenAIEmbeddings

    with pytest.raises(MissingDependencyError) as excinfo:
        OpenAIEmbeddings()
    assert "pip install 'holdout[openai]'" in str(excinfo.value)


@pytest.mark.skipif(
    _anthropic_installed, reason="anthropic SDK installed; missing-dependency path unreachable"
)
def test_anthropic_without_sdk_raises_missing_dependency() -> None:
    from holdout.providers import Anthropic

    with pytest.raises(MissingDependencyError) as excinfo:
        Anthropic("x")
    assert "pip install 'holdout[anthropic]'" in str(excinfo.value)
    assert excinfo.value.package == "anthropic"
    assert excinfo.value.extra == "anthropic"
