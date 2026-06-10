"""Model providers behind one Target protocol.

Providers are lazily imported so users only need the SDKs they actually
use: ``from holdout.providers import Anthropic`` never imports the OpenAI
SDK, and Ollama/StaticTarget need no SDK at all.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holdout.providers.anthropic import Anthropic
    from holdout.providers.base import ModelProvider
    from holdout.providers.embeddings import OllamaEmbeddings, OpenAIEmbeddings
    from holdout.providers.ollama import Ollama
    from holdout.providers.openai import OpenAI
    from holdout.providers.static import StaticTarget

__all__ = [
    "Anthropic",
    "ModelProvider",
    "Ollama",
    "OllamaEmbeddings",
    "OpenAI",
    "OpenAIEmbeddings",
    "StaticTarget",
]

_REGISTRY = {
    "Anthropic": "holdout.providers.anthropic",
    "ModelProvider": "holdout.providers.base",
    "Ollama": "holdout.providers.ollama",
    "OllamaEmbeddings": "holdout.providers.embeddings",
    "OpenAI": "holdout.providers.openai",
    "OpenAIEmbeddings": "holdout.providers.embeddings",
    "StaticTarget": "holdout.providers.static",
}


def __getattr__(name: str) -> object:
    """Lazily import provider classes on first attribute access."""
    try:
        module_name = _REGISTRY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """Expose lazy attributes to introspection."""
    return sorted(__all__)
