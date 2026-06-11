"""Reference resolution for the CLI: evals, targets, and scorers from strings.

Two reference grammars:

- **Targets** — ``provider:model`` shorthand (``ollama:llama3.2``,
  ``openai:gpt-4o-mini``, ``anthropic:claude-sonnet-4-6``) or a Python
  reference ``package.module:attribute`` pointing at any Target object.
- **Evals** — a ``.jsonl`` path (scorers supplied via ``--scorer``) or a
  Python reference pointing at an Eval object.
"""

import importlib
from pathlib import Path

from holdout.core.evalset import Eval
from holdout.core.scoring import Scorer
from holdout.core.target import Target
from holdout.scorers.exact import ExactMatch
from holdout.scorers.regex_match import RegexMatch

_PROVIDERS = ("ollama", "openai", "anthropic")


def load_python_ref(ref: str) -> object:
    """Import ``package.module:attribute`` and return the attribute."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"invalid Python reference {ref!r}; expected 'package.module:attribute'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import module {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"module {module_name!r} has no attribute {attr!r}") from exc


def make_scorer(spec: str) -> Scorer:
    """Build a scorer from a CLI spec: ``exact``, ``exact-strict``, ``regex:<pattern>``."""
    if spec == "exact":
        return ExactMatch()
    if spec == "exact-strict":
        return ExactMatch(normalize=False)
    if spec.startswith("regex:"):
        pattern = spec[len("regex:") :]
        if not pattern:
            raise ValueError("regex scorer needs a pattern: regex:<pattern>")
        return RegexMatch(pattern)
    raise ValueError(
        f"unknown scorer spec {spec!r}; expected 'exact', 'exact-strict', or 'regex:<pattern>' "
        "(for embedding or custom scorers, point --eval at a Python reference instead)"
    )


def load_eval(ref: str, scorer_specs: list[str] | None = None) -> Eval:
    """Resolve an eval reference: a ``.jsonl`` path or ``module:attr``."""
    if ref.endswith(".jsonl"):
        path = Path(ref)
        if not path.exists():
            raise ValueError(f"eval file not found: {path}")
        scorers = [make_scorer(s) for s in (scorer_specs or ["exact"])]
        return Eval.from_jsonl(path, scorers=scorers)
    obj = load_python_ref(ref)
    if not isinstance(obj, Eval):
        raise ValueError(f"{ref!r} resolved to {type(obj).__name__}, not an Eval")
    return obj


def load_target(
    ref: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    base_url: str | None = None,
) -> Target:
    """Resolve a target reference: ``provider:model`` shorthand or ``module:attr``.

    Provider options (``system``, ``temperature``, ...) apply only to the
    shorthand form; a Python reference is returned as-is.
    """
    prefix, sep, model = ref.partition(":")
    if sep and prefix in _PROVIDERS:
        if prefix == "ollama":
            from holdout.providers.ollama import Ollama

            kwargs: dict[str, object] = {}
            if base_url is not None:
                kwargs["base_url"] = base_url
            return Ollama(
                model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,  # type: ignore[arg-type]
            )
        if prefix == "openai":
            from holdout.providers.openai import OpenAI

            return OpenAI(
                model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
            )
        from holdout.providers.anthropic import Anthropic

        return Anthropic(model, system=system, temperature=temperature, max_tokens=max_tokens)

    obj = load_python_ref(ref)
    if not isinstance(obj, Target):
        raise ValueError(f"{ref!r} resolved to {type(obj).__name__}, which is not a Target")
    return obj
