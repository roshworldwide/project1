# Providers and targets

holdout evaluates *targets*. A target is not necessarily a model API — it is
anything that satisfies the `Target` protocol: a provider, a RAG pipeline, an
agent, a function with a prompt in front of it.

## The Target protocol

```python
from typing import Protocol
from holdout import Completion

class Target(Protocol):
    @property
    def name(self) -> str: ...            # e.g. "anthropic:claude-sonnet-4-6"

    @property
    def fingerprint(self) -> str: ...     # content hash of everything that defines behavior

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion: ...
```

Three obligations:

- **`name`** — a human-readable identifier for summaries and listings.
- **`fingerprint`** — a content hash of everything that could change outputs:
  model, system prompt, temperature, decoding parameters, retrieval index
  version. If a change could change outputs, it must change the fingerprint;
  the fingerprint is part of every run's identity, so two runs are comparable
  as "same target" only when they provably were.
- **`generate`** — async, returns a `Completion(text, model, input_tokens,
  output_tokens)`. Honor `seed` where the backend supports it; document when
  determinism is best-effort.

A minimal custom target:

```python
from holdout import Completion
from holdout.core.hashing import fingerprint

class MyPipeline:
    def __init__(self, prompt_version: str) -> None:
        self.prompt_version = prompt_version

    @property
    def name(self) -> str:
        return f"my-pipeline:{self.prompt_version}"

    @property
    def fingerprint(self) -> str:
        return fingerprint({"pipeline": "my-pipeline", "prompt": self.prompt_version})

    async def generate(self, prompt: str, *, seed: int | None = None) -> Completion:
        return Completion(text=await my_rag_answer(prompt))
```

Providers that subclass `ModelProvider` get the rest for free: exponential
backoff with jitter on transient failures (a single retry authority, identical
across providers — SDK-level retries are disabled), and a fingerprint computed
from `config()` (model, system, temperature, max_tokens; operational settings
like timeouts are excluded because they cannot change outputs).

## Built-in providers

All providers default to `temperature=0.0`. They are **lazily imported**:
`from holdout.providers import Anthropic` never imports the OpenAI SDK, and a
missing SDK raises a `MissingDependencyError` that names the extra to install.

| Provider | Install | Network | Honors `seed`? |
|---|---|---|---|
| `Anthropic` | `pip install 'holdout[anthropic]'` | Anthropic API | no — the API has no seed parameter; best-effort determinism at temperature 0 |
| `OpenAI` | `pip install 'holdout[openai]'` | OpenAI API (or any compatible `base_url`) | yes — passed through; backend treats it as best-effort |
| `Ollama` | none (plain HTTP) | localhost only | yes — runs at temperature 0 with a fixed seed are reproducible |
| `MLX` | `pip install 'holdout[mlx]'` (Apple silicon) | none — in-process | yes — seeds the sampler; greedy and deterministic at temperature 0 regardless |
| `StaticTarget` | none | none | n/a — deterministic by construction |

### Anthropic

```python
from holdout.providers import Anthropic

target = Anthropic(
    model="claude-sonnet-4-6",
    system="Answer in one sentence.",   # part of the fingerprint
    temperature=0.0,
    max_tokens=1024,
)
```

API key from the `api_key` argument or `ANTHROPIC_API_KEY`. The Anthropic API
has no seed parameter, so determinism is best-effort at temperature 0.0. The
run seed still participates in the run hash, so reruns are honestly
distinguishable rather than falsely identical.

### OpenAI

```python
from holdout.providers import OpenAI

target = OpenAI(model="gpt-4o-mini", system="Answer in one sentence.")
```

API key from `api_key` or `OPENAI_API_KEY`. The run seed is passed to the API's
`seed` parameter. `base_url` points at any OpenAI-compatible endpoint and, when
set, joins the fingerprint (a different server can mean different weights).

### Ollama

```python
from holdout.providers import Ollama

target = Ollama("llama3.2", system="Answer in one sentence.")
# base_url defaults to http://localhost:11434
```

Talks to the local Ollama daemon over plain HTTP — no SDK, no extra to
install. Ollama honors `seed` in its decoding options, so a fixed seed at
temperature 0.0 is reproducible. The server URL is part of the fingerprint:
a different server may serve a different model file.

### MLX (Apple silicon)

```python
from holdout.providers import MLX

target = MLX("mlx-community/Llama-3.2-3B-Instruct-4bit")
```

Runs the model in-process via `mlx-lm`. The model loads lazily on the first
generation and stays resident; the Hugging Face download is cached locally and
the provider is fully offline thereafter. `seed` seeds the sampler; at the
default temperature 0.0 decoding is greedy and deterministic regardless.

### StaticTarget

```python
from holdout.providers import StaticTarget

target = StaticTarget(
    {"What is 2 + 2?": "4"},
    name="canned-v1",
    default=None,   # unknown inputs raise KeyError -> recorded as a case error
)
```

A fixed input-to-output mapping: fully deterministic, fully offline. Use it to
test eval plumbing, write runnable documentation (every offline example in
these docs uses it), and verify the determinism guarantee — same inputs, same
run hash, every time.

## Embedding backends

The `EmbeddingSimilarity` scorer and the embedding contamination pass take an
`EmbeddingBackend` — anything with a `name` and an async
`embed(texts) -> list[list[float]]`:

```python
from holdout.providers import OllamaEmbeddings   # local, no extra
from holdout.providers import OpenAIEmbeddings   # requires holdout[openai]
from holdout.scorers import EmbeddingSimilarity

scorer = EmbeddingSimilarity(OllamaEmbeddings("nomic-embed-text"), threshold=0.85)
```

With a threshold the scorer reports binary pass/fail (so the engine can run
McNemar) while surfacing the underlying cosine in the score's detail; without
one it reports the raw similarity as a continuous metric.

## Air-gapped / local-first

Local-first is a design invariant, not a fallback mode. A complete holdout
workflow with **zero bytes leaving the machine**:

- **Generation** — `Ollama` (local daemon, plain HTTP to localhost) or `MLX`
  (in-process on Apple silicon). No API keys, no cloud.
- **Scoring** — `ExactMatch` and `RegexMatch` are pure computation;
  `EmbeddingSimilarity` runs locally via `OllamaEmbeddings`.
- **Leakage auditing** — the n-gram contamination and duplicate checks are
  offline by construction; the embedding pass stays local with
  `OllamaEmbeddings`.
- **Statistics** — numpy on your machine. Nothing phones home; there is no
  telemetry anywhere in holdout.
- **Storage** — the `RunStore` is a directory of JSON files plus SQLite.
- **Reporting** — `holdout report` writes a single self-contained HTML file
  with no external assets; it opens on an air-gapped machine.

```console
$ holdout run cases.jsonl --target ollama:llama3.2 --seed 7
$ holdout run cases.jsonl --target ollama:llama3.2 --system "$(cat prompt_v2.txt)" --seed 7
$ holdout compare a1b2c3 d4e5f6
$ holdout report a1b2c3 d4e5f6 -o comparison.html
```

The only network traffic in that entire sequence is HTTP to
`localhost:11434`.
