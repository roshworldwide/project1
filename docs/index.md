# holdout

**The LLM eval framework that reports a confidence interval, not a vanity number.**

Every metric holdout reports carries its uncertainty. There is no public API that
returns a naked point estimate, and there is no verdict that is not backed by a
paired significance test. The question holdout answers is not "what is my score?"
— it is *"did quality actually change, or am I being fooled by noise?"*

## The point, in one runnable example

Two prompt versions, fifty paired cases, no API keys ([`StaticTarget`](guide/providers.md#statictarget)
is offline and deterministic — paste this into a file and run it):

```python
from holdout import Case, Eval, run
from holdout.providers import StaticTarget
from holdout.scorers import ExactMatch
from holdout.testing import assert_no_regression

cases = [Case(input=f"Q{i}: is the invoice total correct?", reference="yes", id=f"q{i:02d}")
         for i in range(50)]
ev = Eval("invoice-qa", cases, scorers=[ExactMatch()])

# v1 answers 46 of 50 correctly; v2 quietly breaks six cases v1 got right.
v1 = StaticTarget({c.input: ("yes" if i < 46 else "no") for i, c in enumerate(cases)}, name="prompt-v1")
v2 = StaticTarget({c.input: ("yes" if 6 <= i < 46 else "no") for i, c in enumerate(cases)}, name="prompt-v2")

assert_no_regression(run(ev, target=v1, seed=7), run(ev, target=v2, seed=7), alpha=0.05)
```

The naked scores are 0.92 and 0.80 — and at n=50 their 95% intervals overlap:

```text
invoice-qa  n=50  target=prompt-v1  run=9401060c1a09
  exact_match  0.920 [95% CI 0.820, 0.980] (n=50, bootstrap-bca)

invoice-qa  n=50  target=prompt-v2  run=7d96bc955d3c
  exact_match  0.800 [95% CI 0.660, 0.900] (n=50, bootstrap-bca)
```

Eyeballing two scores with overlapping intervals, you would shrug: maybe noise.
A fixed threshold would either fire on every flaky run or sleep through this one.
The paired test does neither — it pairs the runs case by case, sees that v2 broke
six cases and fixed none, and that asymmetry is signal:

```text
AssertionError: regression detected on: exact_match
invoice-qa: prompt-v1 (9401060c1a09) vs prompt-v2 (7d96bc955d3c)
  exact_match  REGRESSED  Δ=-0.120  [95% CI -0.240, -0.040]  mcnemar-exact  p=0.03125 (benjamini-hochberg-adjusted)  n=50
verdict: REGRESSED (alpha=0.05)
```

A planted regression that a naked-score comparison cannot distinguish from noise,
caught by an exact McNemar test at p=0.031. That is the product.

## Install

```console
$ pip install holdout
```

The core is dependency-light (numpy, httpx, rich) and fully offline-capable.
Cloud providers are optional extras:

```console
$ pip install 'holdout[openai]'      # OpenAI provider + embeddings
$ pip install 'holdout[anthropic]'   # Anthropic provider
$ pip install 'holdout[mlx]'         # Apple MLX, local in-process inference
```

Ollama and `StaticTarget` need no extra at all.

## What ships in v1.0

| Surface | What it does |
|---|---|
| **Library** | `Eval`, `Case`, `run()`, scorers, the statistics engine, `compare()` |
| **Pytest plugin** | `assert_no_regression` and friends; evals run under plain `pytest` |
| **CLI** | `holdout run / compare / list / report / power / check`, CI-grade exit codes |
| **GitHub Action** | Run an eval on a PR, post the comparison as a comment, gate the check |
| **HTML report** | Self-contained dark report, every metric drawn with its error bar |

## Where to go next

- [Quickstart](guide/quickstart.md) — define an eval, run it, read the intervals, store the run.
- [The regression gate](guide/regression-gate.md) — `compare()` semantics, exit codes, the GitHub Action.
- [How the statistics work](guide/statistics.md) — BCa bootstrap, paired tests, corrections, power. With citations.
- [Leakage and holdout discipline](guide/leakage.md) — contamination, near-duplicates, the reuse ledger.
- [Providers and targets](guide/providers.md) — Anthropic, OpenAI, Ollama, MLX, and air-gapped operation.
- [Pytest plugin](guide/pytest.md) — assertions, the `@llm_eval` decorator, fixtures, markers.
- [CLI reference](guide/cli.md) — every subcommand, the reference grammar, exit codes.
