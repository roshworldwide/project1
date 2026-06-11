# Pytest plugin

holdout registers a pytest plugin automatically when installed (via the
`pytest11` entry point) — there is nothing to enable. The plugin contributes
command-line options, fixtures, and a marker; the assertions live in
`holdout.testing` and work with or without it.

## The assertions

All comparison assertions wrap [`compare()`](regression-gate.md#compare-semantics)
and inherit its machinery: pairing by case id, test selection by score kind,
Benjamini-Hochberg correction, verdicts on corrected p-values. On failure the
`AssertionError` message contains the full comparison table — a failing CI run
tells you *which* metric moved, by how much, with what confidence.

### assert_no_regression

```python
from holdout.testing import assert_no_regression

cmp = assert_no_regression(baseline, candidate, alpha=0.05)
```

Passes on improvement or no significant change. Fails when:

- **any metric regressed** at the corrected significance level, or
- **nothing could be tested** (`insufficient_data`) — certifying "no
  regression" without evidence would be dishonest, so the absence of evidence
  is a failure, not a pass.

Returns the full `RunComparison` so you can log or store it. Keyword arguments
mirror `compare()`: `alpha`, `correction`, `test`, `n_resamples`, `seed`.

### assert_significant_improvement

```python
from holdout.testing import assert_significant_improvement

assert_significant_improvement(baseline, candidate, metric="exact_match")
```

With `metric` set, that metric must have *improved* at the corrected level;
without it, at least one metric must have improved. Either way there is a
second honesty rule: **any significant regression on any metric fails the
assertion** — an improvement that breaks something else is not shippable, and
this assertion refuses to celebrate it.

### assert_adequately_powered

```python
from holdout.testing import assert_adequately_powered

analyses = assert_adequately_powered(baseline, candidate, mde=0.05)
```

Guards against the meaningless null: for each shared metric it measures the
observed SD of per-pair differences and computes the sample size required to
detect `mde` at the stated `alpha` (default 0.05) and `power` (default 0.80).
If the runs have fewer paired cases than required, the assertion fails —
because a "no significant change" verdict from this comparison would be
statistically meaningless for effects of the size you care about:

```text
AssertionError: comparison is underpowered — a null result here would be meaningless:
  exact_match: have 50 pairs, need 339 to detect |Δ| >= 0.05 (sd_diff=0.3283, alpha=0.05, power=0.8)
```

Metrics whose paired differences have zero observed variance are trivially
powered (any real effect would show) and pass. Pair this with
`assert_no_regression` to make a green gate mean something: not regressed,
*and* capable of noticing if it had been.

### assert_no_leakage

```python
from holdout.testing import assert_no_leakage

assert_no_leakage(ev, SYSTEM_PROMPT)               # corpus: str | Sequence[str] | Target
assert_no_leakage(ev, target)                      # audits the target's system prompt
assert_no_leakage(ev, corpus, duplicate_threshold=None)  # skip the duplicate check
```

Fails if any case input/reference appears in the corpus (word-boundary exact
match or n-gram containment) or, by default, if the eval contains
near-duplicate case pairs. Details on the checks in
[Leakage and holdout discipline](leakage.md).

## The @llm_eval decorator

`@llm_eval` runs an eval when the test runs and hands the completed `Run` to
the test body as the `run` keyword:

```python
from holdout.testing import llm_eval

@llm_eval(support_qa, target=Ollama("llama3.2"), seed=7, store=".holdout")
def test_quality(run):
    assert run.n_errors == 0
    assert run.metrics()["exact_match"].ci_low >= 0.75
```

- The eval executes once per decorated test, against `target`, with `seed`
  (the determinism guarantee applies: same seed + same inputs, same run hash).
- `store` accepts a `RunStore` or a path; when given, the run is saved before
  the test body executes — so even a failing test leaves its evidence behind.
- The injected `run` parameter is hidden from pytest's fixture resolution; any
  other parameters of your test function are still resolved as fixtures.
- Under pytest the test is automatically marked `llm_eval`.

## Options, fixtures, and the marker

The plugin adds two command-line options and matching session fixtures:

| Option | Default | Fixture |
|---|---|---|
| `--holdout-store` | `.holdout` | `holdout_store` — the configured `RunStore` |
| `--holdout-seed` | `0` | `holdout_seed` — an `int` |

Tests that call real model targets should carry the `llm_eval` marker
(`@llm_eval` applies it for you; add `@pytest.mark.llm_eval` manually
otherwise). Then the fast suite simply deselects them:

```console
$ pytest -m "not llm_eval"        # unit tests only, no model calls
$ pytest -m llm_eval --holdout-seed 7 --holdout-store .holdout
```

## A complete test file

```python
# test_router_prompt.py
import pytest

from holdout import Eval, run
from holdout.providers import Ollama
from holdout.scorers import ExactMatch
from holdout.testing import (
    assert_adequately_powered,
    assert_no_leakage,
    assert_no_regression,
    llm_eval,
)

PROMPT_V1 = "Route the ticket to exactly one queue: billing or technical."
PROMPT_V2 = PROMPT_V1 + " Prefer billing when payment terms are mentioned."

router_eval = Eval.from_jsonl("evals/router.jsonl", scorers=[ExactMatch()])


def test_eval_hygiene():
    # Offline, fast: the eval set itself is clean for both prompts.
    assert_no_leakage(router_eval, [PROMPT_V1, PROMPT_V2])


# Mark explicitly when not using @llm_eval, so -m "not llm_eval" deselects it.
@pytest.mark.llm_eval
def test_prompt_v2_does_not_regress(holdout_store, holdout_seed):
    baseline = run(router_eval, target=Ollama("llama3.2", system=PROMPT_V1), seed=holdout_seed)
    candidate = run(router_eval, target=Ollama("llama3.2", system=PROMPT_V2), seed=holdout_seed)
    holdout_store.save(baseline)
    holdout_store.save(candidate)

    # A null result must be detectable to be believable.
    assert_adequately_powered(baseline, candidate, mde=0.08)
    assert_no_regression(baseline, candidate, alpha=0.05)


@llm_eval(router_eval, target=Ollama("llama3.2", system=PROMPT_V2), seed=7, store=".holdout")
def test_v2_error_free(run):
    assert run.n_errors == 0
```

A failing regression gate reads like this in the pytest output — the whole
comparison, not just a boolean:

```text
FAILED test_router_prompt.py::test_prompt_v2_does_not_regress
AssertionError: regression detected on: exact_match
support: ollama:llama3.2 (c1394105d81b) vs ollama:llama3.2 (6ad2612c0a8d)
  exact_match  REGRESSED  Δ=-0.087  [95% CI -0.163, -0.037]  mcnemar-exact  p=0.01563 (benjamini-hochberg-adjusted)  n=80
verdict: REGRESSED (alpha=0.05)
```
