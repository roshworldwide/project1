# The regression gate

The regression engine answers one question correctly: *did quality change, or
is this noise?* Two runs go in; an honest verdict comes out. This page covers
the semantics of `compare()`, the CLI exit-code contract, and the GitHub Action
that wires the verdict into CI.

## compare() semantics

```python
from holdout.regression import compare

cmp = compare(baseline, candidate, alpha=0.05, correction="benjamini-hochberg")
print(cmp.summary())
print(cmp.verdict)   # "improved" | "regressed" | "no_significant_change" | "insufficient_data"
```

For every metric shared by the two runs, `compare()` does the following, in order:

1. **Pair by case id.** Per-case scores are aligned on the case ids present in
   both runs. Cases that errored on either side (or exist on only one side)
   are dropped from the pairing — and every drop is disclosed in
   `cmp.warnings`. Pairing is the source of the engine's power: per-case
   differences remove between-case variance from the comparison, so a paired
   test detects effects that comparing two aggregate scores cannot
   (see [How the statistics work](statistics.md#why-pairing-buys-power)).

2. **Pick the test by score kind.** Scores carry a kind — `"binary"` or
   `"continuous"` — and with `test="auto"` (the default) the engine chooses
   accordingly:

    | Score kind (both sides) | Test | p-value |
    |---|---|---|
    | binary | exact McNemar on discordant pairs | exact, conservative |
    | continuous (or mixed) | paired bootstrap, shifted null | Monte Carlo, never zero |

    You can force `test="mcnemar"`, `test="paired-bootstrap"`, or
    `test="permutation"` (sign-flip permutation, exact when `2^n` fits the
    resample budget).

3. **Correct for multiple comparisons.** Each extra metric is an extra chance
   for a fluke "significant" result. Raw p-values are adjusted across metrics —
   Benjamini-Hochberg (false discovery rate) by default, `"holm"`
   (family-wise error rate) if any single false alarm is unacceptable, or
   `"none"` if you have exactly one metric and want to see the raw value.

4. **Issue verdicts on the corrected p-values only.** Per metric:

    | Verdict | Meaning |
    |---|---|
    | `improved` | adjusted p <= alpha and the effect is positive |
    | `regressed` | adjusted p <= alpha and the effect is negative |
    | `no_significant_change` | tested, but adjusted p > alpha |
    | `insufficient_data` | fewer than 2 paired cases — the metric could not be tested at all |

    The sign convention is `candidate - baseline` throughout: a positive effect
    means the candidate scored higher.

The overall `cmp.verdict` is **worst news wins**: any regressed metric makes
the comparison `regressed`; otherwise any improved metric makes it `improved`;
otherwise `no_significant_change` if at least one metric was actually tested;
otherwise `insufficient_data`.

### Why `insufficient_data` fails a no-regression assertion

`insufficient_data` is not a pass. If nothing could be tested, there is no
evidence that quality held — and certifying "no regression" without evidence
would be dishonest. So `assert_no_regression()` raises on `insufficient_data`,
and the CLI exits with a distinct code (2) so CI treats it as a failed gate,
not a green check. A gate that passes when it could not measure anything is a
gate in name only.

### Warnings are part of the result

`cmp.warnings` carries every honesty note: eval fingerprint mismatches (the two
runs were not made on provably identical datasets), unpaired cases dropped
because of errors, metrics present in only one run. Read them — a comparison
never silently narrows its claim.

## The CLI gate

```console
$ holdout compare c1394105d81b 6ad2612c0a8d
support: router-v1 (c1394105d81b) vs router-v2 (6ad2612c0a8d)
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━┓
┃ metric      ┃ verdict   ┃ effect ┃               CI ┃ test          ┃ p (adj) ┃  n ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━┩
│ exact_match │ REGRESSED │ -0.087 │ [-0.163, -0.037] │ mcnemar-exact │ 0.01563 │ 80 │
└─────────────┴───────────┴────────┴──────────────────┴───────────────┴─────────┴────┘
verdict: REGRESSED (alpha=0.05, benjamini-hochberg)
eval 'support' has been used 1 time(s) of a budget of 20
$ echo $?
1
```

The exit code is the contract:

| Exit code | Meaning | What CI should do |
|---|---|---|
| `0` | no significant regression (improved or no significant change) | pass the check |
| `1` | regression detected at the corrected alpha | fail the check |
| `2` | insufficient data — refusing to certify (also used for usage errors) | fail the check |

A misconfigured or unmeasurable gate never comes back green: the only path to
exit 0 is an actual tested non-regression.

The trailing ledger line is the [holdout-discipline ledger](leakage.md#the-holdoutledger)
recording that you looked at this eval one more time; `--no-ledger` skips it,
`--budget` sets the threshold. `--markdown` emits the same table as
GitHub-flavored markdown for PR comments.

## The GitHub Action

The repository ships a composite action that runs the candidate eval on the PR,
compares it to a committed baseline run, posts the comparison table as a PR
comment, and fails the check only on a statistically significant regression —
never on noise.

A complete workflow:

```yaml
# .github/workflows/evals.yml
name: evals

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read
  pull-requests: write   # needed to post the PR comment

jobs:
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: holdout eval gate
        # Until the repo is renamed at launch, reference it as:
        uses: roshworldwide/project1@v1
        with:
          eval: evals/support_qa.jsonl          # .jsonl file or Python 'module:attr'
          target: "anthropic:claude-sonnet-4-6" # or ollama:M | openai:M | module:attr
          baseline: dace0098e767                # run id committed in the store
          store: .holdout                       # commit your baseline artifact inside it
          scorer: exact                         # for .jsonl evals
          alpha: "0.05"
          correction: benjamini-hochberg
          seed: "42"
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Action inputs (from `action.yml`):

| Input | Default | Purpose |
|---|---|---|
| `eval` | — (required) | Eval reference: a `.jsonl` file or Python `module:attr` |
| `target` | — (required) | Target reference: `ollama:M` / `openai:M` / `anthropic:M` / `module:attr` |
| `baseline` | — (required) | Baseline run id (or unambiguous prefix) present in the store |
| `store` | `.holdout` | Run store directory; commit the baseline artifact inside it |
| `scorer` | `exact` | Scorer for `.jsonl` evals: `exact`, `exact-strict`, `regex:<pattern>` |
| `alpha` | `0.05` | Significance level applied to corrected p-values |
| `correction` | `benjamini-hochberg` | `benjamini-hochberg`, `holm`, or `none` |
| `seed` | `0` | Seed for generation and resampling |
| `python-version` | `3.12` | Python to set up |
| `install` | `holdout` | pip spec; use `"."` to install from the repo checkout |
| `comment` | `true` | Post the comparison as a PR comment |
| `github-token` | `${{ github.token }}` | Token used to post the comment |

Outputs: `run-id` (the candidate run's id) and `verdict`
(`no_significant_regression` / `regressed` / `insufficient_data` / `error`).
The comparison table is also appended to the job's step summary, and the final
step exits with the `compare` exit code, so the check status follows the
contract above.

### The dogfood

This repository gates itself with its own action. The workflow runs a
deterministic offline target — no model calls, no keys — against a baseline
artifact committed under `examples/ci/store`:

```yaml
# this repo's .github/workflows/evals.yml (abridged)
env:
  PYTHONPATH: examples/ci   # so 'targets:candidate' resolves

steps:
  - uses: actions/checkout@v4
  - name: holdout eval gate
    uses: ./
    with:
      eval: examples/ci/cases.jsonl
      target: "targets:candidate"     # a StaticTarget in examples/ci/targets.py
      baseline: dace0098e767
      store: examples/ci/store
      seed: "42"
      install: "."
```

The pattern to copy: keep the baseline run artifact in the repo (it is one
JSON file, content-addressed, diffable), point `baseline` at its id, and let
the PR run produce the candidate. When you intentionally improve the system,
re-run the eval on main and commit the new baseline artifact.

## From pytest instead

The same gate is available as an assertion —
`assert_no_regression(baseline, candidate, alpha=0.05)` — with identical
semantics, including failure on `insufficient_data`. See the
[pytest plugin](pytest.md) page.
