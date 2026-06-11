# Quickstart

This walk-through goes from zero to a stored, comparable run. Every snippet is
copy-pasteable; the offline variants use `StaticTarget` and run with no API keys,
no network, and no model installed.

## 1. Define an eval

An [`Eval`](../api.md#holdout.Eval) is a named set of [`Case`](../api.md#holdout.Case)
objects plus the scorers that measure them. Construction validates eagerly:
non-empty cases and scorers, unique case ids, and a `reference` on every case
wherever a scorer requires one.

```python
from holdout import Case, Eval
from holdout.scorers import ExactMatch

ev = Eval(
    name="support-qa",
    cases=[
        Case(input="How do I reset my password?",
             reference="Settings > Security > Reset password."),
        Case(input="What payment methods do you accept?",
             reference="Visa, Mastercard, and ACH transfer."),
        Case(input="How do I export my invoices as CSV?",
             reference="Billing > Invoices > Export."),
    ],
    scorers=[ExactMatch()],
)
```

Case ids matter: they are what pairs a case across runs so comparisons can use
paired statistics. If you omit `id`, a stable content-derived id is assigned.
Set explicit ids when you expect to edit case text later — a content-derived id
changes with the content, and an edited case stops pairing with old runs.

For real evals you will usually load cases from a JSONL file, one object per
line with `input` (required) and optional `reference`, `id`, and `metadata`:

```python
ev = Eval.from_jsonl("cases.jsonl", scorers=[ExactMatch()])
```

```text
{"input": "How do I reset my password?", "reference": "Settings > Security > Reset password.", "id": "reset-pw"}
{"input": "What payment methods do you accept?", "reference": "Visa, Mastercard, and ACH transfer.", "id": "payments"}
```

## 2. Run it against a target

A target is anything that satisfies the [`Target`](../api.md#holdout.Target)
protocol — a built-in provider, or your own RAG pipeline. Against a real model:

```python
from holdout import run
from holdout.providers import Anthropic

result = run(
    ev,
    target=Anthropic(model="claude-sonnet-4-6", system="Answer in one sentence."),
    seed=7,
)
```

Offline, for trying the machinery (this is what the rest of this page uses):

```python
from holdout import run
from holdout.providers import StaticTarget

target = StaticTarget(
    {
        "How do I reset my password?": "Settings > Security > Reset password.",
        "What payment methods do you accept?": "Visa, Mastercard, and ACH transfer.",
        "How do I export my invoices as CSV?": "Billing > Invoices > Export.",
    },
    name="canned-v1",
)
result = run(ev, target=target, seed=7)
```

`run(ev, target=..., seed=..., max_concurrency=...)` executes cases under
bounded async concurrency and returns an immutable
[`Run`](../api.md#holdout.Run). A failing provider call never aborts the run —
the failure is recorded on that case and surfaced in the error count. The
`seed` threads through generation (where the backend supports it) and into the
run's identity: same seed + same inputs gives an identical run hash. Inside an
async application, use `await arun(...)` instead; `run()` refuses to run inside
an event loop.

## 3. Read the summary

```python
print(result.summary())
```

```text
support-qa  n=3  target=canned-v1  run=76f2fdbfdb42
  exact_match  1.000 [95% CI 1.000, 1.000] (n=3, bootstrap-bca)
```

Every metric is an [`Estimate`](../api.md#holdout.stats.Estimate) and renders as:

```text
0.840 [95% CI 0.760, 0.920] (n=50, bootstrap-bca)
```

Read it as: the point estimate is 0.840; the 95% confidence interval is
[0.760, 0.920]; it was computed from 50 cases by the BCa bootstrap. The width
of the interval is the honesty: at n=50 a "0.84" means *somewhere between 0.76
and 0.92*, and any decision that depends on more precision than that needs more
cases, not more confidence. There is no API that returns the bare `0.840` as an
aggregate — `result.metrics()` returns `Estimate` objects, and the method label
even discloses when the BCa interval fell back to percentile
(see [How the statistics work](statistics.md)).

If you ever see `errors: N case(s) failed (excluded from aggregates)` in a
summary, generation or scoring failed for those cases; the aggregate honestly
covers only the cases that completed.

## 4. Save to a store

A [`RunStore`](../api.md#holdout.store.RunStore) is a local directory: one
content-addressed JSON artifact per run plus a rebuildable SQLite index.

```python
from holdout.store import RunStore

store = RunStore(".holdout")     # the default directory
path = store.save(result)        # idempotent: same run, same file
print(result.run_id)             # full SHA-256 content hash
print(result.short_run_id)       # 12-character display prefix
```

Saving is idempotent because the artifact is named by the run's content hash,
and two stores can be merged by copying files. Load runs back by id or any
unambiguous prefix:

```python
baseline = store.load("76f2fdbfdb42")
latest = store.latest(eval_name="support-qa")
```

## 5. The same flow from the CLI

The CLI wraps everything above. With a JSONL file and an offline target defined
in a Python module (`targets.py` containing `v1 = StaticTarget(...)`):

```console
$ holdout run cases.jsonl --target targets:v1 --seed 7
running Eval(name='cases', cases=5, scorers=['exact_match']) against arith-v1 (seed=7)
cases · arith-v1 · run 37b86023c19b
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ metric      ┃ estimate ┃         95% CI ┃ method              ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ exact_match │    1.000 │ [1.000, 1.000] │ bootstrap-bca (n=5) │
└─────────────┴──────────┴────────────────┴─────────────────────┘
saved 37b86023c19b -> .holdout/runs/37b86023c19b...json
```

Against a real provider the target reference is shorthand:
`--target anthropic:claude-sonnet-4-6`, `--target openai:gpt-4o-mini`, or
`--target ollama:llama3.2` for fully local runs.

List what you have stored:

```console
$ holdout list
stored runs
┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━┳━━━━━━━━┳━━━━━━┓
┃ run          ┃ eval  ┃ target   ┃ created               ┃ n ┃ errors ┃ seed ┃
┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━╇━━━━━━━━╇━━━━━━┩
│ bb61949cdbfb │ cases │ arith-v2 │ 2026-06-11T03:31:06.4 │ 5 │      0 │    7 │
│ 37b86023c19b │ cases │ arith-v1 │ 2026-06-11T03:31:06.3 │ 5 │      0 │    7 │
└──────────────┴───────┴──────────┴───────────────────────┴───┴────────┴──────┘
```

And render a self-contained HTML report — one file, no external assets, every
metric drawn as a literal error bar; pass two run ids for a comparison view:

```console
$ holdout report 37b86023c19b -o holdout-report.html
wrote holdout-report.html
```

## Next

You now have two runs in a store, which is exactly what the regression gate
consumes. Continue with [The regression gate](regression-gate.md), or run
`holdout compare <baseline> <candidate>` and read the verdict.
