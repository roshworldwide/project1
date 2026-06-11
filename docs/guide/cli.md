# CLI reference

```console
$ holdout --store .holdout <command> ...
```

`--store` is global (before the subcommand) and defaults to `.holdout`, the
same directory the library and pytest plugin use.

## Reference grammar

Two reference grammars appear throughout the CLI:

**Targets** — provider shorthand or a Python reference:

| Form | Example | Resolves to |
|---|---|---|
| `ollama:MODEL` | `ollama:llama3.2` | `Ollama(MODEL)` — local, no extra needed |
| `openai:MODEL` | `openai:gpt-4o-mini` | `OpenAI(MODEL)` — needs `holdout[openai]` |
| `anthropic:MODEL` | `anthropic:claude-sonnet-4-6` | `Anthropic(MODEL)` — needs `holdout[anthropic]` |
| `package.module:attr` | `targets:candidate` | any `Target` object importable from `PYTHONPATH` |

Provider options (`--system`, `--temperature`, `--max-tokens`, `--base-url`)
apply only to the shorthand forms; a Python reference is used as-is.

**Evals** — a `.jsonl` path (scorers supplied with `--scorer`) or a Python
reference to an `Eval` object. The `--scorer` specs for `.jsonl` evals are
`exact` (normalized exact match, the default), `exact-strict` (no
normalization), and `regex:<pattern>`; for embedding or custom scorers, point
at a Python reference instead.

## Exit codes

| Command | 0 | 1 | 2 |
|---|---|---|---|
| `compare` | no significant regression | regression detected | insufficient data — refusing to certify |
| `check` | clean | leakage or duplicates found | — |
| all | success | — | usage or runtime error |

Any error (bad reference, missing file, ambiguous run id) exits 2 with a
message on stderr — a broken gate never reads as a pass.

## holdout run

Run an eval against a target and save the run to the store.

```console
$ PYTHONPATH=. holdout run cases.jsonl --target targets:v1 --seed 7
running Eval(name='cases', cases=5, scorers=['exact_match']) against arith-v1 (seed=7)
cases · arith-v1 · run 37b86023c19b
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ metric      ┃ estimate ┃         95% CI ┃ method              ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ exact_match │    1.000 │ [1.000, 1.000] │ bootstrap-bca (n=5) │
└─────────────┴──────────┴────────────────┴─────────────────────┘
saved 37b86023c19b -> .holdout/runs/37b86023c19b...json
```

| Flag | Default | Purpose |
|---|---|---|
| `--target` | required | target reference (see grammar above) |
| `--scorer` | `exact` | scorer spec for `.jsonl` evals; repeatable |
| `--system` | none | system prompt for provider shorthand targets |
| `--temperature` | `0.0` | sampling temperature |
| `--max-tokens` | `1024` | generation cap |
| `--base-url` | none | override the provider endpoint (Ollama server, OpenAI-compatible API) |
| `--seed` | `0` | generation + identity seed |
| `--max-concurrency` | `8` | cases in flight at once |
| `--id-file` | none | write the full run id to this file (for CI scripting) |

## holdout compare

Compare two stored runs (by id or unambiguous prefix) and issue a verdict.
This is the regression gate; the [exit code](#exit-codes) is the contract.

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

The last line is the [holdout-discipline ledger](leakage.md#the-holdoutledger):
each compare records one adaptive look at this eval set, and the level
(`ok` / `caution` / `overfit-risk`) tells you how worn out the set is.

| Flag | Default | Purpose |
|---|---|---|
| `--alpha` | `0.05` | significance level applied to corrected p-values |
| `--correction` | `benjamini-hochberg` | `benjamini-hochberg` \| `holm` \| `none` |
| `--test` | `auto` | `auto` \| `paired-bootstrap` \| `mcnemar` \| `permutation` |
| `--seed` | `0` | resampling seed (same runs + seed = identical comparison) |
| `--budget` | `20` | ledger use budget for this eval |
| `--no-ledger` | off | do not record this look in the ledger |
| `--markdown` | off | emit GitHub-flavored markdown instead of a terminal table |

`--markdown` is what the GitHub Action posts as the PR comment:

```console
$ holdout compare c1394105d81b 6ad2612c0a8d --markdown
### holdout · `support` — **REGRESSED**

baseline `c1394105d81b` (router-v1) vs candidate `6ad2612c0a8d` (router-v2) · alpha=0.05 · correction=benjamini-hochberg

| metric | verdict | effect | CI | test | p (adj) | n |
|---|---|---:|---:|---|---:|---:|
| `exact_match` | REGRESSED | -0.087 | [-0.163, -0.037] | mcnemar-exact | 0.01563 | 80 |

_every metric carries its uncertainty — holdout_

> eval 'support' has been used 2 time(s) of a budget of 20 [ok]
```

## holdout list

List stored runs, newest first.

```console
$ holdout list --eval support --limit 10
stored runs
┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━┓
┃ run          ┃ eval    ┃ target    ┃ created               ┃  n ┃ errors ┃ seed ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━┩
│ 6ad2612c0a8d │ support │ router-v2 │ 2026-06-11T03:33:30.7 │ 80 │      0 │    7 │
│ c1394105d81b │ support │ router-v1 │ 2026-06-11T03:33:29.9 │ 80 │      0 │    7 │
└──────────────┴─────────┴───────────┴───────────────────────┴────┴────────┴──────┘
```

| Flag | Default | Purpose |
|---|---|---|
| `--eval` | all | filter by eval name |
| `--limit` | `20` | maximum rows |

## holdout report

Write a self-contained HTML report — one file, no external assets, dark theme,
every metric drawn with its error bar. One run id gives a run view; two give a
comparison view.

```console
$ holdout report c1394105d81b -o run.html
wrote run.html
$ holdout report c1394105d81b 6ad2612c0a8d -o comparison.html
wrote comparison.html
```

| Flag | Default | Purpose |
|---|---|---|
| `-o`, `--output` | `holdout-report.html` | output path |
| `--alpha` | `0.05` | significance level (comparison view) |
| `--seed` | `0` | resampling seed (comparison view) |

## holdout power

The sample-size / minimum-detectable-effect calculator. Provide a measure of
per-pair difference variability — either `--sd` directly, or `--p01`/`--p10`
(expected fix/break rates) for binary metrics — and exactly one of `--n` (to
get the MDE) or `--mde` (to get the required n).

```console
$ holdout power --n 200 --p01 0.05 --p10 0.05
sd_diff = 0.3162 (from p01=0.05, p10=0.05)
n=200 pairs detects |Δ| >= 0.0626 at alpha=0.05 with power 0.8 (sd_diff=0.3162)

$ holdout power --mde 0.05 --sd 0.32
n=322 pairs detects |Δ| >= 0.0500 at alpha=0.05 with power 0.8 (sd_diff=0.3200)
```

Run this *before* trusting any "no significant change" verdict: if your eval
has 80 cases and the MDE at n=80 is 0.10, a null result says nothing about
5-point regressions. See [power and MDE](statistics.md#power-and-minimum-detectable-effect).

| Flag | Default | Purpose |
|---|---|---|
| `--n` | — | paired cases you have (computes the MDE) |
| `--mde` | — | effect size you must detect (computes required n) |
| `--sd` | — | SD of per-pair differences |
| `--p01` / `--p10` | — | expected fix/break rates (binary metrics; computes sd) |
| `--alpha` | `0.05` | two-sided significance level |
| `--power` | `0.80` | desired detection probability |

## holdout check

Leakage audit: contamination of the eval by prompt text, plus near-duplicate
cases. Exit 0 = clean, 1 = findings.

```console
$ holdout check qa.jsonl --corpus system_prompt.txt
contamination check (exact-substring + 5-gram containment >= 0.5): 2/4 case(s) flagged
  reset-pw.input: exact-substring (score=1.000) — appears verbatim in corpus text #0
  reset-pw.reference: exact-substring (score=1.000) — appears verbatim in corpus text #0
near-duplicate cases (effective n is below 4):
  reset-pw ~ reset-pw-dupe (similarity=1.000)
$ echo $?
1
```

| Flag | Default | Purpose |
|---|---|---|
| `--scorer` | `exact` | scorer(s) for `.jsonl` evals (needed only to construct the eval) |
| `--corpus` | none | text file with prompt content; repeatable |
| `--corpus-text` | none | literal prompt text; repeatable |
| `--ngram` | `5` | word n-gram size for containment |
| `--threshold` | `0.5` | containment fraction at which a field is flagged |
| `--dup-threshold` | `0.8` | Jaccard similarity at which a case pair is flagged |

With no corpus, the contamination pass is skipped (it says so) and only the
duplicate check runs. The semantics of both checks are documented in
[Leakage and holdout discipline](leakage.md).
