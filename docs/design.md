# Design

A short map of how holdout is built and the rules it refuses to break. The
full design document lives in the repository as `DESIGN.md`.

## Module map

```text
src/holdout/
  core/           Eval, Case, Target, Run — immutable, validated, content-hashed
  providers/      Anthropic | OpenAI | Ollama | MLX | StaticTarget behind one
                  Target protocol; lazy imports, one shared retry discipline
  scorers/        exact, regex, embedding cosine — scores carry their kind
                  (binary/continuous) so the stats engine picks the right test
  stats/          the moat: BCa bootstrap CIs, paired bootstrap / exact McNemar /
                  sign-flip permutation, Benjamini-Hochberg & Holm, power & MDE.
                  Citations in docstrings.
  regression/     two Runs in -> verdict out (improved / regressed /
                  no_significant_change / insufficient_data)
  leakage/        contamination (exact + n-gram + embeddings), near-duplicates,
                  the holdout-discipline ledger
  store/          SQLite index + content-addressed JSON artifacts
  testing.py      the assertions + @llm_eval (work with or without the plugin)
  pytest_plugin/  options, marker, fixtures — auto-registered via pytest11
  cli/            run | compare | list | report | power | check
  report/         self-contained dark HTML, every metric an error bar
```

Python 3.11+, typed throughout (`mypy --strict`), `src/` layout, MIT.

## The invariants

**No naked point estimates.** Aggregates are
`Estimate(value, ci_low, ci_high, n, level, method)`, and its string form
always renders the interval. Significance tests return `TestResult` — effect,
CI on the effect, p-value, n, test name — never a bare p. The comparison
engine issues verdicts only on corrected p-values. This is cultural as much as
technical: there is no public API to get a metric without its uncertainty.

**Determinism via content addressing.** Every identity in holdout — eval,
target, scorer, run — is a SHA-256 hash of canonical JSON over the content
that defines it. A `Run`'s id hashes the eval fingerprint, target fingerprint,
scorer fingerprints, seed, and per-case results; wall-clock fields
(timestamps, latencies) are recorded but excluded from the hash. The guarantee
that follows: **same seed + same inputs = identical run hash**, proven by
test. Consequences:

- Saving a run is idempotent; two stores merge by copying files.
- The store verifies artifacts on load — a file whose recomputed hash does not
  match its name is rejected as tampered.
- The bootstrap RNG for a run's aggregates is seeded from the run hash, so the
  same run always reports identical intervals.
- A changed scorer threshold is a changed measurement: scorer configuration is
  fingerprinted, so it cannot silently masquerade as the old metric.

**Paired by design.** Runs over the same eval share case ids, so comparisons
always use paired tests — far more power at the same n than unpaired
comparison, and the reason `Eval` rejects duplicate case ids at construction.

**Local-first.** Cloud providers are optional extras. The core, the Ollama and
MLX paths, the store, the statistics, and the HTML report work fully offline.
No accounts, no telemetry, zero bytes out unless you point a provider at a
cloud API.

**Honesty over convenience.** Untestable metrics are reported as
`insufficient_data`, never silently passed; `assert_no_regression` fails on
them. Fallbacks are disclosed in the `method` string. Dropped pairs and
fingerprint mismatches surface as warnings on the comparison. Monte-Carlo
p-values can never be zero.

## Non-goals

Deliberately out of scope, from `DESIGN.md`:

- **No hosted SaaS, no accounts, no telemetry.** Local-first is the moat, not
  a limitation.
- **No prompt optimization or auto-tuning.** Iterating a prompt against a
  fixed eval set is exactly how teams overfit their evals; holdout *detects*
  that (the ledger), it does not sell it.
- **No agent-trajectory tracing or observability.** Other tools do that job.
- **No training, fine-tuning, or dataset generation.**
- **No fabricated benchmarks.** If a number is not measured, it does not ship.
