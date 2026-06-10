# DESIGN — holdout

*A quant-grade LLM evaluation and regression framework. "pytest for LLMs."*

---

## 1. Name

**`holdout`** — `pip install holdout`, `import holdout`.

The holdout set is the single most important discipline quantitative finance and ML share:
the data you are not allowed to touch, so it can still tell you the truth. This product is
that discipline, productized — the name *is* the positioning.

PyPI availability: checked 2026-06-10 — `holdout` returns 404 on
`pypi.org/pypi/holdout/json`, as do the PEP 503 normalization variants `hold-out`,
`hold_out`, and the neighbor `holdouts`. Clean. (Of the other candidates: `walkforward`
and `ledger-eval` were available; `rigor` and `coldstart` are taken.)

Repo: `github.com/roshworldwide/project1` (rename to `roshworldwide/holdout` at launch).

**One-line positioning:** *The LLM eval framework that reports a confidence interval, not a
vanity number.*

---

## 2. The public API — the five things a user writes most

### 2.1 Define an eval

```python
from holdout import Eval, Case
from holdout.scorers import ExactMatch, EmbeddingSimilarity

qa = Eval(
    name="support-qa",
    cases=[
        Case(input="How do I reset my password?", reference="Settings > Security > Reset."),
        # ... or load 500 of them:
    ],
    scorers=[ExactMatch(), EmbeddingSimilarity(threshold=0.85)],
)
# also: Eval.from_jsonl("cases.jsonl", scorers=[...])
```

### 2.2 Run it against a target

```python
from holdout import run
from holdout.providers import Anthropic

baseline = run(
    qa,
    target=Anthropic(model="claude-sonnet-4-6", system=PROMPT_V1),
    seed=7,                      # same seed + same inputs => identical run hash
)
print(baseline.summary())
# support-qa  n=200
#   exact_match            0.84  [95% CI 0.79, 0.88]   (BCa bootstrap, B=10000)
#   embedding_similarity   0.91  [95% CI 0.88, 0.93]
```

Every number ships with its interval. There is no API to get a score without one.

### 2.3 Assert no regression (the pytest moment)

```python
# test_prompts.py — runs under plain `pytest`
from holdout import run
from holdout.testing import assert_no_regression

def test_prompt_v2_does_not_regress(support_qa, anthropic_target):
    baseline  = run(support_qa, target=anthropic_target(system=PROMPT_V1), seed=7)
    candidate = run(support_qa, target=anthropic_target(system=PROMPT_V2), seed=7)
    assert_no_regression(baseline, candidate, alpha=0.05)

# FAILED test_prompts.py::test_prompt_v2_does_not_regress
#   exact_match REGRESSED: Δ = -0.060  [95% CI -0.103, -0.018]
#   paired bootstrap, p = 0.003 (Benjamini-Hochberg corrected, m=2)
#   A naked-score diff (0.84 -> 0.78) cannot tell you this is signal. This test can.
```

Also: `assert_significant_improvement(a, b)`, `assert_no_leakage(eval, target)`,
`assert_adequately_powered(eval, mde=0.05)`.

### 2.4 Compare two runs from the CLI

```console
$ holdout compare a1b2c3 d4e5f6
support-qa: PROMPT_V1 (a1b2c3) vs PROMPT_V2 (d4e5f6), n=200 paired cases

  metric         verdict     effect      95% CI            test               p (BH)
  exact_match    REGRESSED   -0.060      [-0.103, -0.018]  paired bootstrap   0.003
  emb_sim        no sig. Δ   +0.004      [-0.011, +0.020]  paired bootstrap   0.581

exit code 1 (regression detected) — wire it straight into CI
```

### 2.5 Gate CI

```yaml
# .github/workflows/evals.yml
- uses: roshworldwide/holdout@v1
  with:
    eval: evals/support_qa.py
    baseline: main          # compares PR run against baseline branch run
    alpha: "0.05"
    # posts a PR comment with verdict + CIs; fails the check only on
    # statistically significant regression — never on noise
```

---

## 3. Competitive honesty

| | promptfoo | deepeval | ragas | LangSmith | **holdout** |
|---|---|---|---|---|---|
| Core job | prompt matrix testing | pytest-style metrics | RAG-specific metrics | hosted tracing + evals | regression gate with statistics |
| Score output | point estimates | point estimates, pass/fail thresholds | point estimates | point estimates, dashboards | **estimate + CI, always** |
| Significance testing | no | no | no | no | **paired bootstrap, McNemar, permutation; BH correction** |
| "Is my eval set big enough?" | no | no | no | no | **power / MDE analysis** |
| Leakage & holdout discipline | no | no | no | no | **contamination + overfit-to-eval detection** |
| Local / air-gapped | partial | partial | partial | no (hosted) | **first-class: Ollama, Apple MLX, 0 bytes out** |
| CI regression gate | threshold-based | threshold-based | no | hosted | **statistical verdict, exit-code native** |

These are good tools; they answer "what's my score?" `holdout` answers a different question:
**"did quality actually change, or am I being fooled by noise?"** — and refuses to answer
it dishonestly. Fixed thresholds on noisy point estimates produce both false alarms (flaky
CI) and silent misses (real regressions inside the noise band). Statistics fixes both.

---

## 4. Architecture

Python 3.11+, typed throughout (`mypy --strict`), `src/` layout, hatchling, MIT.

```
src/holdout/
  core/           Eval, Case, Target, Run — immutable, versioned, content-hashed
  providers/      OpenAI | Anthropic | Ollama | MLX behind one Provider protocol (lazy imports)
  scorers/        exact, regex, embedding cosine, LLM-as-judge (judge uncertainty surfaced)
  stats/          THE MOAT: BCa bootstrap CIs, paired bootstrap / McNemar / permutation
                  tests, Benjamini-Hochberg, power & MDE analysis. Citations in docstrings.
  regression/     two Runs in -> verdict out (improved / regressed / no significant change)
  leakage/        n-gram + embedding contamination detection, holdout-discipline ledger
  store/          SQLite index + content-addressed JSON artifacts, seeded determinism
  pytest_plugin/  fixtures + assertions; `pytest` just runs your evals
  cli/            run | compare | report | power | dashboard  (Rich output, CIs everywhere)
  report/         self-contained dark HTML report, every metric an error bar
dashboard/        local-first read-only SPA (Vite/React/TS), bundled into the wheel
action/           GitHub Action: run on PR, post verdict comment, gate the check
```

Key invariants:

1. **No naked point estimates.** Aggregates are `Estimate(value, ci_low, ci_high, method, n)`.
   `__str__` renders the interval. There is no public path to a bare float aggregate.
2. **Determinism.** `seed` threads through sampling, bootstrap, and provider calls
   (temperature 0 by default); same seed + same inputs ⇒ identical run hash, proven by test.
3. **Paired by design.** Runs over the same Eval share case IDs, so comparisons use paired
   tests — far higher power at the same n than unpaired comparisons.
4. **Local-first.** Cloud providers are optional extras (`holdout[openai]`,
   `holdout[anthropic]`); the core, Ollama path, and dashboard work fully offline.

### Statistical methods (with the references the docstrings will cite)

- BCa bootstrap CIs — Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, ch. 14.
- Paired bootstrap test for metric deltas — Efron & Tibshirani (1993), ch. 16.
- McNemar's exact test for paired binary outcomes — McNemar (1947); Edwards (1948) correction.
- Permutation tests — Good (2005), *Permutation, Parametric and Bootstrap Tests of Hypotheses*.
- Benjamini–Hochberg FDR control — Benjamini & Hochberg (1995), *JRSS B* 57(1).
- Power / minimum detectable effect for paired proportions and means — standard normal
  approximation per Chow, Shao & Wang (2008), *Sample Size Calculations in Clinical Research*.

---

## 5. Scope & non-goals (v1.0)

**In:** the library (core, providers×4, scorers, stats, regression, leakage, store),
pytest plugin, CLI, HTML report, GitHub Action, docs + examples. v1.1 adds the local
dashboard — a strict enhancement, never a blocker.

**Out (deliberately):**
- No hosted SaaS, no accounts, no telemetry. Local-first is the moat, not a limitation.
- No prompt-optimization / auto-tuning — that is exactly how teams overfit their eval set;
  we *detect* that, we don't sell it.
- No agent-trajectory tracing or observability — promptfoo/LangSmith territory.
- No training, fine-tuning, or dataset generation.
- No fabricated benchmarks. If a number isn't measured by `benchmarks/`, it doesn't ship.
