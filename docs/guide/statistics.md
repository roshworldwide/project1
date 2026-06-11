# How the statistics work

This is the deep dive. The statistics engine is the reason holdout exists, so
this page explains what each method actually computes, why it was chosen, and
where it is weak. Every method here is implemented in `holdout.stats` with the
citation in its docstring; the same references appear at the bottom of this
page.

## Confidence intervals: BCa bootstrap

Every aggregate in holdout is an `Estimate(value, ci_low, ci_high, n, level,
method)` produced by `bootstrap_ci()`. The default method is **BCa** —
bias-corrected and accelerated.[^efron87][^efron93]

```python
from holdout.stats import bootstrap_ci

est = bootstrap_ci(values, level=0.95, n_resamples=10_000, seed=7)
print(est)   # 0.764 [95% CI 0.749, 0.778] (n=200, bootstrap-bca)
```

### Why percentile is not enough

The naive percentile interval takes the 2.5th and 97.5th percentiles of the
bootstrap distribution and calls it a day. That is only first-order accurate:
its actual coverage error shrinks like `1/sqrt(n)`. When the statistic's
sampling distribution is biased or skewed — which describes accuracy on a small
eval, scores clipped to [0, 1], and most things LLM evals measure — percentile
intervals are systematically off-center.

BCa corrects both defects and is second-order accurate (coverage error shrinks
like `1/n`):

- **Bias correction `z0`** — the normal quantile of the fraction of bootstrap
  statistics below the point estimate. If the bootstrap distribution is not
  centered on the estimate, `z0` shifts the interval to compensate.
- **Acceleration `a`** — estimated from the jackknife (leave-one-out
  statistics), it corrects for the statistic's variance changing with its
  value, i.e. skewness.

The nominal quantile levels are mapped through the BCa transformation
(Efron & Tibshirani 1993, eq. 14.9–14.10) before reading off the bootstrap
distribution.

### Implementation honesty

Three edge conventions, all visible in the output rather than hidden:

- **Ties count half.** With heavily discrete data (accuracy on a small eval),
  many bootstrap statistics tie the point estimate exactly. A strictly-less
  convention makes `z0` lurch between extremes; holdout counts ties as half
  ("mid-rank"), which degrades gracefully and reduces BCa to the percentile
  interval for degenerate distributions.
- **Disclosed fallback.** If every bootstrap statistic falls on one side of the
  point estimate, `z0` is mathematically undefined. holdout falls back to the
  percentile interval and says so: the Estimate's method reads
  `bootstrap-percentile (bca z0 undefined)`, not `bootstrap-bca`.
- **`n == 1` is not an interval.** A single observation has no resampling
  distribution. The result is a degenerate interval labeled
  `degenerate (n=1)` — never presented as a real CI.

The resampling RNG is seeded; run aggregates seed it from the run hash, so the
same run always reports identical intervals.

!!! warning "Small-n caveat"
    The bootstrap resamples *your* data; it cannot manufacture information that
    is not there. Below roughly n=20–30, bootstrap intervals (BCa included)
    tend to undercover — they are too narrow, especially for proportions near
    0 or 1. The interval is still the most honest summary available at that n,
    but the real fix is more cases, and the [power analysis](#power-and-minimum-detectable-effect)
    will tell you how many.

## Paired tests

All significance tests in holdout are *paired*: they operate on per-case
differences `d_i = b_i - a_i` between two runs over the same cases (positive
effect = candidate scored higher).

### Why pairing buys power

The variance of a difference of two means over independent samples is
`Var(a) + Var(b)`. The variance of the mean of paired differences is
`Var(a) + Var(b) - 2 Cov(a, b)` — and on the same cases the covariance is
large, because the same hard cases are hard for both prompts. Pairing
subtracts the between-case variance from the comparison; what remains is
exactly the part the two systems disagree on.

Concretely: in the [front-page example](../index.md), two runs at 0.92 and 0.80
have overlapping 95% intervals at n=50 — unpaired, you cannot call it. Paired,
the engine sees the per-case picture (6 cases broke, 0 improved) and rejects
the null at p=0.031. Same data, more truth. This is also why
`Eval` refuses duplicate case ids and why case ids stay stable across runs:
they are the pairing key.

### Paired bootstrap (continuous metrics)

`paired_bootstrap_test(scores_a, scores_b)` tests `H0: mean(b - a) = 0`:

1. Compute the observed mean difference (the effect).
2. Shift the differences to mean zero — sampling *under the null* — and
   bootstrap the mean of the shifted sample.[^efron93]
3. The two-sided p-value is the fraction of null means at least as extreme as
   the observed effect, with the Phipson–Smyth estimator (below).
4. The CI on the effect is the BCa interval on the *raw* differences.

```python
from holdout.stats import paired_bootstrap_test

r = paired_bootstrap_test(a, b, seed=7)
print(r)         # Δ=+0.022 [95% CI +0.011, +0.032], p=9.999e-05 (paired-bootstrap, n=200)
print(r.detail)  # H0: mean(b - a) = 0, two-sided, shifted-null bootstrap
```

### Exact McNemar (binary metrics)

For paired binary outcomes, only **discordant pairs** carry information about a
difference[^mcnemar]: cases where exactly one of the two runs scored 1. Let
`n01` be the cases the candidate fixed and `n10` the cases it broke. Under H0,
each discordant pair is a fair coin, so `min(n01, n10)` follows a
Binomial(m, 1/2) over the `m = n01 + n10` discordant pairs, and the
exact-conditional two-sided p-value is `min(1, 2 * P(X <= min(n01, n10)))`.
The effect is the difference in proportions, `(n01 - n10) / n`, with a BCa
bootstrap CI on the per-pair differences. No discordant pairs means no
information, so p = 1 — not a pass certificate.

```python
from holdout.stats import mcnemar_test

r = mcnemar_test(base, cand, seed=7)
print(r)         # Δ=-0.087 [95% CI -0.163, -0.037], p=0.01563 (mcnemar-exact, n=80)
print(r.detail)  # discordant pairs: improved=0, regressed=7
```

!!! note "Known conservatism"
    The exact-conditional McNemar test is conservative: its true type-I error
    is below the nominal alpha, so it gives up a little power compared to the
    mid-p or asymptotic variants.[^fagerland] holdout chooses it deliberately —
    a CI gate should never be *anti*-conservative — but be aware that with few
    discordant pairs it needs a lopsided split to fire (with zero fixes, six
    breaks is the minimum for p <= 0.05: `2 * 0.5^6 = 0.03125`).

### Sign-flip permutation

`permutation_test(scores_a, scores_b)` uses Fisher's randomization argument:
under H0 the two runs are exchangeable within each pair, so each difference's
sign is a fair coin.[^good] When the full `2^n` sign assignments number at most
`n_resamples`, the test **enumerates them exactly** — at n=12 that is 4,096
assignments, all checked:

```python
from holdout.stats import permutation_test

r = permutation_test(a[:12], b[:12], seed=7)
print(r)         # Δ=+0.044 [95% CI -0.012, +0.092], p=0.1309 (permutation-exact, n=12)
print(r.detail)  # exact enumeration of 2^12 sign assignments
```

The exact enumeration includes the observed assignment, so its p-value can
never be zero. Above the enumeration threshold the test samples random sign
flips (`permutation-mc`) and uses the Phipson–Smyth estimator.

### Why a p-value is never 0

A Monte-Carlo p-value estimated as `r / B` (the fraction of resamples at least
as extreme as observed) can come out exactly zero, which is a lie — you only
checked B resamples. holdout uses the Phipson–Smyth estimator
`(r + 1) / (B + 1)` for every Monte-Carlo p-value,[^phipson] which is the exact
probability statement "the observed result is one of the extreme ones among
B+1 exchangeable arrangements". With the default B = 10,000, the smallest
reportable p is about 1e-4. If you see `p=9.999e-05`, that means "more extreme
than everything we sampled", not "probability zero".

## Multiple-comparison correction

Run an eval with five metrics at alpha = 0.05 and the chance of at least one
fluke "significant" result under the null is `1 - 0.95^5 ≈ 23%`. Every verdict
in holdout is therefore issued on **corrected** p-values, never raw ones.

Two procedures are available:

| | Benjamini–Hochberg (default) | Holm–Bonferroni |
|---|---|---|
| Controls | false discovery rate (FDR): the expected *fraction* of false alarms among rejections[^bh] | family-wise error rate (FWER): the probability of *any* false alarm[^holm] |
| Procedure | step-up: sort ascending, `p_(i) * m / i`, cumulative min from the top | step-down: sort ascending, `(m - i + 1) * p_(i)`, cumulative max from the bottom |
| Character | more power; a small, controlled fraction of flagged metrics may be flukes | stricter; use when any single false alarm is unacceptable |

BH is the right default for an eval gate: you are scanning several metrics for
movement and can tolerate the occasional false flag at a known rate, in
exchange for actually detecting real regressions. Pass `correction="holm"` to
`compare()` when a single false regression alarm is expensive (e.g. it blocks a
release train), or `correction="none"` when there is exactly one metric (the
corrections are identities at m=1 anyway).

BH's FDR guarantee holds under independence or positive regression dependence
of the test statistics; metrics on the same runs are typically positively
dependent, which is the favorable case.

## Power and minimum detectable effect

The silent failure mode of LLM evals: run 50 cases, see "no significant
change", conclude safety. If the eval never had the power to detect the
regression you were asked about, that null result is meaningless — absence of
evidence produced by an instrument too blunt to find it.

holdout makes detectability explicit. For a two-sided paired test on the mean
of per-pair differences, the normal-approximation formulas[^chow] are:

```text
mde = (z_{1-alpha/2} + z_{power}) * sd_diff / sqrt(n)        # what n can detect
n   = ceil(((z_{1-alpha/2} + z_{power}) * sd_diff / mde)^2)  # what detecting mde costs
```

`sd_diff` is the standard deviation of per-pair differences. Measure it from a
pilot comparison with `sd_diff_from_scores(a, b)`, or assume it for binary
metrics with `paired_binary_sd(p01, p10)`, where `p01`/`p10` are the expected
fix/break rates — for paired binary outcomes
`Var(d) = p01 + p10 - (p01 - p10)^2`.[^miettinen] Total discordance of 10–20%
is typical for prompt changes on a stable eval.

```python
from holdout.stats import minimum_detectable_effect, paired_binary_sd, required_sample_size

sd = paired_binary_sd(0.05, 0.05)            # 10% total discordance
print(minimum_detectable_effect(200, sd))
# n=200 pairs detects |Δ| >= 0.0626 at alpha=0.05 with power 0.8 (sd_diff=0.3162)

print(required_sample_size(0.05, 0.32))
# n=322 pairs detects |Δ| >= 0.0500 at alpha=0.05 with power 0.8 (sd_diff=0.3200)
```

Read the first line carefully: a 200-case eval, at typical discordance, cannot
reliably see anything smaller than a ~6-point swing. If a 3-point regression
matters to you, you need roughly 4x the cases — power scales with `sqrt(n)`,
so halving the MDE quadruples the bill. This is also what
`assert_adequately_powered(baseline, candidate, mde=...)` checks, and the same
calculator is available as `holdout power` on the CLI.

!!! warning "Approximation, not gospel"
    These are normal-approximation formulas. They are accurate for moderate n
    and effect sizes, and slightly optimistic for very small n or very rare
    discordance (where the binomial's discreteness bites). Treat the result as
    a planning number with one significant figure of authority — it will tell
    you "you need ~300 cases, not 50", not "you need exactly 322".

## What the engine does not do

- It does not test non-inferiority margins ("no worse than 2 points"); the
  gate is a two-sided test of "no difference" with the verdict sign taken from
  the effect.
- It does not model per-case difficulty or clustering; cases are treated as
  exchangeable units, which is why [near-duplicate detection](leakage.md#near-duplicates-and-effective-n)
  matters — duplicates violate that assumption and silently narrow the
  intervals.
- It does not correct across *time* — twenty sequential comparisons against the
  same eval are twenty looks, which is what the
  [holdout ledger](leakage.md#the-holdoutledger) counts.

[^efron87]: Efron, B. (1987). "Better Bootstrap Confidence Intervals". *Journal of the American Statistical Association*, 82(397), 171–185.
[^efron93]: Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. Ch. 13 (percentile intervals), ch. 14 (BCa), ch. 16 (bootstrap hypothesis testing via the shifted null).
[^mcnemar]: McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages". *Psychometrika*, 12(2), 153–157.
[^fagerland]: Fagerland, M. W., Lydersen, S. & Laake, P. (2013). "The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional". *BMC Medical Research Methodology*, 13:91.
[^good]: Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses*, 3rd ed. Springer.
[^phipson]: Phipson, B. & Smyth, G. K. (2010). "Permutation p-values should never be zero". *Statistical Applications in Genetics and Molecular Biology*, 9(1).
[^bh]: Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing". *Journal of the Royal Statistical Society, Series B*, 57(1), 289–300.
[^holm]: Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test Procedure". *Scandinavian Journal of Statistics*, 6(2), 65–70.
[^chow]: Chow, S.-C., Shao, J. & Wang, H. (2008). *Sample Size Calculations in Clinical Research*, 2nd ed. Chapman & Hall/CRC. Ch. 3 (paired designs).
[^miettinen]: Miettinen, O. S. (1968). "The matched pairs design in the case of all-or-none responses". *Biometrics*, 24(2), 339–352.
