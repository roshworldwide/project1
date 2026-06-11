# Benchmark results

Measured by `python benchmarks/overhead.py` — nothing below is hardcoded;
rerun it on your machine.

## 2026-06-11 · Apple silicon (arm64), macOS 26.5.1, Python 3.11.15

1,000 cases, ExactMatch, `max_concurrency=64`:

| measurement | result |
|---|---|
| framework overhead (instant target) | 0.006 s wall — **6 µs/case** |
| 50 ms-latency target | 0.841 s wall vs 0.781 s theoretical floor — **1.08×** |
| BCa bootstrap (n=1000, B=10,000) | 65.8 ms |
| full `compare()` (1,000 paired cases) | 81.9 ms |

Interpretation: a 1,000-case eval is I/O-bound — wall time is your
provider's latency divided by your concurrency, not the framework. The
statistics add well under 100 ms on top of any real eval.
