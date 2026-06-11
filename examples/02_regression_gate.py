"""The regression gate: noise is not a regression, and a regression is not noise.

Two scenarios, same eval, fully offline:

1. The candidate flips 2 of 60 cases — a naked score drop (0.967 -> 0.933)
   that threshold-based tools would flag. The paired test correctly calls
   it noise.
2. The candidate flips 12 of 60 cases — a real regression. The gate fails
   with the effect size, interval, and corrected p-value.

Run: python examples/02_regression_gate.py
"""

from holdout import Case, Eval, run
from holdout.providers import StaticTarget
from holdout.regression import compare
from holdout.scorers import ExactMatch
from holdout.testing import assert_no_regression

N = 60
cases = [Case(input=f"task {i}", reference="done", id=f"t{i:02d}") for i in range(N)]
ev = Eval("worker-tasks", cases, [ExactMatch()])


def target(wrong: set[int], name: str) -> StaticTarget:
    return StaticTarget(
        {f"task {i}": ("failed" if i in wrong else "done") for i in range(N)}, name=name
    )


baseline = run(ev, target=target(set(), "prompt-v1"), seed=7)

# --- scenario 1: small wobble, not significant -----------------------------
wobble = run(ev, target=target({3, 41}, "prompt-v2-wobble"), seed=7)
verdict = compare(baseline, wobble, seed=7)
print(verdict.summary())
print()
assert_no_regression(baseline, wobble, seed=7)  # passes: noise is not a regression
print("scenario 1: naked score dropped 0.967 -> 0.933, but the gate correctly stays green\n")

# --- scenario 2: real regression --------------------------------------------
broken = run(ev, target=target(set(range(12)), "prompt-v2-broken"), seed=7)
print(compare(baseline, broken, seed=7).summary())
print()
try:
    assert_no_regression(baseline, broken, seed=7)
except AssertionError as exc:
    print(f"scenario 2: the gate fails, with evidence:\n{exc}")
