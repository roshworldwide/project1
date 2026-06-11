"""Deterministic targets for the dogfood eval gate (no model calls in CI)."""

from holdout.providers.static import StaticTarget

# The "current production prompt": answers arithmetic correctly.
candidate = StaticTarget(
    {f"What is {i} + {i}?": str(i + i) for i in range(40)},
    name="examples-ci-candidate",
)
