"""Smoke tests for packaging metadata."""

import holdout


def test_version_is_set() -> None:
    assert holdout.__version__


def test_docstring_states_the_contract() -> None:
    # The cultural rule is part of the package's public documentation.
    assert holdout.__doc__ is not None
    assert "naked point estimate" in holdout.__doc__
