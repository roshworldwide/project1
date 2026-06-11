"""The holdout pytest plugin: options, markers, and fixtures.

Auto-registered via the ``pytest11`` entry point when holdout is
installed. Provides:

- ``--holdout-store`` / ``--holdout-seed`` command-line options,
- the ``llm_eval`` marker (deselect real model calls with
  ``-m "not llm_eval"``),
- ``holdout_store`` and ``holdout_seed`` fixtures.

The assertions live in :mod:`holdout.testing` and work with or without
this plugin.
"""

import pytest

from holdout.store.run_store import RunStore


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register holdout command-line options."""
    group = parser.getgroup("holdout", "holdout LLM evaluation")
    group.addoption(
        "--holdout-store",
        default=".holdout",
        help="run store directory (default: .holdout)",
    )
    group.addoption(
        "--holdout-seed",
        type=int,
        default=0,
        help="seed for eval runs and resampling (default: 0)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the llm_eval marker."""
    config.addinivalue_line(
        "markers",
        "llm_eval: marks a test as an LLM evaluation (may call real model targets)",
    )


@pytest.fixture(scope="session")
def holdout_store(request: pytest.FixtureRequest) -> RunStore:
    """Return the run store configured by ``--holdout-store``."""
    return RunStore(str(request.config.getoption("--holdout-store")))


@pytest.fixture(scope="session")
def holdout_seed(request: pytest.FixtureRequest) -> int:
    """Return the seed configured by ``--holdout-seed``."""
    return int(str(request.config.getoption("--holdout-seed")))
