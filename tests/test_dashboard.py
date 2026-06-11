"""Tests for the local dashboard server: read-only JSON API over the store."""

import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.runner import run
from holdout.dashboard.server import make_server
from holdout.providers.static import StaticTarget
from holdout.scorers import ExactMatch
from holdout.store import RunStore

N = 30


def _store_with_runs(root: Path) -> tuple[RunStore, str, str]:
    cases = [Case(input=f"q{i}", reference="yes", id=f"c{i:02d}") for i in range(N)]
    ev = Eval("dash-test", cases, [ExactMatch()])
    good = StaticTarget({f"q{i}": "yes" for i in range(N)}, name="good")
    bad = StaticTarget({f"q{i}": ("no" if i < 8 else "yes") for i in range(N)}, name="bad")
    store = RunStore(root)
    a = run(ev, target=good, seed=7)
    b = run(ev, target=bad, seed=7)
    store.save(a)
    store.save(b)
    return store, a.run_id, b.run_id


@pytest.fixture()
def server(tmp_path: Path) -> Iterator[tuple[str, str, str]]:
    store, a, b = _store_with_runs(tmp_path)
    srv = make_server(store, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", a, b
    finally:
        srv.shutdown()
        srv.server_close()


def test_meta(server: tuple[str, str, str]) -> None:
    base, _, _ = server
    meta = httpx.get(f"{base}/api/meta").json()
    assert meta["n_runs"] == 2
    assert meta["version"]


def test_runs_listing_includes_metrics_with_intervals(server: tuple[str, str, str]) -> None:
    base, _, _ = server
    payload = httpx.get(f"{base}/api/runs").json()
    assert len(payload["runs"]) == 2
    for entry in payload["runs"]:
        est = entry["metrics"]["exact_match"]
        assert {"value", "ci_low", "ci_high", "n", "level", "method"} <= est.keys()
        assert est["ci_low"] <= est["value"] <= est["ci_high"]


def test_run_detail(server: tuple[str, str, str]) -> None:
    base, a, _ = server
    detail = httpx.get(f"{base}/api/runs/{a[:12]}").json()
    assert detail["run_id"] == a
    assert len(detail["results"]) == N
    assert "exact_match" in detail["metrics"]


def test_compare_endpoint(server: tuple[str, str, str]) -> None:
    base, a, b = server
    cmp = httpx.get(f"{base}/api/compare", params={"baseline": a, "candidate": b}).json()
    assert cmp["verdict"] == "regressed"
    assert cmp["comparisons"][0]["result"]["test"] == "mcnemar-exact"

    missing = httpx.get(f"{base}/api/compare", params={"baseline": a})
    assert missing.status_code == 400


def test_ledger_endpoint(server: tuple[str, str, str]) -> None:
    base, _, _ = server
    payload = httpx.get(f"{base}/api/ledger").json()
    (entry,) = payload["evals"]
    assert entry["eval_name"] == "dash-test"
    assert entry["level"] == "ok"


def test_unknown_run_is_404_and_unknown_endpoint_is_404(server: tuple[str, str, str]) -> None:
    base, _, _ = server
    assert httpx.get(f"{base}/api/runs/feedfacecafe").status_code == 404
    assert httpx.get(f"{base}/api/nope").status_code == 404


def test_server_is_read_only(server: tuple[str, str, str]) -> None:
    base, a, _b = server
    # No mutating verb is implemented anywhere.
    assert httpx.post(f"{base}/api/runs", json={}).status_code == 501
    assert httpx.delete(f"{base}/api/runs/{a}").status_code == 501
    assert httpx.put(f"{base}/api/compare").status_code == 501


def test_static_root_responds_even_without_built_assets(server: tuple[str, str, str]) -> None:
    base, _, _ = server
    resp = httpx.get(f"{base}/")
    # Either the bundled SPA (200 with html) or the helpful 503 explainer.
    assert resp.status_code in (200, 503)
    assert "html" in resp.headers["content-type"]
