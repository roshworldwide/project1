"""Tests for the run store: content-addressed persistence, indexing, integrity.

Fully offline: every Run comes from StaticTarget plus local scorers.
"""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.run import Run
from holdout.core.runner import run
from holdout.core.scoring import Score, Scorer
from holdout.providers.static import StaticTarget
from holdout.scorers.exact import ExactMatch
from holdout.store import RunStore, StoredRunInfo

QA: dict[str, str] = {
    "capital of France?": "Paris",
    "2+2?": "4",
    "color of the sky?": "blue",
    "opposite of hot?": "cold",
}

# One wrong answer => exact_match 0.75, non-degenerate intervals.
RESPONSES: dict[str, str] = {**QA, "2+2?": "5"}


class LengthRatio(Scorer):
    """Continuous scorer: output length scaled into [0, 1]."""

    @property
    def name(self) -> str:
        return "length_ratio"

    async def score(self, case: Case, output: str) -> Score:
        return Score(value=min(len(output), 16) / 16.0, kind="continuous")


def make_eval(name: str = "smoke") -> Eval:
    cases = [Case(input=q, reference=a) for q, a in QA.items()]
    return Eval(name=name, cases=cases, scorers=[ExactMatch(), LengthRatio()])


def make_run(
    *,
    seed: int = 0,
    eval_name: str = "smoke",
    target_name: str = "static",
) -> Run:
    target = StaticTarget(RESPONSES, name=target_name)
    return run(make_eval(eval_name), target=target, seed=seed)


def two_runs_sharing_first_hex_char() -> tuple[Run, Run]:
    """Deterministically find two runs whose run ids share their first char.

    run_id is a SHA-256 hex digest, so the first character has 16 possible
    values: by pigeonhole, at most 17 seeds are needed. Run ids are
    deterministic, so the loop always finds the same pair.
    """
    seen: dict[str, Run] = {}
    for seed in range(32):
        r = make_run(seed=seed)
        first = r.run_id[0]
        if first in seen:
            return seen[first], r
        seen[first] = r
    raise AssertionError("unreachable: 32 distinct runs must collide on 16 hex chars")


# --- save / load round-trip ---------------------------------------------------


def test_save_load_round_trip_preserves_run_id_and_metrics(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    r = make_run(seed=7)

    path = store.save(r)
    assert path.exists()
    assert path.name == f"{r.run_id}.json"

    loaded = store.load(r.run_id)
    assert loaded.run_id == r.run_id
    assert loaded.metrics(n_resamples=200) == r.metrics(n_resamples=200)
    assert loaded.created_at == r.created_at
    assert loaded.seed == r.seed
    assert [cr.case_id for cr in loaded.results] == [cr.case_id for cr in r.results]


def test_save_is_idempotent(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    r = make_run(seed=1)

    p1 = store.save(r)
    files_after_first = sorted(store.runs_dir.iterdir())
    p2 = store.save(r)
    files_after_second = sorted(store.runs_dir.iterdir())

    assert p1 == p2
    assert files_after_first == files_after_second
    assert len(files_after_second) == 1  # no duplicates, no leftover tmp files
    assert len(store) == 1


# --- load by reference --------------------------------------------------------


def test_load_by_unique_prefix(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    r1 = make_run(seed=0)
    r2 = make_run(seed=1)
    store.save(r1)
    store.save(r2)

    prefix = r1.run_id[:16]
    assert not r2.run_id.startswith(prefix)  # deterministic: ids are content hashes
    assert store.load(prefix).run_id == r1.run_id


def test_ambiguous_prefix_raises_key_error_listing_candidates(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    ra, rb = two_runs_sharing_first_hex_char()
    shared = ra.run_id[0]
    assert rb.run_id[0] == shared
    store.save(ra)
    store.save(rb)

    with pytest.raises(KeyError, match="ambiguous") as excinfo:
        store.load(shared)
    message = str(excinfo.value)
    assert ra.run_id[:12] in message
    assert rb.run_id[:12] in message


def test_unknown_ref_raises_key_error(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    store.save(make_run(seed=0))
    with pytest.raises(KeyError, match="no run matching"):
        store.load("ffffffffffffdeadbeef")


def test_empty_ref_raises_key_error(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    with pytest.raises(KeyError, match="empty run reference"):
        store.load("")


# --- runs(): listing, ordering, filters ----------------------------------------


def stamped_runs() -> list[Run]:
    """Three distinct runs with controlled, strictly increasing created_at."""
    base = [make_run(seed=s) for s in (0, 1, 2)]
    stamps = [
        "2026-01-01T00:00:00+00:00",
        "2026-01-02T00:00:00+00:00",
        "2026-01-03T00:00:00+00:00",
    ]
    return [replace(r, created_at=ts) for r, ts in zip(base, stamps, strict=True)]


def test_runs_lists_newest_first(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    oldest, middle, newest = stamped_runs()
    for r in (middle, newest, oldest):  # save order deliberately shuffled
        store.save(r)

    infos = store.runs()
    assert [i.run_id for i in infos] == [newest.run_id, middle.run_id, oldest.run_id]
    assert [i.created_at for i in infos] == sorted((i.created_at for i in infos), reverse=True)


def test_runs_eval_name_filter_and_limit(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    oldest, middle, newest = stamped_runs()
    other = replace(make_run(seed=3, eval_name="other"), created_at="2026-01-04T00:00:00+00:00")
    for r in (oldest, middle, newest, other):
        store.save(r)

    smoke = store.runs(eval_name="smoke")
    assert [i.run_id for i in smoke] == [newest.run_id, middle.run_id, oldest.run_id]
    assert all(i.eval_name == "smoke" for i in smoke)

    assert [i.run_id for i in store.runs(eval_name="other")] == [other.run_id]
    assert store.runs(eval_name="nope") == []

    limited = store.runs(limit=2)
    assert [i.run_id for i in limited] == [other.run_id, newest.run_id]
    assert [i.run_id for i in store.runs(eval_name="smoke", limit=1)] == [newest.run_id]


def test_runs_info_fields_and_short_run_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    r = make_run(seed=5)
    store.save(r)

    info = store.runs()[0]
    assert isinstance(info, StoredRunInfo)
    assert info.run_id == r.run_id
    assert info.eval_name == "smoke"
    assert info.target_name == "static"
    assert info.n_cases == len(QA)
    assert info.n_errors == 0
    assert info.seed == 5
    assert len(info.short_run_id) == 12
    assert r.run_id.startswith(info.short_run_id)


# --- latest() -------------------------------------------------------------------


def test_latest_with_filters_and_none_when_nothing_matches(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    assert store.latest() is None

    oldest, middle, newest = stamped_runs()
    alt = replace(
        make_run(seed=9, eval_name="other", target_name="alt"),
        created_at="2025-12-31T00:00:00+00:00",  # older than every smoke run
    )
    for r in (oldest, middle, newest, alt):
        store.save(r)

    overall = store.latest()
    assert overall is not None
    assert overall.run_id == newest.run_id

    by_eval = store.latest(eval_name="other")
    assert by_eval is not None
    assert by_eval.run_id == alt.run_id

    by_target = store.latest(target_name="alt")
    assert by_target is not None
    assert by_target.run_id == alt.run_id

    assert store.latest(eval_name="smoke", target_name="alt") is None
    assert store.latest(eval_name="nope") is None
    assert store.latest(target_name="nope") is None


# --- reindex() --------------------------------------------------------------------


def test_reindex_rebuilds_after_index_deletion(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = RunStore(root)
    r1 = make_run(seed=0)
    r2 = make_run(seed=1)
    store.save(r1)
    store.save(r2)

    (root / "index.sqlite3").unlink()
    rebuilt = RunStore(root)
    assert len(rebuilt) == 0  # fresh index knows nothing yet

    assert rebuilt.reindex() == 2
    assert {i.run_id for i in rebuilt.runs()} == {r1.run_id, r2.run_id}
    assert rebuilt.load(r1.run_id).run_id == r1.run_id


def test_reindex_merges_artifacts_copied_from_another_store(tmp_path: Path) -> None:
    store_a = RunStore(tmp_path / "a")
    store_b = RunStore(tmp_path / "b")
    ra = make_run(seed=0)
    rb = make_run(seed=1)
    artifact_a = store_a.save(ra)
    store_b.save(rb)

    shutil.copy(artifact_a, store_b.runs_dir / artifact_a.name)

    assert store_b.reindex() == 2
    assert {i.run_id for i in store_b.runs()} == {ra.run_id, rb.run_id}
    assert store_b.load(ra.run_id).run_id == ra.run_id
    assert store_b.load(rb.run_id).run_id == rb.run_id


# --- tamper detection ---------------------------------------------------------------


def test_tampered_artifact_fails_content_address_verification(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "store")
    r = make_run(seed=0)
    path = store.save(r)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data["results"][0]["output"], str)
    data["results"][0]["output"] = "tampered output"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="content-address verification"):
        store.load(r.run_id)


# --- dunder methods -------------------------------------------------------------------


def test_len_and_repr(tmp_path: Path) -> None:
    root = tmp_path / "store"
    store = RunStore(root)
    assert len(store) == 0
    assert repr(store) == f"RunStore(root={str(root)!r}, runs=0)"

    store.save(make_run(seed=0))
    store.save(make_run(seed=1))
    assert len(store) == 2
    assert repr(store) == f"RunStore(root={str(root)!r}, runs=2)"
