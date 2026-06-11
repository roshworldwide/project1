"""The run store: versioned, content-addressed persistence for Runs.

Layout under the store root (default ``.holdout/``):

- ``runs/<run_id>.json`` — one artifact per run, named by its content hash,
  so saving is idempotent and two stores can be merged by copying files.
- ``index.sqlite3`` — a rebuildable index for fast listing; the JSON
  artifacts are the source of truth and :meth:`RunStore.reindex` restores
  the index from them at any time.
"""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from holdout.core.run import Run

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    eval_name   TEXT NOT NULL,
    target_name TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    n_cases     INTEGER NOT NULL,
    n_errors    INTEGER NOT NULL,
    seed        INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runs_eval ON runs (eval_name, created_at);
"""


@dataclass(frozen=True, slots=True)
class StoredRunInfo:
    """A lightweight index row describing one stored run."""

    run_id: str
    eval_name: str
    target_name: str
    created_at: str
    n_cases: int
    n_errors: int
    seed: int | None

    @property
    def short_run_id(self) -> str:
        """Twelve-character display prefix of the run id."""
        return self.run_id[:12]


class RunStore:
    """Save, list, and load Runs from a local directory.

    Parameters
    ----------
    root
        Store directory (created if missing). Default ``".holdout"``.
    """

    def __init__(self, root: str | Path = ".holdout") -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "index.sqlite3"
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def save(self, run: Run) -> Path:
        """Persist ``run``; idempotent because the artifact is content-addressed.

        Returns the path of the JSON artifact.
        """
        path = self.runs_dir / f"{run.run_id}.json"
        if not path.exists():
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(run.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)  # atomic on POSIX: a reader never sees a partial file
        self._index(run)
        return path

    def _index(self, run: Run) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, eval_name, target_name, created_at, n_cases, n_errors, seed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.eval_name,
                    run.target_name,
                    run.created_at,
                    len(run.results),
                    run.n_errors,
                    run.seed,
                ),
            )

    def load(self, ref: str) -> Run:
        """Load a run by full run id or unambiguous prefix.

        Raises
        ------
        KeyError
            If no run matches ``ref``, or if the prefix is ambiguous (the
            message lists the candidates).
        """
        if not ref:
            raise KeyError("empty run reference")
        exact = self.runs_dir / f"{ref}.json"
        if exact.exists():
            return self._read(exact)
        matches = sorted(self.runs_dir.glob(f"{ref}*.json"))
        if not matches:
            raise KeyError(f"no run matching {ref!r} in {self.root}")
        if len(matches) > 1:
            ids = ", ".join(p.stem[:12] for p in matches[:5])
            raise KeyError(f"run reference {ref!r} is ambiguous: matches {ids}")
        return self._read(matches[0])

    def _read(self, path: Path) -> Run:
        data = json.loads(path.read_text(encoding="utf-8"))
        run = Run.from_dict(data)
        stored_id = path.stem
        if run.run_id != stored_id:
            raise ValueError(
                f"artifact {path.name} fails content-address verification: recomputed "
                f"run_id {run.run_id[:12]}... does not match the filename. The file was "
                "modified after it was written."
            )
        return run

    def runs(
        self, *, eval_name: str | None = None, limit: int | None = None
    ) -> list[StoredRunInfo]:
        """List stored runs, newest first, optionally filtered by eval name."""
        query = (
            "SELECT run_id, eval_name, target_name, created_at, n_cases, n_errors, seed FROM runs"
        )
        params: list[object] = []
        if eval_name is not None:
            query += " WHERE eval_name = ?"
            params.append(eval_name)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [StoredRunInfo(*row) for row in rows]

    def latest(self, *, eval_name: str | None = None, target_name: str | None = None) -> Run | None:
        """Load the most recent run, optionally filtered by eval/target name."""
        for info in self.runs(eval_name=eval_name):
            if target_name is None or info.target_name == target_name:
                return self.load(info.run_id)
        return None

    def reindex(self) -> int:
        """Rebuild the SQLite index from the JSON artifacts; returns row count."""
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM runs")
        count = 0
        for path in sorted(self.runs_dir.glob("*.json")):
            self._index(self._read(path))
            count += 1
        return count

    def __len__(self) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return int(row[0])

    def __repr__(self) -> str:
        return f"RunStore(root={str(self.root)!r}, runs={len(self)})"
