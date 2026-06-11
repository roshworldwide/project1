"""The holdout-discipline ledger — counting how many times you have peeked.

Every time a team tunes a prompt against the same eval set, the eval stops
measuring generalization and starts measuring memorization-by-iteration —
the silent killer quant finance calls backtest overfitting. Each adaptive
look at the same data biases the next decision (Dwork et al. 2015, "The
reusable holdout", *Science* 349(6248); Russo & Zou 2016, "Controlling
bias in adaptive data analysis", AISTATS).

The ledger counts comparisons per eval fingerprint and warns when a budget
is spent. It cannot stop you — it makes the peeking visible.
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

DisciplineLevel = Literal["ok", "caution", "overfit-risk"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_fingerprint TEXT NOT NULL,
    eval_name        TEXT NOT NULL,
    kind             TEXT NOT NULL,
    context          TEXT,
    used_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uses_fp ON uses (eval_fingerprint);
"""


@dataclass(frozen=True, slots=True)
class DisciplineReport:
    """How worn out an eval set is.

    Parameters
    ----------
    eval_name, eval_fingerprint
        The eval in question.
    uses
        Recorded tuning/comparison uses so far.
    budget
        The use budget you granted this eval.
    level
        ``ok`` (under half the budget), ``caution`` (over half), or
        ``overfit-risk`` (budget spent).
    """

    eval_name: str
    eval_fingerprint: str
    uses: int
    budget: int
    level: DisciplineLevel

    def __str__(self) -> str:
        msg = (
            f"eval {self.eval_name!r} has been used {self.uses} time(s) "
            f"of a budget of {self.budget} [{self.level}]"
        )
        if self.level == "overfit-risk":
            msg += (
                " — results on this eval now reflect tuning-to-the-test as much as "
                "quality; cut a fresh holdout set (Dwork et al. 2015)"
            )
        elif self.level == "caution":
            msg += " — plan a fresh holdout set before the budget runs out"
        return msg

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "eval_name": self.eval_name,
            "eval_fingerprint": self.eval_fingerprint,
            "uses": self.uses,
            "budget": self.budget,
            "level": self.level,
        }


class HoldoutLedger:
    """Counts adaptive uses of each eval set, persisted next to the run store.

    Parameters
    ----------
    root
        Directory holding ``ledger.sqlite3`` (default ``".holdout"`` — the
        same root the run store uses).
    """

    def __init__(self, root: str | Path = ".holdout") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "ledger.sqlite3"
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record_use(
        self,
        eval_fingerprint: str,
        eval_name: str,
        *,
        kind: str = "compare",
        context: str | None = None,
    ) -> int:
        """Record one adaptive use; returns the total recorded so far.

        Parameters
        ----------
        eval_fingerprint, eval_name
            Identity of the eval that was consulted.
        kind
            What kind of look this was (e.g. ``"compare"``, ``"tune"``).
        context
            Free-form note (a branch name, a prompt version, a PR number).
        """
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO uses (eval_fingerprint, eval_name, kind, context, used_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    eval_fingerprint,
                    eval_name,
                    kind,
                    context,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return self.uses(eval_fingerprint)

    def uses(self, eval_fingerprint: str) -> int:
        """Return the number of recorded uses of this eval."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM uses WHERE eval_fingerprint = ?", (eval_fingerprint,)
            ).fetchone()
        return int(row[0])

    def check(self, eval_fingerprint: str, eval_name: str, *, budget: int = 20) -> DisciplineReport:
        """Report how much of the eval's use budget is spent.

        Parameters
        ----------
        eval_fingerprint, eval_name
            Identity of the eval.
        budget
            How many adaptive looks you grant this eval before treating its
            verdicts as tuned-to-the-test. Default 20 — generous; quant
            desks would say lower.
        """
        if budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        n = self.uses(eval_fingerprint)
        level: DisciplineLevel
        if n >= budget:
            level = "overfit-risk"
        elif n * 2 >= budget:
            level = "caution"
        else:
            level = "ok"
        return DisciplineReport(
            eval_name=eval_name,
            eval_fingerprint=eval_fingerprint,
            uses=n,
            budget=budget,
            level=level,
        )
