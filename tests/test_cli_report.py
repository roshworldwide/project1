"""Tests for the CLI (holdout.cli) and the HTML report (holdout.report)."""

import json
from pathlib import Path

import pytest

from holdout.cli.main import main
from holdout.cli.refs import load_eval, load_target, make_scorer
from holdout.core.case import Case
from holdout.core.evalset import Eval
from holdout.core.runner import run as run_eval
from holdout.providers.static import StaticTarget
from holdout.regression import compare
from holdout.report.html import render_comparison_report, render_run_report
from holdout.scorers import ExactMatch, RegexMatch
from holdout.store import RunStore

TARGETS_MOD = """
from holdout.providers.static import StaticTarget

good = StaticTarget({f"q{i}": "yes" for i in range(40)}, name="good")
bad = StaticTarget({f"q{i}": ("no" if i < 10 else "yes") for i in range(40)}, name="bad")
not_a_target = 42
"""


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp project: cases.jsonl + an importable targets module."""
    lines = [
        json.dumps({"input": f"q{i}", "reference": "yes", "id": f"c{i:03d}"}) for i in range(40)
    ]
    (tmp_path / "cases.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "cli_targets_mod.py").write_text(TARGETS_MOD, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_both(project: Path) -> tuple[str, str]:
    """Run good and bad targets via the CLI; return their run id prefixes."""
    assert main(["run", "cases.jsonl", "--target", "cli_targets_mod:good", "--seed", "7"]) == 0
    assert main(["run", "cases.jsonl", "--target", "cli_targets_mod:bad", "--seed", "7"]) == 0
    infos = RunStore(project / ".holdout").runs()
    by_target = {i.target_name: i.run_id for i in infos}
    return by_target["good"], by_target["bad"]


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def test_run_stores_and_prints_ci(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", "cases.jsonl", "--target", "cli_targets_mod:good", "--seed", "7"])
    out = capsys.readouterr().out
    assert code == 0
    assert "exact_match" in out
    assert "bootstrap-bca" in out
    assert "saved" in out
    assert len(RunStore(project / ".holdout").runs()) == 1


def test_compare_exit_codes_and_ledger(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good, bad = _run_both(project)
    capsys.readouterr()

    code = main(["compare", good[:12], bad[:12]])
    out = capsys.readouterr().out
    assert code == 1  # regression detected
    assert "REGRESSED" in out
    assert "mcnemar" in out
    assert "has been used 1 time(s)" in out

    # Identical runs: exit 0; --no-ledger leaves the count unchanged.
    code = main(["compare", good[:12], good[:12], "--no-ledger"])
    out = capsys.readouterr().out
    assert code == 0
    assert "no sig." in out
    assert "has been used" not in out


def test_list_shows_runs(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run_both(project)
    capsys.readouterr()
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "good" in out and "bad" in out


def test_report_single_and_comparison(project: Path) -> None:
    good, bad = _run_both(project)
    assert main(["report", good[:12], "-o", "run.html"]) == 0
    single = (project / "run.html").read_text(encoding="utf-8")
    assert single.startswith("<!doctype html")
    assert "exact_match" in single
    assert "#C9A876" in single  # Starlight Gold
    assert "http" not in single  # fully self-contained, air-gap safe

    assert main(["report", good[:12], bad[:12], "-o", "cmp.html"]) == 0
    cmp_html = (project / "cmp.html").read_text(encoding="utf-8")
    assert "REGRESSED" in cmp_html
    assert "baseline" in cmp_html and "candidate" in cmp_html


def test_report_wrong_arity_errors(project: Path) -> None:
    good, bad = _run_both(project)
    assert main(["report", good[:12], bad[:12], good[:12]]) == 2


def test_power_paths(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power", "--n", "200", "--sd", "0.35"]) == 0
    assert "detects |Δ| >=" in capsys.readouterr().out

    assert main(["power", "--mde", "0.05", "--p01", "0.05", "--p10", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "sd_diff = 0.3162" in out
    assert "n=" in out

    assert main(["power", "--sd", "0.35"]) == 2  # neither --n nor --mde
    assert main(["power", "--n", "10", "--mde", "0.1", "--sd", "0.35"]) == 2  # both
    assert main(["power", "--n", "10"]) == 2  # no sd and no p01/p10


def test_check_clean_and_contaminated(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", "cases.jsonl", "--corpus-text", "totally unrelated prompt"]) == 0
    capsys.readouterr()
    code = main(["check", "cases.jsonl", "--corpus-text", "the answer to q3 is yes obviously"])
    out = capsys.readouterr().out
    assert code == 1
    assert "exact-substring" in out


def test_unknown_run_ref_is_a_clean_error(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", "deadbeef", "cafebabe"]) == 2
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------


def test_make_scorer_specs() -> None:
    assert isinstance(make_scorer("exact"), ExactMatch)
    strict = make_scorer("exact-strict")
    assert isinstance(strict, ExactMatch)
    assert strict.config()["normalize"] is False
    assert isinstance(make_scorer("regex:ok"), RegexMatch)
    with pytest.raises(ValueError, match="unknown scorer spec"):
        make_scorer("embedding")
    with pytest.raises(ValueError, match="needs a pattern"):
        make_scorer("regex:")


def test_load_eval_jsonl_and_python_ref(project: Path) -> None:
    ev = load_eval("cases.jsonl", ["regex:yes"])
    assert len(ev) == 40
    assert ev.scorers[0].name == "regex_match"
    with pytest.raises(ValueError, match="not found"):
        load_eval("missing.jsonl")
    with pytest.raises(ValueError, match="not an Eval"):
        load_eval("cli_targets_mod:good")


def test_load_target_refs(project: Path) -> None:
    target = load_target("cli_targets_mod:good")
    assert target.name == "good"
    ollama = load_target("ollama:llama3.2", system="s", temperature=0.5)
    assert ollama.name == "ollama:llama3.2"
    with pytest.raises(ValueError, match="not a Target"):
        load_target("cli_targets_mod:not_a_target")
    with pytest.raises(ValueError, match="invalid Python reference"):
        load_target("nonsense")
    with pytest.raises(ValueError, match="cannot import"):
        load_target("definitely.not.a.module:x")
    with pytest.raises(ValueError, match="no attribute"):
        load_target("cli_targets_mod:nope")


# ---------------------------------------------------------------------------
# HTML rendering details
# ---------------------------------------------------------------------------


def _small_runs() -> tuple[Eval, StaticTarget, StaticTarget]:
    cases = [Case(input=f"q{i}", reference="yes", id=f"c{i}") for i in range(12)]
    ev = Eval("<b>esc & test</b>", cases, [ExactMatch()])
    good = StaticTarget({f"q{i}": "yes" for i in range(12)}, name="good")
    bad = StaticTarget({f"q{i}": ("no" if i < 4 else "yes") for i in range(12)}, name="b<ad")
    return ev, good, bad


def test_run_report_escapes_and_labels() -> None:
    ev, good, _ = _small_runs()
    html = render_run_report(run_eval(ev, target=good, seed=1))
    assert "&lt;b&gt;esc &amp; test&lt;/b&gt;" in html
    assert "<b>esc" not in html
    assert 'aria-label="1.000' in html
    assert "every metric carries its uncertainty" in html


def test_comparison_report_renders_bars_and_warnings() -> None:
    ev, good, bad = _small_runs()
    a = run_eval(ev, target=good, seed=1)
    b = run_eval(ev, target=bad, seed=1)
    html = render_comparison_report(compare(a, b, seed=0))
    assert html.count("<svg") == 2  # baseline + candidate error bars
    assert "b&lt;ad" in html
    assert "Δ=" in html
    # Insufficient-data row renders the note instead of bars.
    only_one = StaticTarget({"q0": "yes"}, name="one")
    c = run_eval(ev, target=only_one, seed=1)
    html2 = render_comparison_report(compare(a, c, seed=0))
    assert "INSUFFICIENT DATA" in html2
    assert "paired case(s)" in html2
