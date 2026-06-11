"""Rich terminal rendering for runs and comparisons — intervals always."""

from rich.console import Console
from rich.table import Table

from holdout.core.run import Run
from holdout.regression.compare import RunComparison
from holdout.store.run_store import StoredRunInfo

_VERDICT_STYLE = {
    "regressed": "bold red",
    "improved": "bold green",
    "no_significant_change": "dim",
    "insufficient_data": "bold yellow",
}
_VERDICT_TEXT = {
    "regressed": "REGRESSED",
    "improved": "IMPROVED",
    "no_significant_change": "no sig. Δ",
    "insufficient_data": "insufficient data",
}


def print_run(console: Console, run: Run, *, level: float = 0.95) -> None:
    """Print a run's metrics table, every estimate with its interval."""
    table = Table(
        title=f"{run.eval_name} · {run.target_name} · run {run.short_run_id}",
        title_justify="left",
    )
    table.add_column("metric")
    table.add_column("estimate", justify="right")
    table.add_column(f"{level * 100:g}% CI", justify="right")
    table.add_column("method")
    metrics = run.metrics(level=level)
    for name, est in metrics.items():
        table.add_row(
            name,
            f"{est.value:.3f}",
            f"[{est.ci_low:.3f}, {est.ci_high:.3f}]",
            f"{est.method} (n={est.n})",
        )
    for name in run.scorer_names:
        if name not in metrics:
            table.add_row(name, "—", "—", "no data (all cases errored)")
    console.print(table)
    if run.n_errors:
        console.print(
            f"[yellow]{run.n_errors} case(s) failed and are excluded from aggregates[/yellow]"
        )


def print_comparison(console: Console, cmp: RunComparison) -> None:
    """Print a comparison table plus verdict line and warnings."""
    table = Table(
        title=(
            f"{cmp.eval_name}: {cmp.baseline_target} ({cmp.baseline_run_id[:12]}) vs "
            f"{cmp.candidate_target} ({cmp.candidate_run_id[:12]})"
        ),
        title_justify="left",
    )
    table.add_column("metric")
    table.add_column("verdict")
    table.add_column("effect", justify="right")
    table.add_column("CI", justify="right")
    table.add_column("test")
    table.add_column("p (adj)", justify="right")
    table.add_column("n", justify="right")
    for c in cmp.comparisons:
        verdict = f"[{_VERDICT_STYLE[c.verdict]}]{_VERDICT_TEXT[c.verdict]}[/]"
        if c.result is None:
            table.add_row(c.metric, verdict, "—", "—", c.note or "—", "—", str(c.n_pairs))
            continue
        r = c.result
        p = c.p_adjusted if c.p_adjusted is not None else r.p_value
        table.add_row(
            c.metric,
            verdict,
            f"{r.effect:+.3f}",
            f"[{r.ci.ci_low:+.3f}, {r.ci.ci_high:+.3f}]",
            r.test,
            f"{p:.4g}",
            str(c.n_pairs),
        )
    console.print(table)
    style = _VERDICT_STYLE[cmp.verdict]
    console.print(
        f"verdict: [{style}]{_VERDICT_TEXT[cmp.verdict]}[/] (alpha={cmp.alpha:g}, {cmp.correction})"
    )
    for w in cmp.warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")


def print_run_list(console: Console, infos: list[StoredRunInfo]) -> None:
    """Print the stored-runs listing."""
    table = Table(title="stored runs", title_justify="left")
    table.add_column("run")
    table.add_column("eval")
    table.add_column("target")
    table.add_column("created")
    table.add_column("n", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("seed", justify="right")
    for info in infos:
        table.add_row(
            info.short_run_id,
            info.eval_name,
            info.target_name,
            info.created_at,
            str(info.n_cases),
            str(info.n_errors),
            "—" if info.seed is None else str(info.seed),
        )
    console.print(table)
