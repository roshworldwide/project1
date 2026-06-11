"""The holdout CLI: run, compare, list, report, power, check.

Exit codes are CI-grade contracts:

- ``compare``: 0 = no significant regression, 1 = regression detected,
  2 = insufficient data (refusing to certify).
- ``check``: 0 = clean, 1 = leakage or duplicates found.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from holdout.exceptions import HoldoutError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holdout",
        description=(
            "Quant-grade LLM evaluation: confidence intervals, significance "
            "tests, and regression gates — not vanity numbers."
        ),
    )
    parser.add_argument("--store", default=".holdout", help="run store directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an eval against a target and store the result")
    p_run.add_argument("eval", help="a .jsonl file or Python reference 'module:attr'")
    p_run.add_argument(
        "--target",
        required=True,
        help="'ollama:MODEL' | 'openai:MODEL' | 'anthropic:MODEL' | 'module:attr'",
    )
    p_run.add_argument(
        "--scorer",
        action="append",
        default=None,
        help="scorer for .jsonl evals: exact | exact-strict | regex:<pattern> (repeatable)",
    )
    p_run.add_argument("--system", default=None, help="system prompt for provider targets")
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--max-tokens", type=int, default=1024)
    p_run.add_argument("--base-url", default=None, help="override the provider endpoint")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--max-concurrency", type=int, default=8)
    p_run.add_argument(
        "--id-file", default=None, help="write the full run id to this file (for CI scripting)"
    )

    p_cmp = sub.add_parser("compare", help="compare two stored runs and issue a verdict")
    p_cmp.add_argument("baseline", help="run id or unambiguous prefix")
    p_cmp.add_argument("candidate", help="run id or unambiguous prefix")
    p_cmp.add_argument("--alpha", type=float, default=0.05)
    p_cmp.add_argument(
        "--correction",
        choices=["benjamini-hochberg", "holm", "none"],
        default="benjamini-hochberg",
    )
    p_cmp.add_argument(
        "--test",
        choices=["auto", "paired-bootstrap", "mcnemar", "permutation"],
        default="auto",
    )
    p_cmp.add_argument("--seed", type=int, default=0)
    p_cmp.add_argument("--budget", type=int, default=20, help="holdout-discipline use budget")
    p_cmp.add_argument(
        "--no-ledger", action="store_true", help="do not record this look in the ledger"
    )
    p_cmp.add_argument(
        "--markdown",
        action="store_true",
        help="emit GitHub-flavored markdown instead of a terminal table (for PR comments)",
    )

    p_list = sub.add_parser("list", help="list stored runs")
    p_list.add_argument("--eval", dest="eval_name", default=None)
    p_list.add_argument("--limit", type=int, default=20)

    p_rep = sub.add_parser("report", help="write a self-contained HTML report")
    p_rep.add_argument("runs", nargs="+", help="one run (run view) or two (comparison view)")
    p_rep.add_argument("-o", "--output", default="holdout-report.html")
    p_rep.add_argument("--alpha", type=float, default=0.05)
    p_rep.add_argument("--seed", type=int, default=0)

    p_pow = sub.add_parser("power", help="sample-size / minimum-detectable-effect calculator")
    p_pow.add_argument("--n", type=int, default=None, help="paired cases you have")
    p_pow.add_argument("--mde", type=float, default=None, help="effect size you must detect")
    p_pow.add_argument("--sd", type=float, default=None, help="SD of per-pair differences")
    p_pow.add_argument("--p01", type=float, default=None, help="expected fix rate (binary)")
    p_pow.add_argument("--p10", type=float, default=None, help="expected break rate (binary)")
    p_pow.add_argument("--alpha", type=float, default=0.05)
    p_pow.add_argument("--power", type=float, default=0.80)

    p_chk = sub.add_parser("check", help="leakage check: contamination + near-duplicates")
    p_chk.add_argument("eval", help="a .jsonl file or Python reference 'module:attr'")
    p_chk.add_argument("--scorer", action="append", default=None, help="scorer(s) for .jsonl evals")
    p_chk.add_argument(
        "--corpus", action="append", default=None, help="text file with prompt content (repeatable)"
    )
    p_chk.add_argument("--corpus-text", action="append", default=None, help="literal prompt text")
    p_chk.add_argument("--ngram", type=int, default=5)
    p_chk.add_argument("--threshold", type=float, default=0.5)
    p_chk.add_argument("--dup-threshold", type=float, default=0.8)
    return parser


def _cmd_run(args: argparse.Namespace, console: Console) -> int:
    from holdout.cli.refs import load_eval, load_target
    from holdout.cli.render import print_run
    from holdout.core.runner import run as run_eval
    from holdout.store.run_store import RunStore

    ev = load_eval(args.eval, args.scorer)
    target = load_target(
        args.target,
        system=args.system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )
    console.print(f"running {ev!r} against [bold]{target.name}[/bold] (seed={args.seed})")
    result = run_eval(ev, target=target, seed=args.seed, max_concurrency=args.max_concurrency)
    path = RunStore(args.store).save(result)
    print_run(console, result)
    console.print(f"saved [bold]{result.short_run_id}[/bold] -> {path}")
    if args.id_file is not None:
        Path(args.id_file).write_text(result.run_id, encoding="utf-8")
    return 0


def _cmd_compare(args: argparse.Namespace, console: Console) -> int:
    from holdout.cli.render import comparison_markdown, print_comparison
    from holdout.leakage.ledger import HoldoutLedger
    from holdout.regression.compare import compare
    from holdout.store.run_store import RunStore

    store = RunStore(args.store)
    baseline = store.load(args.baseline)
    candidate = store.load(args.candidate)
    cmp = compare(
        baseline,
        candidate,
        alpha=args.alpha,
        correction=args.correction,
        test=args.test,
        seed=args.seed,
    )
    if args.markdown:
        print(comparison_markdown(cmp))
    else:
        print_comparison(console, cmp)

    if not args.no_ledger:
        ledger = HoldoutLedger(args.store)
        ledger.record_use(
            baseline.eval_fingerprint,
            baseline.eval_name,
            kind="compare",
            context=f"{baseline.short_run_id} vs {candidate.short_run_id}",
        )
        report = ledger.check(baseline.eval_fingerprint, baseline.eval_name, budget=args.budget)
        if args.markdown:
            print(f"\n> {report}")
        else:
            style = {"ok": "dim", "caution": "yellow", "overfit-risk": "bold red"}[report.level]
            console.print(f"[{style}]{report}[/]")

    if cmp.verdict == "regressed":
        return 1
    if cmp.verdict == "insufficient_data":
        return 2
    return 0


def _cmd_list(args: argparse.Namespace, console: Console) -> int:
    from holdout.cli.render import print_run_list
    from holdout.store.run_store import RunStore

    infos = RunStore(args.store).runs(eval_name=args.eval_name, limit=args.limit)
    print_run_list(console, infos)
    return 0


def _cmd_report(args: argparse.Namespace, console: Console) -> int:
    from holdout.regression.compare import compare
    from holdout.report.html import render_comparison_report, render_run_report
    from holdout.store.run_store import RunStore

    if len(args.runs) not in (1, 2):
        raise HoldoutError("report takes one run (run view) or two runs (comparison view)")
    store = RunStore(args.store)
    if len(args.runs) == 1:
        html = render_run_report(store.load(args.runs[0]))
    else:
        baseline = store.load(args.runs[0])
        candidate = store.load(args.runs[1])
        html = render_comparison_report(
            compare(baseline, candidate, alpha=args.alpha, seed=args.seed)
        )
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    console.print(f"wrote {out}")
    return 0


def _cmd_power(args: argparse.Namespace, console: Console) -> int:
    from holdout.stats.power import (
        minimum_detectable_effect,
        paired_binary_sd,
        required_sample_size,
    )

    if args.sd is not None:
        sd = args.sd
    elif args.p01 is not None and args.p10 is not None:
        sd = paired_binary_sd(args.p01, args.p10)
        console.print(f"sd_diff = {sd:.4f} (from p01={args.p01:g}, p10={args.p10:g})")
    else:
        raise HoldoutError("provide --sd, or both --p01 and --p10 for binary metrics")

    if (args.n is None) == (args.mde is None):
        raise HoldoutError("provide exactly one of --n (to get the MDE) or --mde (to get n)")
    if args.n is not None:
        analysis = minimum_detectable_effect(args.n, sd, alpha=args.alpha, power=args.power)
    else:
        analysis = required_sample_size(args.mde, sd, alpha=args.alpha, power=args.power)
    console.print(str(analysis))
    return 0


def _cmd_check(args: argparse.Namespace, console: Console) -> int:
    from holdout.cli.refs import load_eval
    from holdout.leakage.contamination import check_contamination
    from holdout.leakage.duplicates import find_near_duplicates

    ev = load_eval(args.eval, args.scorer)
    corpus: list[str] = []
    for file_ref in args.corpus or []:
        corpus.append(Path(file_ref).read_text(encoding="utf-8"))
    corpus.extend(args.corpus_text or [])

    dirty = False
    if corpus:
        report = check_contamination(ev, corpus, ngram_size=args.ngram, threshold=args.threshold)
        console.print(report.summary())
        dirty = dirty or not report.clean
    else:
        console.print("[dim]no corpus given; skipping contamination check[/dim]")

    dupes = find_near_duplicates(ev, threshold=args.dup_threshold)
    if dupes:
        console.print(f"near-duplicate cases (effective n is below {len(ev.cases)}):")
        for d in dupes:
            console.print(f"  {d}")
        dirty = True
    else:
        console.print("no near-duplicate cases")
    return 1 if dirty else 0


_COMMANDS = {
    "run": _cmd_run,
    "compare": _cmd_compare,
    "list": _cmd_list,
    "report": _cmd_report,
    "power": _cmd_power,
    "check": _cmd_check,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch, and return the exit code."""
    args = _build_parser().parse_args(argv)
    console = Console()
    try:
        return _COMMANDS[args.command](args, console)
    except (HoldoutError, ValueError, KeyError, OSError) as exc:
        message = str(exc) if not isinstance(exc, KeyError) else str(exc.args[0])
        Console(stderr=True).print(f"[bold red]error:[/bold red] {message}")
        return 2


def app() -> None:
    """Console-script entry point."""
    sys.exit(main())
