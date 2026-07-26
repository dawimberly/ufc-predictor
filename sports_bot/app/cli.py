"""CLI entrypoint: python -m sports_bot.app.cli"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_demo(_: argparse.Namespace) -> int:
    from sports_bot.app.pipeline import run_card
    from sports_bot.bank.ledger import accuracy_stats
    from sports_bot.core import config

    config.ensure_dirs()
    picks = run_card(send_alerts=False)
    table = Table(title="Demo card picks")
    for col in ("Event", "Pick", "Prob", "Odds", "Stake", "Conf"):
        table.add_column(col)
    for p in picks:
        table.add_row(
            str(p["event"]),
            str(p["selection"]),
            f"{p['prob']:.0%}",
            f"{p['odds']:.2f}" if p.get("odds") else "-",
            f"${p['stake']:.2f}",
            str(p["confidence"]),
        )
    console.print(table)
    for p in picks:
        console.print(f"[dim]{p['reasons']}[/dim]")
    console.print(f"Bank stats: {accuracy_stats()}")
    return 0


def cmd_bank_stats(_: argparse.Namespace) -> int:
    from sports_bot.bank.ledger import accuracy_stats
    from sports_bot.core import config

    config.ensure_dirs()
    console.print_json(json.dumps(accuracy_stats()))
    return 0


def cmd_learn(_: argparse.Namespace) -> int:
    from sports_bot.bank.learning import run_thinking_review
    from sports_bot.core import config

    config.ensure_dirs()
    result = run_thinking_review()
    console.print_json(json.dumps(result, default=str))
    return 0 if result.get("ok") else 1


def cmd_telegram_test(_: argparse.Namespace) -> int:
    from sports_bot.alerts.telegram import send_telegram, telegram_configured

    if not telegram_configured():
        console.print("[yellow]Telegram not configured (TELEGRAM_ENABLED + token + chat id).[/yellow]")
        return 1
    body = send_telegram("sports-bot skeleton online ✅")
    console.print(body)
    return 0 if body.get("ok") else 1


def cmd_backtest(args: argparse.Namespace) -> int:
    from sports_bot.app.backtest_recent import run_recent_card_backtest, save_report

    last = int(getattr(args, "last", 5) or 5)
    console.print(f"[cyan]Running recent-card backtest (last={last})…[/cyan]")
    try:
        report = run_recent_card_backtest(last=last)
        path = save_report(report)
    except Exception as exc:
        console.print(f"[red]Backtest failed:[/red] {exc}")
        return 1

    bet = report.get("betting") or {}
    clf = report.get("classification") or {}
    table = Table(title=f"Last {last} cards")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Fights", str(report.get("n_fights")))
    table.add_row("Pick accuracy", f"{report.get('pick_accuracy', 0):.1%}")
    table.add_row("Model accuracy (f1)", f"{float(clf.get('accuracy') or 0):.1%}")
    table.add_row("Log-loss", f"{clf.get('log_loss')}")
    table.add_row("Brier", f"{clf.get('brier_score')}")
    table.add_row("Betting ROI", f"{float(bet.get('roi_pct') or (float(bet.get('roi') or 0) * 100)):.1f}%")
    table.add_row("Betting PnL", f"{float(bet.get('pnl') or 0):+.2f}")
    table.add_row("Bets", str(bet.get("n_bets")))
    console.print(table)

    ev = Table(title="Per event")
    ev.add_column("Event")
    ev.add_column("Fights", justify="right")
    ev.add_column("Acc", justify="right")
    for row in report.get("per_event") or []:
        ev.add_row(str(row["event"])[:48], str(row["fights"]), f"{row['accuracy']:.0%}")
    console.print(ev)
    console.print(f"[green]Report:[/green] {path}")
    return 0


def cmd_view_log(args: argparse.Namespace) -> int:
    """Show UFC-Predictor predictions.log (or create stub if missing)."""
    from pathlib import Path

    from sports_bot.app.backtest_recent import resolve_ufc_root

    root = resolve_ufc_root()
    log_path = root / "data" / "logs" / "predictions.log"
    lines = int(getattr(args, "lines", 80) or 80)
    if not log_path.is_file():
        console.print(f"[yellow]No predictions.log yet at[/yellow] {log_path}")
        console.print("It is created when the Prediction Bank logs card picks.")
        bank = root / "data" / "prediction_bank.csv"
        if bank.is_file():
            console.print(f"[cyan]Showing prediction_bank.csv instead:[/cyan] {bank}")
            text = bank.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in text[:lines]:
                console.print(line)
            return 0
        return 1
    text = log_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    console.print(f"[cyan]{log_path}[/cyan] ({len(text)} lines, showing last {lines})")
    for line in text[-lines:]:
        console.print(line.replace("\ufeff", ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sports-bot", description="UFC + sports betting bot skeleton")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Run demo UFC card through CompuBox+Markov+Kelly+Bank")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("bank-stats", help="Show prediction bank accuracy")
    s.set_defaults(func=cmd_bank_stats)

    l = sub.add_parser("learn", help="Post-fight thinking-model review")
    l.set_defaults(func=cmd_learn)

    t = sub.add_parser("telegram-test", help="Send a test Telegram message")
    t.set_defaults(func=cmd_telegram_test)

    b = sub.add_parser("backtest", help="Backtest last N UFC cards with current model")
    b.add_argument("--last", type=int, default=5, help="Number of most recent cards (default 5)")
    b.set_defaults(func=cmd_backtest)

    v = sub.add_parser("view-log", help="View UFC-Predictor data/logs/predictions.log")
    v.add_argument("--lines", type=int, default=80, help="Lines to show (default 80)")
    v.set_defaults(func=cmd_view_log)
    return p


def main(argv: list[str] | None = None) -> int:
    # Allow `python -m sports_bot.app.cli` without install by fixing path.
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
