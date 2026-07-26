"""Leakage-fixed WF with tighter path-risk + conf/odds compounding stakes."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import config
from src.ha_backtest import (
    format_ha_backtest_summary,
    run_ha_walkforward_backtest,
    save_ha_backtest_reports,
)

STAMP = datetime.now().strftime("%Y%m%d")
AS_OF = datetime(2026, 7, 26)
BASELINE = Path("reports/ha_wf_tighter_path_20260726_summary.json")
BASELINE_TICKETS = Path("reports/ha_wf_tighter_path_20260726_tickets.csv")
PREFIX = "ha_wf_conf_odds_sizing"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("conf_odds_wf")

    import src.high_accuracy_strategy as ha

    paper_profile = getattr(config, "_PROFILE_PAPER", None)
    if not isinstance(paper_profile, dict):
        raise RuntimeError("config._PROFILE_PAPER missing")

    orig_paper = dict(ha._PAPER)
    orig_card_risk = float(paper_profile.get("max_card_risk_fraction") or 0.55)

    try:
        # Tighter path controls (same as overnight tighter-path run)
        ha._PAPER["max_parlay_share"] = 0.25
        ha._PAPER["stake_power"] = 0.75
        ha._PAPER["drawdown_soft_pct"] = 0.20
        ha._PAPER["drawdown_hard_pct"] = 0.35
        ha._PAPER["drawdown_soft_mult"] = 0.60
        ha._PAPER["drawdown_hard_mult"] = 0.35
        # Keep conf/odds Paper curve (already default); slightly flatter max under tighter path
        ha._PAPER["sizing_max_ticket_pct"] = min(
            float(ha._PAPER.get("sizing_max_ticket_pct") or 0.48), 0.42
        )
        ha._PAPER["sizing_curve_gamma"] = max(
            float(ha._PAPER.get("sizing_curve_gamma") or 1.20), 1.25
        )
        paper_profile["max_card_risk_fraction"] = min(orig_card_risk, 0.35)

        log.info(
            "Running conf/odds sizing WF stamp=%s tighter_path + sizing_max=%.2f gamma=%.2f",
            STAMP,
            ha._PAPER["sizing_max_ticket_pct"],
            ha._PAPER["sizing_curve_gamma"],
        )

        report = run_ha_walkforward_backtest(
            bankroll_start=100.0,
            last_year=True,
            use_dynamic_thresholds=True,
            profile="paper",
            as_of=AS_OF,
        )
        notes = list((report.get("summary") or {}).get("notes") or [])
        notes.extend(
            [
                "Conf/odds compounding: strength from model-prob, confidence, edge vs market, "
                "uncertainty penalty, parlay discount → absolute % of card (sum ≤ 100%).",
                "Tighter path-risk: max_parlay_share=25%, earlier DD cuts, max_card_risk≤35%.",
                "Fail-closed: missing odds / skip-level uncertainty → strength 0 (no inflate).",
            ]
        )
        if report.get("summary") is not None:
            report["summary"]["notes"] = notes
            report["summary"]["path_risk_mode"] = "tighter+conf_odds"
            report["summary"]["sizing_mode"] = "conf_odds"

        paths = save_ha_backtest_reports(
            report,
            stamp=STAMP,
            prefix=PREFIX,
            baseline_path=BASELINE if BASELINE.is_file() else None,
            baseline_label="vs prior tighter-path Paper WF",
            baseline_tickets_path=BASELINE_TICKETS if BASELINE_TICKETS.is_file() else None,
        )
        print(format_ha_backtest_summary(report))

        # Side-by-side comparison artifact
        base = {}
        if BASELINE.is_file():
            base = (json.loads(BASELINE.read_text(encoding="utf-8")).get("summary") or {})
        cur = report.get("summary") or {}

        def _f(s: dict, k: str, default=None):
            v = s.get(k)
            return default if v is None else v

        cmp = {
            "stamp": STAMP,
            "baseline": "ha_wf_tighter_path_20260726",
            "current": f"{PREFIX}_{STAMP}",
            "metrics": {
                "roi_on_stake_pct": {
                    "baseline": _f(base, "roi_on_stake_pct"),
                    "current": _f(cur, "roi_on_stake_pct"),
                },
                "max_drawdown_pct": {
                    "baseline": _f(base, "max_drawdown_pct"),
                    "current": _f(cur, "max_drawdown_pct"),
                },
                "hit_rate": {
                    "baseline": _f(base, "hit_rate"),
                    "current": _f(cur, "hit_rate"),
                },
                "bankroll_final": {
                    "baseline": _f(base, "bankroll_final"),
                    "current": _f(cur, "bankroll_final"),
                },
                "n_tickets": {
                    "baseline": _f(base, "n_tickets"),
                    "current": _f(cur, "n_tickets"),
                },
            },
            "paths": {k: str(v) for k, v in paths.items()},
        }
        cmp_path = Path("reports") / f"{PREFIX}_{STAMP}_vs_tighter.json"
        cmp_path.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        log.info("Comparison -> %s", cmp_path)
        print(
            "\nvs tighter-path baseline:\n"
            f"  ROI-stake {float(base.get('roi_on_stake_pct') or 0):.1f}% → "
            f"{float(cur.get('roi_on_stake_pct') or 0):.1f}%\n"
            f"  maxDD {float(base.get('max_drawdown_pct') or 0):.1f}% → "
            f"{float(cur.get('max_drawdown_pct') or 0):.1f}%\n"
            f"  hit {100 * float(base.get('hit_rate') or 0):.1f}% → "
            f"{100 * float(cur.get('hit_rate') or 0):.1f}%\n"
            f"  bankroll ${float(base.get('bankroll_final') or 0):.2f} → "
            f"${float(cur.get('bankroll_final') or 0):.2f}\n"
            f"HTML: {paths.get('html')}\n"
            f"CSV:  {paths.get('tickets_csv') or paths.get('csv')}"
        )
        return 0
    finally:
        ha._PAPER.clear()
        ha._PAPER.update(orig_paper)
        paper_profile["max_card_risk_fraction"] = orig_card_risk


if __name__ == "__main__":
    raise SystemExit(main())
