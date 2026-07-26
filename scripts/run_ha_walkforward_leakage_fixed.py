"""Run leakage-fixed HA walk-forward only (features/model already rebuilt)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.ha_backtest import (
    format_ha_backtest_summary,
    run_ha_walkforward_backtest,
    save_ha_backtest_reports,
)

STAMP = datetime.now().strftime("%Y%m%d")
AS_OF = datetime(2026, 7, 26)
REF_SUMMARY = Path("reports/ha_walkforward_post_enrich_20260725_summary.json")
REF_TICKETS = Path("reports/ha_walkforward_post_enrich_20260725_tickets.csv")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(
        f"Running true walk-forward HA backtest as_of={AS_OF.date()} bankroll=$100 …",
        flush=True,
    )
    report = run_ha_walkforward_backtest(
        bankroll_start=100.0,
        last_year=True,
        use_dynamic_thresholds=True,
        profile="paper",
        as_of=AS_OF,
    )
    notes = list((report.get("summary") or {}).get("notes") or [])
    notes.append(
        "Leakage-fixed features: no Sherdog career W-L fallback, no Greco career "
        "static fills, same-card Elo frozen at card open."
    )
    if report.get("summary") is not None:
        report["summary"]["notes"] = notes

    save_ha_backtest_reports(
        report,
        stamp=STAMP,
        prefix="ha_walkforward_leakage_fixed",
        baseline_path=REF_SUMMARY if REF_SUMMARY.is_file() else None,
        baseline_label="vs pre-fix post-enrich WF (reference only — trust leakage-fixed)",
        baseline_tickets_path=REF_TICKETS if REF_TICKETS.is_file() else None,
    )
    print(format_ha_backtest_summary(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
