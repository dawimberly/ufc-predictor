"""Run post-enrich HA walk-forward and compare to path-risk controlled baseline."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.ha_backtest import (
    format_ha_backtest_summary,
    run_ha_walkforward_backtest,
    save_ha_backtest_reports,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Match prior controlled WF window: last-12m as of report day (features end ~2026-06-06).
    as_of = datetime(2026, 7, 26)
    baseline = Path("reports/ha_walkforward_drawdown_controls_20260725_summary.json")
    print(f"Running true walk-forward HA backtest as_of={as_of.date()} bankroll=$100 …")
    report = run_ha_walkforward_backtest(
        bankroll_start=100.0,
        last_year=True,
        use_dynamic_thresholds=True,
        profile="paper",
        as_of=as_of,
    )
    save_ha_backtest_reports(
        report,
        stamp="20260725",
        prefix="ha_walkforward_post_enrich",
        baseline_path=baseline if baseline.is_file() else None,
        baseline_label="vs prior controlled walk-forward (path-risk HA)",
        baseline_tickets_path=Path(
            "reports/ha_walkforward_drawdown_controls_20260725_tickets.csv"
        ),
    )
    print(format_ha_backtest_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
