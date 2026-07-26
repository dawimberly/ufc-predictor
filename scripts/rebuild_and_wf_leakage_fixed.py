"""Rebuild features (leakage-fixed), retrain, then HA walk-forward.

Saves reports/ha_walkforward_leakage_fixed_YYYYMMDD.{html,csv,...}
Compares to pre-fix post-enrich run as reference only.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import config
from src.data_loader import ensure_data_dirs, load_fights
from src.feature_engineering import build_feature_matrix, save_features
from src.ha_backtest import (
    format_ha_backtest_summary,
    run_ha_walkforward_backtest,
    save_ha_backtest_reports,
)
from src.model_trainer import train_model

STAMP = datetime.now().strftime("%Y%m%d")
AS_OF = datetime(2026, 7, 26)  # same window as post-enrich / controlled runs
REF_SUMMARY = Path("reports/ha_walkforward_post_enrich_20260725_summary.json")
REF_TICKETS = Path("reports/ha_walkforward_post_enrich_20260725_tickets.csv")


def rebuild_and_train() -> None:
    ensure_data_dirs()
    print("Loading fights...", flush=True)
    fights = load_fights()
    print(f"fights={len(fights)}", flush=True)
    print(
        f"Building features (schema {config.FEATURE_SCHEMA_VERSION}, leakage-fixed)…",
        flush=True,
    )
    features = build_feature_matrix(fights)
    path = save_features(features)
    print(f"Saved {len(features)} rows -> {path}", flush=True)
    print("Training…", flush=True)
    result = train_model(
        features,
        tune="none",
        calibration_method=config.CALIBRATION_METHOD,
        run_backtest_hook=False,
    )
    print(
        f"AUC={result.metrics.get('roc_auc')} acc={result.metrics.get('accuracy')}",
        flush=True,
    )


def run_walkforward() -> int:
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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rebuild_and_train()
    return run_walkforward()


if __name__ == "__main__":
    sys.exit(main())
