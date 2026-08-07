"""Productionize HV features: retrain + backtest-2025 report.

Uses cached feature matrix when present (FORCE_REBUILD_FEATURES=1 to rebuild).
Keeps ENABLE_HIGH_VALUE_FEATURES=true (AUC primary keep rule).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("ENABLE_HIGH_VALUE_FEATURES", "true")
os.environ.setdefault("FEATURE_SCHEMA_VERSION", "5")

import numpy as np
import pandas as pd

import config

config.refresh_runtime_env()

from src.data_loader import ensure_data_dirs, load_fights
from src.feature_engineering import build_feature_matrix, save_features
from src.high_value_features import HIGH_VALUE_DIFF_COLUMNS, log_hv_coverage
from src.model_freshness import features_fingerprint
from src.model_trainer import load_trained_model, train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prod_hv")


def _load_or_build_features() -> pd.DataFrame:
    cache = config.PROCESSED_DIR / "ab_feature_matrix_v5.parquet"
    force = os.getenv("FORCE_REBUILD_FEATURES", "").lower() in ("1", "true", "yes")
    if cache.is_file() and not force:
        logger.info("Loading cached features %s", cache)
        features = pd.read_parquet(cache)
    else:
        logger.info("Building feature matrix from fights…")
        features = build_feature_matrix(load_fights(), keep_unlabeled=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache)
    missing = [c for c in HIGH_VALUE_DIFF_COLUMNS if c not in features.columns]
    if missing:
        raise SystemExit(f"HV columns missing from feature matrix: {missing}")
    log_hv_coverage(features, year=2025, label="productionize")
    path = save_features(features)
    logger.info("Saved production features → %s (%s rows)", path, len(features))
    return features


def _flat_edge_roi(
    y: np.ndarray,
    proba: np.ndarray,
    f1_odds: np.ndarray | None,
    f2_odds: np.ndarray | None,
    *,
    edge_min: float = 0.03,
) -> dict[str, float]:
    if f1_odds is None or f2_odds is None:
        return {"flat_edge_roi": float("nan"), "n_bets": 0.0}
    pnl = 0.0
    n_bets = 0
    for yt, p, o1, o2 in zip(y, proba, f1_odds, f2_odds):
        try:
            o1f, o2f = float(o1), float(o2)
        except (TypeError, ValueError):
            continue
        if not (o1f > 1.0 and o2f > 1.0):
            continue
        imp1 = (1.0 / o1f) / ((1.0 / o1f) + (1.0 / o2f))
        edge1 = float(p) - imp1
        edge2 = (1.0 - float(p)) - (1.0 - imp1)
        if edge1 >= edge_min and edge1 >= edge2:
            n_bets += 1
            pnl += (o1f - 1.0) if int(yt) == 1 else -1.0
        elif edge2 >= edge_min:
            n_bets += 1
            pnl += (o2f - 1.0) if int(yt) == 0 else -1.0
    return {
        "flat_edge_roi": (pnl / n_bets) if n_bets else float("nan"),
        "n_bets": float(n_bets),
    }


def _holdout_odds_coverage(features: pd.DataFrame, year: int = 2025) -> dict:
    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    test = features.loc[dts.dt.year == year]
    has = False
    n_with = 0
    if {"f1_odds", "f2_odds"}.issubset(test.columns):
        mask = test["f1_odds"].notna() & test["f2_odds"].notna()
        n_with = int(mask.sum())
        has = n_with > 0
    return {"year": year, "n": int(len(test)), "n_with_odds": n_with, "has_odds": has}


def main() -> int:
    ensure_data_dirs()
    assert config.ENABLE_HIGH_VALUE_FEATURES, "ENABLE_HIGH_VALUE_FEATURES must be true"
    logger.info(
        "Config: HV=%s schema=%s FEATURE_COLUMNS=%s (HV subset=%s)",
        config.ENABLE_HIGH_VALUE_FEATURES,
        config.FEATURE_SCHEMA_VERSION,
        len(config.FEATURE_COLUMNS),
        sum(1 for c in config.HIGH_VALUE_FEATURE_COLUMNS if c in config.FEATURE_COLUMNS),
    )

    features = _load_or_build_features()
    fp = features_fingerprint(config.PROCESSED_FEATURES_CSV)
    logger.info("features_fingerprint=%s", fp)

    logger.info("Training production model (HV on)…")
    result = train_model(
        features,
        model_path=config.DEFAULT_MODEL_PATH,
        tune="none",
        calibration_method=config.CALIBRATION_METHOD,
        run_backtest_hook=False,
    )
    artifact = load_trained_model(result.model_path)
    logger.info(
        "Saved artifact %s | n_features=%s fingerprint=%s HV_cols=%s",
        result.model_path,
        artifact.get("n_features") or len(artifact.get("feature_columns") or []),
        artifact.get("features_fingerprint"),
        artifact.get("high_value_feature_columns"),
    )
    logger.info(
        "Train holdout metrics: acc=%s auc=%s",
        result.metrics.get("accuracy"),
        result.metrics.get("roc_auc"),
    )

    # --- backtest-2025 via project entry ---
    from src.backtester import backtest_2025, print_backtest_2025_summary

    logger.info("Running backtest-2025…")
    bt = backtest_2025(save_outputs=True, target_year=2025)
    print_backtest_2025_summary(bt)
    overall = dict(getattr(bt, "overall_metrics", None) or {})
    bank = dict(getattr(bt, "bankroll_summary", None) or {})
    staking = getattr(bt, "staking_modes", None)
    flat_roi = None
    if isinstance(staking, pd.DataFrame) and not staking.empty:
        # Prefer a flat / flat-edge style row if present
        for key in ("mode", "staking", "name"):
            if key in staking.columns:
                flat_rows = staking[
                    staking[key].astype(str).str.lower().str.contains("flat")
                ]
                if not flat_rows.empty:
                    row = flat_rows.iloc[0]
                    flat_roi = {
                        k: row.get(k)
                        for k in ("roi_pct", "roi", "n_bets", "max_drawdown", "max_dd")
                        if k in row.index
                    }
                    break

    odds_cov = _holdout_odds_coverage(features, 2025)
    logger.info("2025 odds coverage: %s", odds_cov)

    # Optional flat-edge ROI on 2025 holdout (odds present) via FightPredictor
    # so interaction specs match the production artifact (no column mismatch).
    roi_ab = {"skipped": True, "reason": "no_odds_on_2025_holdout"}
    if odds_cov["has_odds"]:
        from sklearn.metrics import accuracy_score, roc_auc_score
        from src.predictor import FightPredictor

        pred = FightPredictor(result.model_path)
        dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
        test = features.loc[dts.dt.year == 2025].copy()
        prepared = pred._prepare_features(test)
        y = prepared[config.TARGET_COLUMN].astype(int).to_numpy()
        proba = pred.model.predict_proba(prepared[pred.feature_columns])[:, 1]
        f1o = prepared["f1_odds"].to_numpy() if "f1_odds" in prepared.columns else None
        f2o = prepared["f2_odds"].to_numpy() if "f2_odds" in prepared.columns else None
        roi_on = _flat_edge_roi(y, proba, f1o, f2o)
        roi_ab = {
            "skipped": False,
            "with_hv_production": {
                "accuracy": float(accuracy_score(y, (proba >= 0.5).astype(int))),
                "auc": float(roc_auc_score(y, proba)),
                **roi_on,
            },
            "note": (
                "HV kept (AUC primary). Off-arm ROI not re-run here; "
                "prior A/B already kept HV. Flat-edge uses production artifact + interactions."
            ),
        }
        logger.info("Flat-edge ROI (HV production, 2025 odds): %s", roi_ab)
    report = {
        "enable_high_value_features": True,
        "feature_schema_version": config.FEATURE_SCHEMA_VERSION,
        "n_features": len(artifact.get("feature_columns") or []),
        "features_fingerprint": artifact.get("features_fingerprint"),
        "high_value_feature_columns": artifact.get("high_value_feature_columns"),
        "train_metrics": {
            "accuracy": result.metrics.get("accuracy"),
            "roc_auc": result.metrics.get("roc_auc"),
        },
        "backtest_2025": {
            "overall": overall,
            "bankroll_summary": bank,
            "fights_with_odds": getattr(bt, "fights_with_odds", None),
            "flat_staking": flat_roi,
        },
        "odds_coverage_2025": odds_cov,
        "flat_edge_roi_check": roi_ab,
        "keep_rule": "AUC primary; HV kept despite small accuracy dip",
    }
    out = config.DATA_DIR / "reports" / "hv_productionize_2025.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info(
        "SUMMARY n_feat=%s fp=%s train_auc=%s bt_acc=%s bt_auc=%s odds=%s",
        report["n_features"],
        report["features_fingerprint"],
        report["train_metrics"]["roc_auc"],
        overall.get("accuracy"),
        overall.get("roc_auc") or overall.get("auc"),
        odds_cov,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
