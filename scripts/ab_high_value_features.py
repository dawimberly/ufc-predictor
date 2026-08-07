"""Phase 2 A/B: 2025 holdout with vs without high-value feature block.

Builds features once (HV cols always present), trains two models on pre-2025,
evaluates accuracy / AUC / flat-edge ROI on 2025. Keeps HV enabled only if
AUC or ROI improves.

Does not touch HA color logic or dashboard tabs.
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

import numpy as np
import pandas as pd

import config
from src.data_loader import load_fights
from src.feature_engineering import build_feature_matrix
from src.high_value_features import HIGH_VALUE_DIFF_COLUMNS, log_hv_coverage
from src.model_trainer import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ab_hv")


def _metrics_from_probs(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    f1_odds: np.ndarray | None = None,
    f2_odds: np.ndarray | None = None,
    edge_min: float = 0.03,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score

    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, pred))
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = float("nan")

    # Flat $1 on side with model edge >= edge_min vs implied odds.
    roi = float("nan")
    n_bets = 0
    pnl = 0.0
    if f1_odds is not None and f2_odds is not None:
        for yt, p, o1, o2 in zip(y_true, proba, f1_odds, f2_odds):
            try:
                o1f, o2f = float(o1), float(o2)
            except (TypeError, ValueError):
                continue
            if not (o1f > 1.0 and o2f > 1.0):
                continue
            imp1 = (1.0 / o1f) / ((1.0 / o1f) + (1.0 / o2f))
            imp2 = 1.0 - imp1
            edge1 = float(p) - imp1
            edge2 = (1.0 - float(p)) - imp2
            if edge1 >= edge_min and edge1 >= edge2:
                n_bets += 1
                pnl += (o1f - 1.0) if int(yt) == 1 else -1.0
            elif edge2 >= edge_min:
                n_bets += 1
                pnl += (o2f - 1.0) if int(yt) == 0 else -1.0
        if n_bets > 0:
            roi = pnl / n_bets
    return {
        "accuracy": acc,
        "auc": auc,
        "flat_edge_roi": roi,
        "n_bets": float(n_bets),
        "n": float(len(y_true)),
    }


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    # EnsembleArtifact-style
    if hasattr(model, "predict"):
        out = model.predict(X)
        if isinstance(out, tuple):
            return np.asarray(out[0], dtype=float)
        return np.asarray(out, dtype=float)
    raise TypeError(type(model))


def run_arm(features: pd.DataFrame, *, use_hv: bool, year: int = 2025) -> dict:
    os.environ["ENABLE_HIGH_VALUE_FEATURES"] = "true" if use_hv else "false"
    os.environ["INTERACTION_DISCOVERY_ENABLED"] = "false"
    config.refresh_runtime_env()
    hv = set(config.HIGH_VALUE_FEATURE_COLUMNS)
    cols = [c for c in config.FEATURE_COLUMNS if c in features.columns]
    if use_hv:
        assert any(c in hv for c in cols), "HV arm missing HV columns"
    else:
        cols = [c for c in cols if c not in hv]
        assert not any(c in hv for c in cols)

    date_col = config.DATE_COLUMN
    dts = pd.to_datetime(features[date_col], errors="coerce")
    train = features.loc[dts.dt.year < year].copy()
    test = features.loc[dts.dt.year == year].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Need pre-{year} train and {year} test rows")

    saved = list(config.FEATURE_COLUMNS)
    config.FEATURE_COLUMNS = cols
    arm_dir = config.MODELS_DIR / "ab_hv"
    arm_dir.mkdir(parents=True, exist_ok=True)
    model_path = arm_dir / ("with_hv.joblib" if use_hv else "without_hv.joblib")
    try:
        result = train_model(
            train,
            model_path=model_path,
            tune="none",
            run_backtest_hook=False,
        )
    finally:
        config.FEATURE_COLUMNS = saved

    import joblib

    artifact = joblib.load(result.model_path)
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected model artifact dict, got {type(artifact)}")
    feat_cols = list(artifact.get("feature_columns") or result.feature_columns)
    feat_cols = [c for c in feat_cols if c in test.columns]

    from src.feature_engineering import apply_imputer

    test_x = test.copy()
    imputer = artifact.get("imputer")
    if imputer is not None:
        test_x = apply_imputer(test_x, imputer)

    X = test_x[feat_cols]
    y = test[config.TARGET_COLUMN].astype(int).to_numpy()
    clf = artifact["model"]
    proba = np.asarray(clf.predict_proba(X)[:, 1], dtype=float)

    f1o = test["f1_odds"].to_numpy() if "f1_odds" in test.columns else None
    f2o = test["f2_odds"].to_numpy() if "f2_odds" in test.columns else None
    metrics = _metrics_from_probs(y, proba, f1_odds=f1o, f2_odds=f2o)
    metrics["arm"] = "with_hv" if use_hv else "without_hv"
    metrics["n_features"] = float(len(feat_cols))
    return metrics


def main() -> int:
    cache_path = config.PROCESSED_DIR / "ab_feature_matrix_v5.parquet"
    force = os.getenv("FORCE_REBUILD_FEATURES", "").lower() in ("1", "true", "yes")
    if cache_path.is_file() and not force:
        logger.info("Loading cached feature matrix %s", cache_path)
        features = pd.read_parquet(cache_path)
    else:
        logger.info("Loading fights + building feature matrix (HV cols always computed)…")
        fights = load_fights()
        features = build_feature_matrix(fights, keep_unlabeled=False, use_fighter_cache=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path)
        logger.info("Cached feature matrix → %s", cache_path)

    log_hv_coverage(features, year=2025, label="phase2 matrix")

    present = [c for c in HIGH_VALUE_DIFF_COLUMNS if c in features.columns]
    missing = [c for c in HIGH_VALUE_DIFF_COLUMNS if c not in features.columns]
    logger.info("HV cols present=%s missing=%s", len(present), missing)

    without = run_arm(features, use_hv=False)
    logger.info("WITHOUT HV: %s", without)
    with_hv = run_arm(features, use_hv=True)
    logger.info("WITH HV:    %s", with_hv)

    d_acc = with_hv["accuracy"] - without["accuracy"]
    d_auc = with_hv["auc"] - without["auc"]
    d_roi = with_hv["flat_edge_roi"] - without["flat_edge_roi"]
    keep = (d_auc > 0) or (
        not np.isnan(d_roi) and d_roi > 0
    )
    # Prefer keep when either improves; if both NaN/flat, keep for coverage
    if np.isnan(d_auc) and (np.isnan(d_roi) or without["n_bets"] == 0):
        keep = True
        reason = "inconclusive_odds_coverage_keep"
    elif keep:
        reason = "auc_or_roi_improved"
    else:
        reason = "no_improvement"

    report = {
        "without_hv": without,
        "with_hv": with_hv,
        "delta": {"accuracy": d_acc, "auc": d_auc, "flat_edge_roi": d_roi},
        "keep_high_value_features": bool(keep),
        "reason": reason,
    }
    out_path = config.DATA_DIR / "reports" / "hv_features_ab_2025.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info(
        "DECISION keep=%s (%s)  d_acc=%+.4f d_auc=%+.4f d_roi=%+.4f",
        keep,
        reason,
        d_acc,
        d_auc,
        d_roi if not np.isnan(d_roi) else float("nan"),
    )

    # Persist flag recommendation into a small sidecar (does not rewrite .env)
    flag_path = config.DATA_DIR / "reports" / "hv_features_flag.txt"
    flag_path.write_text(
        "ENABLE_HIGH_VALUE_FEATURES=" + ("true" if keep else "false") + f"  # {reason}\n",
        encoding="utf-8",
    )
    if not keep:
        # Flip default for this process; user can set .env from flag file
        os.environ["ENABLE_HIGH_VALUE_FEATURES"] = "false"
        config.refresh_runtime_env()
        logger.warning(
            "HV block did not improve AUC/ROI — set ENABLE_HIGH_VALUE_FEATURES=false"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
