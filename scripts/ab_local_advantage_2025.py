"""2025 A/B: local_advantage_diff (home / gym-vs-venue) vs HV baseline.

Research only — does NOT retrain ensemble_winner.joblib, does NOT enable
pathway/market flags, does NOT change Live HA thresholds.

Keep rule (same spirit as pathway A/B): AUC >= +0.005 vs BASE, or flat-edge
ROI/maxDD clearly better with >=20 bets. Sparse coverage alone → DROP.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.setdefault("ENABLE_HIGH_VALUE_FEATURES", "true")
os.environ.setdefault("ENABLE_PATHWAY_FEATURES", "false")
os.environ.setdefault("ENABLE_MARKET_FEATURES", "false")
os.environ.setdefault("INTERACTION_DISCOVERY_ENABLED", "false")

import joblib
import numpy as np
import pandas as pd

import config

config.refresh_runtime_env()

from src.data_loader import load_fights
from src.feature_engineering import apply_imputer
from src.gym_data import _location_overlap
from src.model_trainer import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ab_local")

YEAR = 2025
KEEP_AUC_DELTA = 0.005
LOCAL_COL = "local_advantage_diff"
MIN_NONZERO_FOR_KEEP = 30  # too sparse → force DROP regardless of AUC noise

REPORTS = config.DATA_DIR / "reports"
REPORT_MD = REPORTS / "local_advantage_ab_2025.md"
REPORT_JSON = REPORTS / "local_advantage_ab_2025.json"
FLAG_TXT = REPORTS / "local_advantage_flag.txt"


def _metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    pred = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")
    return {
        "n": float(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": auc,
        "brier": float(brier_score_loss(y, proba)),
    }


def _ensure_local_col(features: pd.DataFrame) -> pd.DataFrame:
    """Merge event location from fights and recompute local_advantage_*."""
    out = features.copy()
    fid = config.FIGHT_ID_COLUMN
    fights = load_fights()
    if fid in fights.columns and "location" in fights.columns:
        loc = fights[[fid, "location"]].drop_duplicates(fid, keep="last")
        if "location" in out.columns:
            out = out.drop(columns=["location"])
        out = out.merge(loc, on=fid, how="left")

    if "f1_gym_location" not in out.columns or "f2_gym_location" not in out.columns:
        from src.gym_data import attach_gym_features

        out = attach_gym_features(out)
    else:
        ev = out["location"].fillna("").astype(str) if "location" in out.columns else ""
        if isinstance(ev, str):
            ev = pd.Series([""] * len(out), index=out.index)
        f1 = [
            int(_location_overlap(str(g or ""), str(e or "")))
            for g, e in zip(out["f1_gym_location"].fillna(""), ev)
        ]
        f2 = [
            int(_location_overlap(str(g or ""), str(e or "")))
            for g, e in zip(out["f2_gym_location"].fillna(""), ev)
        ]
        out["f1_local_advantage"] = f1
        out["f2_local_advantage"] = f2
        out["local_advantage_diff"] = [a - b for a, b in zip(f1, f2)]
    return out


def _coverage(features: pd.DataFrame, year: int) -> dict[str, Any]:
    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    sub = features.loc[dts.dt.year == year]
    diff = pd.to_numeric(sub.get(LOCAL_COL), errors="coerce").fillna(0)
    loc_cov = 0.0
    if "location" in sub.columns:
        loc_cov = float(sub["location"].fillna("").astype(str).str.len().gt(0).mean())
    return {
        "n": int(len(sub)),
        "location_coverage": loc_cov,
        "nonzero_diff": int((diff != 0).sum()),
        "f1_local_sum": int(pd.to_numeric(sub.get("f1_local_advantage"), errors="coerce").fillna(0).sum())
        if "f1_local_advantage" in sub.columns
        else 0,
        "f2_local_sum": int(pd.to_numeric(sub.get("f2_local_advantage"), errors="coerce").fillna(0).sum())
        if "f2_local_advantage" in sub.columns
        else 0,
        "diff_value_counts": {str(k): int(v) for k, v in diff.value_counts().to_dict().items()},
    }


def run_arm(features: pd.DataFrame, *, use_local: bool) -> dict[str, Any]:
    os.environ["ENABLE_HIGH_VALUE_FEATURES"] = "true"
    os.environ["ENABLE_PATHWAY_FEATURES"] = "false"
    os.environ["ENABLE_MARKET_FEATURES"] = "false"
    os.environ["INTERACTION_DISCOVERY_ENABLED"] = "false"
    config.refresh_runtime_env()

    # Strip pathway/market if present on FEATURE_COLUMNS
    path = set(getattr(config, "PATHWAY_FEATURE_COLUMNS", []) or [])
    mkt = set(getattr(config, "MARKET_FEATURE_COLUMNS", []) or [])
    cols = [
        c
        for c in config.FEATURE_COLUMNS
        if c in features.columns and c not in path and c not in mkt
    ]
    if use_local:
        if LOCAL_COL not in features.columns:
            raise RuntimeError(f"Missing {LOCAL_COL}")
        if LOCAL_COL not in cols:
            cols = list(cols) + [LOCAL_COL]
    else:
        cols = [c for c in cols if c != LOCAL_COL]

    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    train = features.loc[dts.dt.year < YEAR].copy()
    test = features.loc[dts.dt.year == YEAR].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Need pre-{YEAR} train and {YEAR} test")

    saved = list(config.FEATURE_COLUMNS)
    config.FEATURE_COLUMNS = cols
    arm_dir = config.MODELS_DIR / "ab_local_advantage"
    arm_dir.mkdir(parents=True, exist_ok=True)
    model_path = arm_dir / ("with_local.joblib" if use_local else "base.joblib")
    try:
        result = train_model(
            train,
            model_path=model_path,
            tune="none",
            run_backtest_hook=False,
        )
    finally:
        config.FEATURE_COLUMNS = saved

    artifact = joblib.load(result.model_path)
    feat_cols = [c for c in (artifact.get("feature_columns") or cols) if c in test.columns]
    test_x = apply_imputer(test, artifact["imputer"]) if artifact.get("imputer") else test
    X = test_x[feat_cols]
    y = test[config.TARGET_COLUMN].astype(int).to_numpy()
    proba = np.asarray(artifact["model"].predict_proba(X)[:, 1], dtype=float)
    metrics = _metrics(y, proba)
    metrics["arm"] = "LOCAL" if use_local else "BASE"
    metrics["n_features"] = float(len(feat_cols))
    metrics["has_local_col"] = LOCAL_COL in feat_cols
    metrics["model_path"] = str(model_path)
    return metrics


def _keep(base: dict, local: dict, coverage: dict) -> tuple[bool, str]:
    if int(coverage.get("nonzero_diff") or 0) < MIN_NONZERO_FOR_KEEP:
        return (
            False,
            f"insufficient_coverage_nonzero={coverage.get('nonzero_diff')} "
            f"(need>={MIN_NONZERO_FOR_KEEP})",
        )
    d_auc = float(local["auc"] - base["auc"])
    if d_auc >= KEEP_AUC_DELTA:
        return True, f"auc_improved_by_{d_auc:+.4f}_ge_{KEEP_AUC_DELTA}"
    return False, f"no_keep_d_auc={d_auc:+.4f}"


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(f):
        return "—"
    return f"{f:.{digits}f}"


def main() -> int:
    cache = config.PROCESSED_DIR / "ab_feature_matrix_pathway_v2.parquet"
    if not cache.is_file():
        cache = config.PROCESSED_DIR / "ab_feature_matrix_v5.parquet"
    if not cache.is_file():
        raise SystemExit(f"Missing feature cache under {config.PROCESSED_DIR}")

    logger.info("Loading %s", cache)
    features = pd.read_parquet(cache)
    # Drop pathway/market cols from training matrix if present (arms stay HV-only)
    drop_extra = [
        c
        for c in list(getattr(config, "PATHWAY_FEATURE_COLUMNS", []) or [])
        + list(getattr(config, "MARKET_FEATURE_COLUMNS", []) or [])
        if c in features.columns
    ]
    # Keep them in frame but arms won't select them

    logger.info("Recomputing local_advantage with event location merge…")
    features = _ensure_local_col(features)
    cov = _coverage(features, YEAR)
    logger.info("2025 coverage: %s", cov)

    base = run_arm(features, use_local=False)
    logger.info("BASE: %s", {k: base[k] for k in ("accuracy", "auc", "n_features")})
    local = run_arm(features, use_local=True)
    logger.info("LOCAL: %s", {k: local[k] for k in ("accuracy", "auc", "n_features")})

    keep, reason = _keep(base, local, cov)
    decision = "keep" if keep else "drop"
    report = {
        "year": YEAR,
        "decision": decision,
        "reason": reason,
        "keep_auc_delta": KEEP_AUC_DELTA,
        "coverage_2025": cov,
        "arms": {"BASE": base, "LOCAL": local},
        "delta_auc": float(local["auc"] - base["auc"]),
        "delta_accuracy": float(local["accuracy"] - base["accuracy"]),
        "isolation": {
            "retrained_ensemble_winner": False,
            "enable_pathway_features": False,
            "enable_market_features": False,
            "live_ha_thresholds_changed": False,
            "added_to_production_feature_columns": False,
        },
        "notes": [
            "local_advantage_diff = f1_local - f2_local from gym city/region vs event location.",
            "Feature matrix historically lacked event location → local flags were always 0; "
            "this run merges fights.location and recomputes before training.",
            "Signal remains sparse (few gym profiles + partial event location coverage).",
        ],
        "dropped_cols_ignored": drop_extra[:20],
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Local advantage A/B — {YEAR} holdout",
        "",
        "UFC-only research. HV baseline. Pathway/market OFF. "
        "No production retrain / Live HA changes.",
        "",
        f"**Decision: {decision.upper()}** — {reason}",
        "",
        "## Coverage (2025)",
        "",
        f"- Fights: **{cov['n']}**",
        f"- Event location non-empty: **{_fmt(cov['location_coverage'] * 100, 1)}%**",
        f"- Nonzero `local_advantage_diff`: **{cov['nonzero_diff']}** "
        f"(f1 local={cov['f1_local_sum']}, f2 local={cov['f2_local_sum']})",
        f"- Diff counts: `{cov['diff_value_counts']}`",
        "",
        "## Side-by-side",
        "",
        "| Arm | n_feat | Acc | AUC | Brier |",
        "|-----|-------:|----:|----:|------:|",
        f"| BASE | {int(base['n_features'])} | {_fmt(base['accuracy'])} | "
        f"{_fmt(base['auc'])} | {_fmt(base['brier'])} |",
        f"| LOCAL | {int(local['n_features'])} | {_fmt(local['accuracy'])} | "
        f"{_fmt(local['auc'])} | {_fmt(local['brier'])} |",
        "",
        f"ΔAUC = {_fmt(report['delta_auc'])} (keep if ≥ +{KEEP_AUC_DELTA} and coverage OK)",
        "",
        "## Notes",
        "",
    ]
    for n in report["notes"]:
        lines.append(f"- {n}")
    lines += [
        "",
        f"Artifacts: `{REPORT_JSON.name}`, `{FLAG_TXT.name}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    FLAG_TXT.write_text(
        f"# local_advantage_diff production flag\n"
        f"# {decision.upper()}: {reason}\n"
        f"ADD_LOCAL_ADVANTAGE_TO_FEATURES=false\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s (%s)", REPORT_MD, decision)
    print(REPORT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
