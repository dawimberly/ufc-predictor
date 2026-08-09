"""2025 A/B: decision_profile features vs HV BASE.

Arms
  BASE — HV on, pathway/market off, interactions frozen
  DEC  — BASE + decision_profile diffs (dec/split/finish-share)

Does NOT retrain ensemble_winner.joblib unless KEEP (still no auto-retrain here).
Does NOT enable pathway/market/home flags. No judge/geo features. No Live HA changes.
"""

from __future__ import annotations

import hashlib
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
from src.decision_profile import (
    DECISION_PROFILE_DIFF_COLUMNS,
    attach_decision_profile_to_wide,
)
from src.feature_engineering import apply_imputer
from src.model_trainer import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ab_dec")

YEAR = 2025
KEEP_AUC_DELTA = 0.005
EDGE_MIN = 0.03

REPORTS = config.DATA_DIR / "reports"
REPORT_MD = REPORTS / "decision_profile_ab_2025.md"
REPORT_JSON = REPORTS / "decision_profile_ab_2025.json"
FLAG_TXT = REPORTS / "decision_profile_flag.txt"


def _fingerprint_cols(cols: list[str]) -> str:
    return hashlib.sha256("|".join(cols).encode("utf-8")).hexdigest()[:16]


def _max_drawdown(pnls: list[float]) -> float:
    if not pnls:
        return float("nan")
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def _flat_edge(
    y: np.ndarray,
    proba: np.ndarray,
    f1_odds: np.ndarray | None,
    f2_odds: np.ndarray | None,
    *,
    edge_min: float = EDGE_MIN,
) -> dict[str, float]:
    if f1_odds is None or f2_odds is None:
        return {
            "flat_edge_roi": float("nan"),
            "n_bets": 0.0,
            "hit_rate": float("nan"),
            "max_dd": float("nan"),
            "n_with_odds": 0.0,
        }
    o1 = pd.to_numeric(pd.Series(f1_odds), errors="coerce").to_numpy()
    o2 = pd.to_numeric(pd.Series(f2_odds), errors="coerce").to_numpy()
    n_odds = float(((o1 > 1.0) & (o2 > 1.0)).sum())
    pnls: list[float] = []
    hits = 0
    for yt, p, a, b in zip(y, proba, o1, o2):
        try:
            o1f, o2f = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if not (o1f > 1.0 and o2f > 1.0):
            continue
        imp1 = (1.0 / o1f) / ((1.0 / o1f) + (1.0 / o2f))
        edge1 = float(p) - imp1
        edge2 = (1.0 - float(p)) - (1.0 - imp1)
        if edge1 >= edge_min and edge1 >= edge2:
            won = int(yt) == 1
            pnls.append((o1f - 1.0) if won else -1.0)
            hits += int(won)
        elif edge2 >= edge_min:
            won = int(yt) == 0
            pnls.append((o2f - 1.0) if won else -1.0)
            hits += int(won)
    n = len(pnls)
    return {
        "flat_edge_roi": (sum(pnls) / n) if n else float("nan"),
        "n_bets": float(n),
        "hit_rate": (hits / n) if n else float("nan"),
        "max_dd": _max_drawdown(pnls),
        "n_with_odds": n_odds,
    }


def _metrics(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    f1_odds: np.ndarray | None,
    f2_odds: np.ndarray | None,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    pred = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")
    out = {
        "n": float(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": auc,
        "brier": float(brier_score_loss(y, proba)),
    }
    out.update(_flat_edge(y, proba, f1_odds, f2_odds))
    return out


def _coverage(features: pd.DataFrame, year: int) -> dict[str, Any]:
    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    sub = features.loc[dts.dt.year == year]
    cov = {"n": int(len(sub))}
    for col in DECISION_PROFILE_DIFF_COLUMNS:
        if col not in sub.columns:
            cov[col] = {"present": False}
            continue
        s = pd.to_numeric(sub[col], errors="coerce")
        cov[col] = {
            "present": True,
            "nonnull_pct": float(s.notna().mean()),
            "nonzero_pct": float((s.fillna(0) != 0).mean()),
        }
    return cov


def run_arm(features: pd.DataFrame, *, use_dec: bool) -> dict[str, Any]:
    os.environ["ENABLE_HIGH_VALUE_FEATURES"] = "true"
    os.environ["ENABLE_PATHWAY_FEATURES"] = "false"
    os.environ["ENABLE_MARKET_FEATURES"] = "false"
    os.environ["INTERACTION_DISCOVERY_ENABLED"] = "false"
    config.refresh_runtime_env()

    path = set(getattr(config, "PATHWAY_FEATURE_COLUMNS", []) or [])
    mkt = set(getattr(config, "MARKET_FEATURE_COLUMNS", []) or [])
    dec = set(DECISION_PROFILE_DIFF_COLUMNS)

    cols = [
        c
        for c in config.FEATURE_COLUMNS
        if c in features.columns and c not in path and c not in mkt and c not in dec
    ]
    if use_dec:
        for c in DECISION_PROFILE_DIFF_COLUMNS:
            if c in features.columns and c not in cols:
                cols.append(c)

    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    train = features.loc[dts.dt.year < YEAR].copy()
    test = features.loc[dts.dt.year == YEAR].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Need pre-{YEAR} train and {YEAR} test")

    saved = list(config.FEATURE_COLUMNS)
    config.FEATURE_COLUMNS = cols
    arm_dir = config.MODELS_DIR / "ab_decision_profile"
    arm_dir.mkdir(parents=True, exist_ok=True)
    model_path = arm_dir / ("dec.joblib" if use_dec else "base.joblib")
    try:
        result = train_model(
            train, model_path=model_path, tune="none", run_backtest_hook=False
        )
    finally:
        config.FEATURE_COLUMNS = saved

    artifact = joblib.load(result.model_path)
    feat_cols = [c for c in (artifact.get("feature_columns") or cols) if c in test.columns]
    test_x = apply_imputer(test, artifact["imputer"]) if artifact.get("imputer") else test
    y = test[config.TARGET_COLUMN].astype(int).to_numpy()
    proba = np.asarray(artifact["model"].predict_proba(test_x[feat_cols])[:, 1], dtype=float)
    f1o = test["f1_odds"].to_numpy() if "f1_odds" in test.columns else None
    f2o = test["f2_odds"].to_numpy() if "f2_odds" in test.columns else None
    metrics = _metrics(y, proba, f1_odds=f1o, f2_odds=f2o)
    metrics["arm"] = "DEC" if use_dec else "BASE"
    metrics["n_features"] = float(len(feat_cols))
    metrics["feature_fingerprint"] = _fingerprint_cols(feat_cols)
    metrics["artifact_fingerprint"] = str(artifact.get("features_fingerprint") or "")
    metrics["dec_cols_in_model"] = [c for c in feat_cols if c in dec]
    metrics["model_path"] = str(model_path)
    return metrics


def _keep(base: dict, dec: dict) -> tuple[bool, str]:
    d_auc = float(dec["auc"] - base["auc"])
    if d_auc >= KEEP_AUC_DELTA:
        return True, f"auc_improved_by_{d_auc:+.4f}_ge_{KEEP_AUC_DELTA}"
    b_roi, c_roi = base.get("flat_edge_roi"), dec.get("flat_edge_roi")
    b_dd, c_dd = base.get("max_dd"), dec.get("max_dd")
    if (
        b_roi is not None
        and c_roi is not None
        and not np.isnan(b_roi)
        and not np.isnan(c_roi)
        and float(dec.get("n_bets") or 0) >= 20
    ):
        roi_better = float(c_roi) > float(b_roi) + 1e-6
        dd_ok = True
        if b_dd is not None and c_dd is not None and not np.isnan(b_dd) and not np.isnan(c_dd):
            dd_ok = float(c_dd) >= float(b_dd) - 0.05
        if roi_better and dd_ok:
            return True, f"flat_edge_roi_dd_tradeoff_better_d_roi={c_roi - b_roi:+.4f}"
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
    logger.info("Loading %s", cache)
    features = pd.read_parquet(cache)

    logger.info("Attaching decision_profile columns…")
    features = attach_decision_profile_to_wide(features, fights=load_fights())
    cov = _coverage(features, YEAR)
    logger.info("2025 coverage keys=%s", list(cov.keys())[:6])

    base = run_arm(features, use_dec=False)
    logger.info("BASE auc=%s n_feat=%s fp=%s", base["auc"], base["n_features"], base["feature_fingerprint"])
    dec = run_arm(features, use_dec=True)
    logger.info(
        "DEC auc=%s n_feat=%s fp=%s cols=%s",
        dec["auc"],
        dec["n_features"],
        dec["feature_fingerprint"],
        dec["dec_cols_in_model"],
    )

    keep, reason = _keep(base, dec)
    decision = "keep" if keep else "drop"
    report = {
        "year": YEAR,
        "decision": decision,
        "reason": reason,
        "keep_auc_delta": KEEP_AUC_DELTA,
        "coverage_2025": cov,
        "arms": {"BASE": base, "DEC": dec},
        "delta_auc": float(dec["auc"] - base["auc"]),
        "delta_accuracy": float(dec["accuracy"] - base["accuracy"]),
        "isolation": {
            "retrained_ensemble_winner": False,
            "enable_pathway_features": False,
            "enable_market_features": False,
            "added_to_feature_columns": bool(keep),
            "live_ha_changed": False,
            "judge_geo_included": False,
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Decision-profile A/B — {YEAR}",
        "",
        "UFC-only. HV BASE vs +decision_profile. Pathway/market/home OFF. "
        "Interactions frozen. No judge/geo features. No Live HA changes.",
        "",
        f"**Decision: {decision.upper()}** — {reason}",
        "",
        "## Side-by-side",
        "",
        "| Arm | n_feat | fingerprint | Acc | AUC | Brier | ROI@3% | bets | hit | maxDD |",
        "|-----|-------:|---|----:|----:|------:|-------:|-----:|----:|------:|",
        f"| BASE | {int(base['n_features'])} | `{base['feature_fingerprint']}` | "
        f"{_fmt(base['accuracy'])} | {_fmt(base['auc'])} | {_fmt(base['brier'])} | "
        f"{_fmt(base['flat_edge_roi'])} | {int(base['n_bets'])} | {_fmt(base['hit_rate'], 3)} | "
        f"{_fmt(base['max_dd'])} |",
        f"| DEC | {int(dec['n_features'])} | `{dec['feature_fingerprint']}` | "
        f"{_fmt(dec['accuracy'])} | {_fmt(dec['auc'])} | {_fmt(dec['brier'])} | "
        f"{_fmt(dec['flat_edge_roi'])} | {int(dec['n_bets'])} | {_fmt(dec['hit_rate'], 3)} | "
        f"{_fmt(dec['max_dd'])} |",
        "",
        f"ΔAUC = {_fmt(report['delta_auc'])} (keep if ≥ +{KEEP_AUC_DELTA})",
        f"DEC cols in model: `{dec.get('dec_cols_in_model')}`",
        "",
        "## Coverage (2025)",
        "",
        "| Column | nonnull% | nonzero% |",
        "|---|---:|---:|",
    ]
    for col in DECISION_PROFILE_DIFF_COLUMNS:
        info = cov.get(col) or {}
        if not info.get("present"):
            lines.append(f"| {col} | — | — |")
        else:
            lines.append(
                f"| {col} | {_fmt(100 * info['nonnull_pct'], 1)} | "
                f"{_fmt(100 * info['nonzero_pct'], 1)} |"
            )
    lines += [
        "",
        "## Recommendation",
        "",
        f"**{decision.upper()}** — leave `dec_*` out of production `FEATURE_COLUMNS`; "
        "display-only context strip remains fine."
        if decision == "drop"
        else f"**KEEP** — candidate for FEATURE_COLUMNS add + controlled retrain "
        "(not applied automatically by this script).",
        "",
        f"Flag file: `{FLAG_TXT.name}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    FLAG_TXT.write_text(
        f"# decision_profile production flag\n"
        f"# {decision.upper()}: {reason}\n"
        f"ADD_DECISION_PROFILE_TO_FEATURES={'true' if keep else 'false'}\n",
        encoding="utf-8",
    )

    # Update research_keep_drop.md
    keep_drop = REPORTS / "research_keep_drop.md"
    if keep_drop.is_file():
        text = keep_drop.read_text(encoding="utf-8")
        row = (
            f"| Decision profile (dec/split/share) | **{decision.upper()}** | "
            f"ΔAUC {_fmt(report['delta_auc'])}; {reason} |"
        )
        old = "| Decision profile (dec/split/share) | **Display now** | Optional one-shot A/B later; not in FEATURE_COLUMNS yet |"
        if old in text:
            text = text.replace(old, row)
        elif "Decision profile (dec/split/share)" not in text:
            text = text.replace(
                "| Judge × geography | **DROP (model)** | Usable n=41; display notes OK |",
                "| Judge × geography | **DROP (model)** | Usable n=41; display notes OK |\n" + row,
            )
        else:
            # replace any existing decision profile row
            import re

            text = re.sub(
                r"\| Decision profile \(dec/split/share\) \|.*?\|.*?\|",
                row,
                text,
            )
        keep_drop.write_text(text, encoding="utf-8")

    print(REPORT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
