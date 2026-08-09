"""2025 A/B: home_country_diff (card / regional home) vs HV baseline.

Tests the \"UFC in Australia helps Aussies\" style hypothesis using a
leakage-safe country proxy (gym country, else modal prior event country).
True nationality caches are empty/noisy — documented in the report.

Does NOT retrain ensemble_winner.joblib, enable pathway/market, or change Live HA.
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
from src.home_country import (
    HOME_COUNTRY_FEATURE_COLUMNS,
    attach_home_country_features,
    log_home_country_coverage,
)
from src.model_trainer import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ab_home_country")

YEAR = 2025
KEEP_AUC_DELTA = 0.005
MIN_NONZERO = 30

REPORTS = config.DATA_DIR / "reports"
REPORT_MD = REPORTS / "home_country_ab_2025.md"
REPORT_JSON = REPORTS / "home_country_ab_2025.json"
FLAG_TXT = REPORTS / "home_country_flag.txt"


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


def _coverage(df: pd.DataFrame, year: int) -> dict[str, Any]:
    dts = pd.to_datetime(df[config.DATE_COLUMN], errors="coerce")
    sub = df.loc[dts.dt.year == year].copy()
    diff = pd.to_numeric(sub.get("home_country_diff"), errors="coerce").fillna(0)
    ec = sub["event_country"].fillna("").astype(str) if "event_country" in sub.columns else ""
    non_us = ec.ne("") & ec.ne("usa") if hasattr(ec, "ne") else pd.Series([False] * len(sub))
    return {
        "n": int(len(sub)),
        "event_country_pct": float(ec.str.len().gt(0).mean()) if hasattr(ec, "str") else 0.0,
        "nonzero_diff": int((diff != 0).sum()),
        "any_home": int(
            (
                (
                    pd.to_numeric(sub.get("f1_home_country"), errors="coerce").fillna(0)
                    + pd.to_numeric(sub.get("f2_home_country"), errors="coerce").fillna(0)
                )
                > 0
            ).sum()
        ),
        "non_usa_events": int(non_us.sum()) if hasattr(non_us, "sum") else 0,
        "nonzero_on_non_usa": int(((diff != 0) & non_us).sum()) if hasattr(non_us, "sum") else 0,
        "diff_vc": {str(k): int(v) for k, v in diff.value_counts().to_dict().items()},
        "top_event_countries": (
            ec[ec.str.len() > 0].value_counts().head(8).to_dict() if hasattr(ec, "str") else {}
        ),
    }


def run_arm(features: pd.DataFrame, *, use_home: bool) -> dict[str, Any]:
    os.environ["ENABLE_HIGH_VALUE_FEATURES"] = "true"
    os.environ["ENABLE_PATHWAY_FEATURES"] = "false"
    os.environ["ENABLE_MARKET_FEATURES"] = "false"
    os.environ["INTERACTION_DISCOVERY_ENABLED"] = "false"
    config.refresh_runtime_env()

    path = set(getattr(config, "PATHWAY_FEATURE_COLUMNS", []) or [])
    mkt = set(getattr(config, "MARKET_FEATURE_COLUMNS", []) or [])
    cols = [
        c
        for c in config.FEATURE_COLUMNS
        if c in features.columns and c not in path and c not in mkt
    ]
    home_cols = [c for c in HOME_COUNTRY_FEATURE_COLUMNS if c in features.columns]
    if use_home:
        for c in home_cols:
            if c not in cols:
                cols.append(c)
    else:
        cols = [c for c in cols if c not in set(HOME_COUNTRY_FEATURE_COLUMNS)]

    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    train = features.loc[dts.dt.year < YEAR].copy()
    test = features.loc[dts.dt.year == YEAR].copy()

    saved = list(config.FEATURE_COLUMNS)
    config.FEATURE_COLUMNS = cols
    arm_dir = config.MODELS_DIR / "ab_home_country"
    arm_dir.mkdir(parents=True, exist_ok=True)
    model_path = arm_dir / ("with_home.joblib" if use_home else "base.joblib")
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
    metrics = _metrics(y, proba)
    metrics["arm"] = "HOME" if use_home else "BASE"
    metrics["n_features"] = float(len(feat_cols))
    metrics["home_cols"] = [c for c in feat_cols if c in HOME_COUNTRY_FEATURE_COLUMNS]
    metrics["model_path"] = str(model_path)

    # Slice: non-USA events only (where the hypothesis lives)
    if "event_country" in test.columns:
        mask = test["event_country"].fillna("").astype(str).ne("") & test[
            "event_country"
        ].fillna("").astype(str).ne("usa")
        if mask.any():
            metrics["non_usa"] = _metrics(y[mask.to_numpy()], proba[mask.to_numpy()])
            metrics["non_usa"]["n"] = float(mask.sum())
    return metrics


def _keep(base: dict, home: dict, cov: dict) -> tuple[bool, str]:
    if int(cov.get("nonzero_diff") or 0) < MIN_NONZERO:
        return False, f"insufficient_coverage_nonzero={cov.get('nonzero_diff')} (need>={MIN_NONZERO})"
    d_auc = float(home["auc"] - base["auc"])
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
    logger.info("Loading %s", cache)
    features = pd.read_parquet(cache)
    fights = load_fights()
    logger.info("Attaching home_country features…")
    features = attach_home_country_features(features, fights=fights)
    log_home_country_coverage(features, year=YEAR, label="2025")
    cov = _coverage(features, YEAR)
    logger.info("coverage %s", cov)

    base = run_arm(features, use_home=False)
    logger.info("BASE auc=%s", base.get("auc"))
    home = run_arm(features, use_home=True)
    logger.info("HOME auc=%s cols=%s", home.get("auc"), home.get("home_cols"))

    keep, reason = _keep(base, home, cov)
    decision = "keep" if keep else "drop"
    report = {
        "year": YEAR,
        "decision": decision,
        "reason": reason,
        "coverage_2025": cov,
        "arms": {"BASE": base, "HOME": home},
        "delta_auc": float(home["auc"] - base["auc"]),
        "delta_accuracy": float(home["accuracy"] - base["accuracy"]),
        "proxy_note": (
            "Fighter country = gym country if known, else modal country of prior UFC "
            "event locations (as-of). Not scraped passport nationality (cache empty/noisy)."
        ),
        "isolation": {
            "retrained_ensemble_winner": False,
            "enable_pathway_features": False,
            "enable_market_features": False,
            "live_ha_thresholds_changed": False,
            "added_to_production_feature_columns": False,
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    nu_b = base.get("non_usa") or {}
    nu_h = home.get("non_usa") or {}
    lines = [
        f"# Home-country (card) A/B — {YEAR}",
        "",
        "Hypothesis: fighters with a regional/national tie to the event country "
        "(e.g. UFC Australia) outperform relative to the HV baseline.",
        "",
        f"**Decision: {decision.upper()}** — {reason}",
        "",
        f"Proxy: {report['proxy_note']}",
        "",
        "## Coverage (2025)",
        "",
        f"- Fights: **{cov['n']}**",
        f"- Event country mapped: **{_fmt(100 * cov['event_country_pct'], 1)}%**",
        f"- Non-USA events: **{cov['non_usa_events']}**",
        f"- Any home fighter: **{cov['any_home']}**",
        f"- Nonzero `home_country_diff`: **{cov['nonzero_diff']}** "
        f"(on non-USA: {cov['nonzero_on_non_usa']})",
        f"- Diff counts: `{cov['diff_vc']}`",
        f"- Top event countries: `{cov['top_event_countries']}`",
        "",
        "## Side-by-side (all 2025)",
        "",
        "| Arm | n_feat | Acc | AUC | Brier |",
        "|-----|-------:|----:|----:|------:|",
        f"| BASE | {int(base['n_features'])} | {_fmt(base['accuracy'])} | "
        f"{_fmt(base['auc'])} | {_fmt(base['brier'])} |",
        f"| HOME | {int(home['n_features'])} | {_fmt(home['accuracy'])} | "
        f"{_fmt(home['auc'])} | {_fmt(home['brier'])} |",
        "",
        f"ΔAUC = {_fmt(report['delta_auc'])} (keep if ≥ +{KEEP_AUC_DELTA} and coverage OK)",
        "",
        "## Non-USA event slice",
        "",
        f"| Arm | n | Acc | AUC |",
        f"|-----|--:|----:|----:|",
        f"| BASE | {int(nu_b.get('n') or 0)} | {_fmt(nu_b.get('accuracy'))} | {_fmt(nu_b.get('auc'))} |",
        f"| HOME | {int(nu_h.get('n') or 0)} | {_fmt(nu_h.get('accuracy'))} | {_fmt(nu_h.get('auc'))} |",
        "",
        "Flags stay off / not added to production FEATURE_COLUMNS unless keep passes.",
        "",
        f"Artifacts: `{REPORT_JSON.name}`, `{FLAG_TXT.name}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    FLAG_TXT.write_text(
        f"# home_country_diff production flag\n"
        f"# {decision.upper()}: {reason}\n"
        f"ADD_HOME_COUNTRY_TO_FEATURES=false\n",
        encoding="utf-8",
    )
    print(REPORT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
