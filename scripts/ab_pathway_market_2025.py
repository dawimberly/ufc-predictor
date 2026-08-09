"""2025 A/B: pathway + market features vs current HV baseline (UFC only).

Arms
  BASE     — HV on (production)
  PATH     — BASE + pathway win/loss features
  PATH+MKT — PATH + mkt_implied_prob (+ line_move when available)
  CAL      — PATH+MKT probs with research-only wide-CI shrink toward market

Same chronological split as hv A/B: train on pre-2025, score 2025 holdout.
Interactions frozen (INTERACTION_DISCOVERY_ENABLED=false) across arms.

Does NOT change dashboard colors, Paper override, Live HA thresholds, or
any trading-bot code/paths. Reports only under data/reports/.
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

# Keep production defaults until arms flip flags explicitly.
os.environ.setdefault("ENABLE_HIGH_VALUE_FEATURES", "true")
os.environ.setdefault("ENABLE_PATHWAY_FEATURES", "false")
os.environ.setdefault("ENABLE_MARKET_FEATURES", "false")
os.environ.setdefault("ENABLE_PATHWAY_MARKET_CAL", "false")
os.environ.setdefault("INTERACTION_DISCOVERY_ENABLED", "false")

import numpy as np
import pandas as pd

import config

config.refresh_runtime_env()

from src.data_loader import ensure_data_dirs, load_fights
from src.ensemble import prediction_interval
from src.feature_engineering import apply_imputer, build_feature_matrix
from src.market_features import (
    MARKET_FEATURE_COLUMNS,
    log_market_coverage,
    model_minus_market,
    shrink_proba_toward_market,
)
from src.model_trainer import train_model
from src.pathway_features import PATHWAY_DIFF_COLUMNS, log_pathway_coverage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ab_pathway_mkt")

YEAR = 2025
KEEP_AUC_DELTA = 0.005
# Match paper-ish wide skip for research diagnostics (not Live HA).
WIDE_CI_THRESHOLD = float(os.getenv("AB_WIDE_CI_THRESHOLD", "0.40"))


def _nanmean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _fingerprint_cols(cols: list[str]) -> str:
    blob = "|".join(cols).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _max_drawdown(pnls: list[float]) -> float:
    if not pnls:
        return float("nan")
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min()) if len(dd) else float("nan")


def _flat_edge_stats(
    y: np.ndarray,
    proba: np.ndarray,
    f1_odds: np.ndarray | None,
    f2_odds: np.ndarray | None,
    *,
    edge_min: float,
) -> dict[str, float]:
    if f1_odds is None or f2_odds is None:
        return {
            "flat_edge_roi": float("nan"),
            "n_bets": 0.0,
            "hit_rate": float("nan"),
            "max_dd": float("nan"),
        }
    pnl_list: list[float] = []
    hits = 0
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
            won = int(yt) == 1
            pnl = (o1f - 1.0) if won else -1.0
            pnl_list.append(pnl)
            hits += int(won)
        elif edge2 >= edge_min:
            won = int(yt) == 0
            pnl = (o2f - 1.0) if won else -1.0
            pnl_list.append(pnl)
            hits += int(won)
    n = len(pnl_list)
    return {
        "flat_edge_roi": (sum(pnl_list) / n) if n else float("nan"),
        "n_bets": float(n),
        "hit_rate": (hits / n) if n else float("nan"),
        "max_dd": _max_drawdown(pnl_list),
    }


def _metrics_bundle(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    f1_odds: np.ndarray | None,
    f2_odds: np.ndarray | None,
    ci_width: np.ndarray | None,
    mkt: np.ndarray | None,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, roc_auc_score

    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y, pred))
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")

    odds_mask = None
    if f1_odds is not None and f2_odds is not None:
        o1 = pd.to_numeric(pd.Series(f1_odds), errors="coerce").to_numpy()
        o2 = pd.to_numeric(pd.Series(f2_odds), errors="coerce").to_numpy()
        odds_mask = (o1 > 1.0) & (o2 > 1.0)

    out: dict[str, Any] = {
        "n": float(len(y)),
        "accuracy": acc,
        "auc": auc,
        "n_with_odds": float(int(odds_mask.sum())) if odds_mask is not None else 0.0,
    }
    for edge_min, key in ((0.03, "edge_3"), (0.05, "edge_5"), (0.08, "edge_8")):
        stats = _flat_edge_stats(y, proba, f1_odds, f2_odds, edge_min=edge_min)
        out[f"{key}_roi"] = stats["flat_edge_roi"]
        out[f"{key}_n_bets"] = stats["n_bets"]
        out[f"{key}_hit_rate"] = stats["hit_rate"]
        out[f"{key}_max_dd"] = stats["max_dd"]
    # Alias primary flat-stake ROI at 3% for keep-rule compatibility
    out["flat_edge_roi"] = out["edge_3_roi"]
    out["n_bets"] = out["edge_3_n_bets"]
    out["hit_rate"] = out["edge_3_hit_rate"]
    out["max_dd"] = out["edge_3_max_dd"]

    if ci_width is not None and len(ci_width) == len(y):
        wide = np.asarray(ci_width, dtype=float) >= WIDE_CI_THRESHOLD
        narrow = ~wide
        out["wide_ci_rate"] = float(wide.mean()) if len(wide) else float("nan")
        if wide.any():
            out["wide_ci_miss_rate"] = float(1.0 - accuracy_score(y[wide], pred[wide]))
        else:
            out["wide_ci_miss_rate"] = float("nan")
        if narrow.any():
            out["narrow_ci_miss_rate"] = float(
                1.0 - accuracy_score(y[narrow], pred[narrow])
            )
        else:
            out["narrow_ci_miss_rate"] = float("nan")
    else:
        out["wide_ci_rate"] = float("nan")
        out["wide_ci_miss_rate"] = float("nan")
        out["narrow_ci_miss_rate"] = float("nan")

    if mkt is not None:
        resid = model_minus_market(proba, mkt)
        finite = np.isfinite(resid)
        out["mean_model_minus_mkt"] = (
            float(np.mean(resid[finite])) if finite.any() else float("nan")
        )
        out["mkt_coverage_pct"] = float(np.isfinite(mkt).mean() * 100.0)
    else:
        out["mean_model_minus_mkt"] = float("nan")
        out["mkt_coverage_pct"] = 0.0
    return out


def _set_arm_flags(*, pathway: bool, market: bool) -> None:
    os.environ["ENABLE_HIGH_VALUE_FEATURES"] = "true"
    os.environ["ENABLE_PATHWAY_FEATURES"] = "true" if pathway else "false"
    os.environ["ENABLE_MARKET_FEATURES"] = "true" if market else "false"
    os.environ["INTERACTION_DISCOVERY_ENABLED"] = "false"
    config.refresh_runtime_env()


def run_arm(
    features: pd.DataFrame,
    *,
    arm: str,
    pathway: bool,
    market: bool,
    apply_cal: bool = False,
) -> dict[str, Any]:
    _set_arm_flags(pathway=pathway, market=market)
    hv = set(config.HIGH_VALUE_FEATURE_COLUMNS)
    path_cols = set(config.PATHWAY_FEATURE_COLUMNS)
    mkt_cols = set(config.MARKET_FEATURE_COLUMNS)
    cols = [c for c in config.FEATURE_COLUMNS if c in features.columns]

    # Sanity: BASE must include HV; PATH must include pathway; etc.
    assert any(c in hv for c in cols), f"{arm}: missing HV columns"
    if pathway:
        assert any(c in path_cols for c in cols), f"{arm}: missing pathway columns"
    else:
        cols = [c for c in cols if c not in path_cols]
    if market:
        assert any(c in mkt_cols for c in cols), f"{arm}: missing market columns"
        # line_move may be all-NaN — still include mkt_implied_prob
        cols = [c for c in cols if c in features.columns]
        # Drop all-NaN market cols so imputer/dropna does not kill the arm
        keep = []
        for c in cols:
            if c in mkt_cols and features[c].notna().sum() == 0:
                logger.warning("%s: dropping all-NaN market col %s", arm, c)
                continue
            keep.append(c)
        cols = keep
    else:
        cols = [c for c in cols if c not in mkt_cols]

    date_col = config.DATE_COLUMN
    dts = pd.to_datetime(features[date_col], errors="coerce")
    train = features.loc[dts.dt.year < YEAR].copy()
    test = features.loc[dts.dt.year == YEAR].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Need pre-{YEAR} train and {YEAR} test rows")

    saved = list(config.FEATURE_COLUMNS)
    config.FEATURE_COLUMNS = cols
    arm_dir = config.MODELS_DIR / "ab_pathway_market"
    arm_dir.mkdir(parents=True, exist_ok=True)
    safe = arm.replace("+", "_").replace(" ", "_").lower()
    model_path = arm_dir / f"{safe}.joblib"
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
        raise TypeError(type(artifact))
    feat_cols = list(artifact.get("feature_columns") or result.feature_columns)
    feat_cols = [c for c in feat_cols if c in test.columns]

    test_x = test.copy()
    imputer = artifact.get("imputer")
    if imputer is not None:
        test_x = apply_imputer(test_x, imputer)

    X = test_x[feat_cols]
    y = test[config.TARGET_COLUMN].astype(int).to_numpy()
    clf = artifact["model"]
    proba = np.asarray(clf.predict_proba(X)[:, 1], dtype=float)

    conformal_q = float(artifact.get("conformal_q") or 0.0)
    _, _, ci_width = prediction_interval(proba, conformal_q=conformal_q)

    mkt = (
        test["mkt_implied_prob"].to_numpy(dtype=float)
        if "mkt_implied_prob" in test.columns
        else None
    )
    if apply_cal and mkt is not None:
        proba = shrink_proba_toward_market(
            proba,
            mkt,
            ci_width,
            width_threshold=float(config.PATHWAY_MARKET_CAL_WIDTH),
            shrink=float(config.PATHWAY_MARKET_CAL_SHRINK),
        )

    f1o = test["f1_odds"].to_numpy() if "f1_odds" in test.columns else None
    f2o = test["f2_odds"].to_numpy() if "f2_odds" in test.columns else None
    metrics = _metrics_bundle(
        y, proba, f1_odds=f1o, f2_odds=f2o, ci_width=ci_width, mkt=mkt
    )
    metrics["arm"] = arm
    metrics["n_features"] = float(len(feat_cols))
    metrics["feature_fingerprint"] = _fingerprint_cols(feat_cols)
    metrics["pathway"] = bool(pathway)
    metrics["market"] = bool(market)
    metrics["cal_applied"] = bool(apply_cal)
    metrics["conformal_q"] = conformal_q
    metrics["features_fingerprint_artifact"] = str(
        artifact.get("features_fingerprint") or ""
    )
    return metrics


def _load_or_build_features() -> pd.DataFrame:
    cache = config.PROCESSED_DIR / "ab_feature_matrix_pathway_v2.parquet"
    force = os.getenv("FORCE_REBUILD_FEATURES", "").lower() in ("1", "true", "yes")
    if cache.is_file() and not force:
        logger.info("Loading cached features %s", cache)
        features = pd.read_parquet(cache)
    else:
        logger.info("Building feature matrix (pathway + market cols always computed)…")
        features = build_feature_matrix(
            load_fights(), keep_unlabeled=False, use_fighter_cache=False
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache)
        logger.info("Cached → %s (%s rows)", cache, len(features))

    missing_path = [c for c in PATHWAY_DIFF_COLUMNS if c not in features.columns]
    if missing_path:
        raise SystemExit(f"Pathway columns missing: {missing_path}")
    if "mkt_implied_prob" not in features.columns:
        from src.market_features import attach_market_features

        features = attach_market_features(features)
    return features


def _keep_decision(base: dict, challenger: dict) -> tuple[bool, str]:
    d_auc = float(challenger["auc"] - base["auc"])
    if d_auc >= KEEP_AUC_DELTA:
        return True, f"auc_improved_by_{d_auc:+.4f}_ge_{KEEP_AUC_DELTA}"

    # Flat-edge ROI / maxDD tradeoff (prefer higher ROI and not worse DD)
    b_roi, c_roi = base.get("flat_edge_roi"), challenger.get("flat_edge_roi")
    b_dd, c_dd = base.get("max_dd"), challenger.get("max_dd")
    if (
        b_roi is not None
        and c_roi is not None
        and not np.isnan(b_roi)
        and not np.isnan(c_roi)
        and float(challenger.get("n_bets") or 0) >= 20
    ):
        roi_better = float(c_roi) > float(b_roi) + 1e-6
        dd_ok = True
        if (
            b_dd is not None
            and c_dd is not None
            and not np.isnan(b_dd)
            and not np.isnan(c_dd)
        ):
            # max_dd is negative; "not worse" means c_dd >= b_dd - small tol
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


def _write_md(report: dict[str, Any], path: Path) -> None:
    arms = report["arms"]
    lines = [
        "# Pathway + Market A/B — 2025 holdout",
        "",
        f"Fights: **{int(report.get('n_2025', 0))}** · "
        f"with odds: **{int(report.get('n_with_odds', 0))}** · "
        f"interactions frozen · HV baseline on",
        "",
        "## Side-by-side",
        "",
        "| Arm | n_feat | Acc | AUC | ROI@3% | bets | hit | maxDD | "
        "ROI@5% | ROI@8% | wide% | wide miss | narrow miss |",
        "|-----|-------:|----:|----:|-------:|-----:|----:|------:|"
        "-------:|-------:|------:|----------:|------------:|",
    ]
    order = ["BASE", "PATH", "PATH+MKT", "CAL"]
    for name in order:
        a = arms.get(name)
        if not a:
            continue
        lines.append(
            f"| {name} | {int(a['n_features'])} | {_fmt(a['accuracy'])} | "
            f"{_fmt(a['auc'])} | {_fmt(a['flat_edge_roi'])} | "
            f"{int(a['n_bets'])} | {_fmt(a['hit_rate'], 3)} | {_fmt(a['max_dd'])} | "
            f"{_fmt(a['edge_5_roi'])} | {_fmt(a['edge_8_roi'])} | "
            f"{_fmt(a['wide_ci_rate'], 3)} | {_fmt(a['wide_ci_miss_rate'], 3)} | "
            f"{_fmt(a['narrow_ci_miss_rate'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Coverage (2025)",
            "",
            f"- Pathway mean non-null: **{_fmt(report.get('pathway_coverage_mean_pct'), 1)}%**",
            f"- mkt_implied_prob: **{_fmt(report.get('market_coverage', {}).get('mkt_implied_prob'), 1)}%**",
            f"- line_move: **{_fmt(report.get('market_coverage', {}).get('line_move'), 1)}%** "
            "(0% expected — no opening odds in current UFC sources)",
            "",
            "## Recommendation",
            "",
            f"**{report['recommendation']['decision'].upper()}** — "
            f"{report['recommendation']['reason']}",
            "",
            f"- Keep PATH: `{report['recommendation']['keep_path']}`",
            f"- Keep PATH+MKT: `{report['recommendation']['keep_path_mkt']}`",
            f"- Keep CAL (research): `{report['recommendation']['keep_cal']}`",
            "",
            "Keep rule: AUC ≥ +0.005 vs BASE, or flat-edge ROI/maxDD clearly better "
            "(≥20 bets).",
            "",
            "Flags stay UFC-scoped defaults OFF unless keep passes "
            "(`ENABLE_PATHWAY_FEATURES` / `ENABLE_MARKET_FEATURES`).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_data_dirs()
    features = _load_or_build_features()
    path_cov = log_pathway_coverage(features, year=YEAR, label="ab matrix")
    mkt_cov = log_market_coverage(features, year=YEAR, label="ab matrix")

    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    test = features.loc[dts.dt.year == YEAR]
    n_odds = 0
    if {"f1_odds", "f2_odds"}.issubset(test.columns):
        n_odds = int(
            (test["f1_odds"].notna() & test["f2_odds"].notna()).sum()
        )

    logger.info("Running BASE…")
    base = run_arm(features, arm="BASE", pathway=False, market=False)
    logger.info("BASE: auc=%s acc=%s roi=%s", base["auc"], base["accuracy"], base["flat_edge_roi"])

    logger.info("Running PATH…")
    path = run_arm(features, arm="PATH", pathway=True, market=False)
    logger.info("PATH: auc=%s acc=%s roi=%s", path["auc"], path["accuracy"], path["flat_edge_roi"])

    logger.info("Running PATH+MKT…")
    path_mkt = run_arm(features, arm="PATH+MKT", pathway=True, market=True)
    logger.info(
        "PATH+MKT: auc=%s acc=%s roi=%s",
        path_mkt["auc"],
        path_mkt["accuracy"],
        path_mkt["flat_edge_roi"],
    )

    logger.info("Running CAL (research shrink)…")
    cal = run_arm(
        features, arm="CAL", pathway=True, market=True, apply_cal=True
    )
    logger.info("CAL: auc=%s acc=%s roi=%s", cal["auc"], cal["accuracy"], cal["flat_edge_roi"])

    keep_path, reason_path = _keep_decision(base, path)
    keep_pm, reason_pm = _keep_decision(base, path_mkt)
    keep_cal, reason_cal = _keep_decision(base, cal)

    if keep_pm:
        decision = "keep_path_mkt"
        reason = reason_pm
        flag_path, flag_mkt = True, True
    elif keep_path:
        decision = "keep_path"
        reason = reason_path
        flag_path, flag_mkt = True, False
    else:
        decision = "drop"
        reason = f"path={reason_path}; path_mkt={reason_pm}; cal={reason_cal}"
        flag_path, flag_mkt = False, False

    report = {
        "year": YEAR,
        "n_2025": int(len(test)),
        "n_with_odds": n_odds,
        "wide_ci_threshold": WIDE_CI_THRESHOLD,
        "keep_auc_delta": KEEP_AUC_DELTA,
        "interactions_frozen": True,
        "arms": {
            "BASE": base,
            "PATH": path,
            "PATH+MKT": path_mkt,
            "CAL": cal,
        },
        "deltas_vs_base": {
            "PATH": {
                "accuracy": path["accuracy"] - base["accuracy"],
                "auc": path["auc"] - base["auc"],
                "flat_edge_roi": (
                    path["flat_edge_roi"] - base["flat_edge_roi"]
                    if not (
                        np.isnan(path["flat_edge_roi"])
                        or np.isnan(base["flat_edge_roi"])
                    )
                    else float("nan")
                ),
            },
            "PATH+MKT": {
                "accuracy": path_mkt["accuracy"] - base["accuracy"],
                "auc": path_mkt["auc"] - base["auc"],
                "flat_edge_roi": (
                    path_mkt["flat_edge_roi"] - base["flat_edge_roi"]
                    if not (
                        np.isnan(path_mkt["flat_edge_roi"])
                        or np.isnan(base["flat_edge_roi"])
                    )
                    else float("nan")
                ),
            },
            "CAL": {
                "accuracy": cal["accuracy"] - base["accuracy"],
                "auc": cal["auc"] - base["auc"],
                "flat_edge_roi": (
                    cal["flat_edge_roi"] - base["flat_edge_roi"]
                    if not (
                        np.isnan(cal["flat_edge_roi"])
                        or np.isnan(base["flat_edge_roi"])
                    )
                    else float("nan")
                ),
            },
        },
        "pathway_coverage": path_cov,
        "pathway_coverage_mean_pct": _nanmean(list(path_cov.values())),
        "market_coverage": mkt_cov,
        "recommendation": {
            "decision": decision,
            "reason": reason,
            "keep_path": keep_path,
            "keep_path_mkt": keep_pm,
            "keep_cal": keep_cal,
            "ENABLE_PATHWAY_FEATURES": flag_path,
            "ENABLE_MARKET_FEATURES": flag_mkt,
            "ENABLE_PATHWAY_MARKET_CAL": False,  # research only; never auto-enable
        },
    }

    out_dir = config.DATA_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pathway_market_ab_2025.json"
    md_path = out_dir / "pathway_market_ab_2025.md"

    def _json_default(o: Any) -> Any:
        if isinstance(o, (np.floating, float)):
            f = float(o)
            return None if np.isnan(f) else f
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    json_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    _write_md(report, md_path)
    flag_path_file = out_dir / "pathway_market_flag.txt"
    flag_path_file.write_text(
        f"ENABLE_PATHWAY_FEATURES={'true' if flag_path else 'false'}  # {decision}\n"
        f"ENABLE_MARKET_FEATURES={'true' if flag_mkt else 'false'}\n"
        f"ENABLE_PATHWAY_MARKET_CAL=false  # research only\n",
        encoding="utf-8",
    )
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    logger.info(
        "DECISION %s | PATH keep=%s (%s) | PATH+MKT keep=%s (%s)",
        decision,
        keep_path,
        reason_path,
        keep_pm,
        reason_pm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
