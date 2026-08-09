"""2025 holdout: calibrate conformal CI width vs accuracy / ROI / SKIP rates.

Uses existing AB BASE model (HV, pathway/market OFF) — does NOT retrain
ensemble_winner.joblib or flip ENABLE_PATHWAY_FEATURES / ENABLE_MARKET_FEATURES.
Does NOT change Live HA thresholds.

Reports under data/reports/ only.
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

from src.ensemble import EnsembleClassifier, ensemble_disagreement, prediction_interval
from src.feature_engineering import apply_imputer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ci_cal")

YEAR = 2025
# Overlay display badge threshold (fight_context.py)
DISPLAY_WIDE = 0.40
EDGE_MIN = 0.03

REPORTS_DIR = config.DATA_DIR / "reports"
REPORT_MD = REPORTS_DIR / "ci_width_calibration_2025.md"
REPORT_JSON = REPORTS_DIR / "ci_width_calibration_2025.json"


def _flat_edge_pnl(
    y: np.ndarray,
    proba: np.ndarray,
    f1_odds: np.ndarray | None,
    f2_odds: np.ndarray | None,
    *,
    mask: np.ndarray | None = None,
    edge_min: float = EDGE_MIN,
) -> dict[str, float]:
    if f1_odds is None or f2_odds is None:
        return {"n_bets": 0.0, "roi": float("nan"), "hit_rate": float("nan")}
    pnls: list[float] = []
    hits = 0
    for i, (yt, p, o1, o2) in enumerate(zip(y, proba, f1_odds, f2_odds)):
        if mask is not None and not bool(mask[i]):
            continue
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
            pnls.append((o1f - 1.0) if won else -1.0)
            hits += int(won)
        elif edge2 >= edge_min:
            won = int(yt) == 0
            pnls.append((o2f - 1.0) if won else -1.0)
            hits += int(won)
    n = len(pnls)
    return {
        "n_bets": float(n),
        "roi": (sum(pnls) / n) if n else float("nan"),
        "hit_rate": (hits / n) if n else float("nan"),
    }


def _bucket_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    mask: np.ndarray,
    *,
    f1_odds: np.ndarray | None,
    f2_odds: np.ndarray | None,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss

    n = int(mask.sum())
    if n == 0:
        return {
            "n": 0.0,
            "share": 0.0,
            "accuracy": float("nan"),
            "brier": float("nan"),
            "mean_proba_margin": float("nan"),
            **{f"edge_{k}": float("nan") for k in ("n_bets", "roi", "hit_rate")},
        }
    yt = y[mask]
    p = proba[mask]
    pred = (p >= 0.5).astype(int)
    edge = _flat_edge_pnl(y, proba, f1_odds, f2_odds, mask=mask)
    return {
        "n": float(n),
        "share": float(n / len(y)),
        "accuracy": float(accuracy_score(yt, pred)),
        "brier": float(brier_score_loss(yt, p)),
        "mean_proba_margin": float(np.mean(np.abs(p - 0.5))),
        "edge_n_bets": edge["n_bets"],
        "edge_roi": edge["roi"],
        "edge_hit_rate": edge["hit_rate"],
    }


def _tertile_masks(values: np.ndarray) -> dict[str, np.ndarray]:
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    out: dict[str, np.ndarray] = {
        "low": np.zeros(len(v), dtype=bool),
        "mid": np.zeros(len(v), dtype=bool),
        "high": np.zeros(len(v), dtype=bool),
    }
    if finite.sum() < 3:
        out["mid"] = finite
        return out
    q1, q2 = np.quantile(v[finite], [1 / 3, 2 / 3])
    out["low"] = finite & (v <= q1)
    out["high"] = finite & (v > q2)
    out["mid"] = finite & ~out["low"] & ~out["high"]
    return out


def _gate_counts(
    widths: np.ndarray,
    disagrees: np.ndarray,
    *,
    settings: dict[str, Any],
) -> dict[str, float]:
    from src.uncertainty_gates import _evaluate_uncertainty_gate_core

    actions = {"allow": 0, "tighten": 0, "skip": 0}
    skip_wide = 0
    for w, d in zip(widths, disagrees):
        # Core gate only (no Paper wide override) for threshold study
        g = _evaluate_uncertainty_gate_core(
            disagreement=float(d),
            interval_width=float(w),
            settings=settings,
        )
        actions[g.action] = actions.get(g.action, 0) + 1
        if "wide_interval" in (g.reasons or []):
            skip_wide += 1
    n = max(len(widths), 1)
    return {
        "allow_rate": actions["allow"] / n,
        "tighten_rate": actions["tighten"] / n,
        "skip_rate": actions["skip"] / n,
        "skip_wide_rate": skip_wide / n,
        "n": float(len(widths)),
    }


def _load_features() -> pd.DataFrame:
    cache = config.PROCESSED_DIR / "ab_feature_matrix_pathway_v2.parquet"
    if not cache.is_file():
        cache = config.PROCESSED_DIR / "ab_feature_matrix_v5.parquet"
    if not cache.is_file():
        raise SystemExit(f"Missing feature cache under {config.PROCESSED_DIR}")
    logger.info("Loading %s", cache)
    return pd.read_parquet(cache)


def _score_holdout(features: pd.DataFrame) -> dict[str, Any]:
    model_path = config.MODELS_DIR / "ab_pathway_market" / "base.joblib"
    if not model_path.is_file():
        raise SystemExit(f"Missing {model_path} — run ab_pathway_market_2025 BASE first")

    artifact = joblib.load(model_path)
    feat_cols = [c for c in (artifact.get("feature_columns") or []) if c in features.columns]
    conformal_q = float(artifact.get("conformal_q") or 0.0)

    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    test = features.loc[dts.dt.year == YEAR].copy()
    if test.empty:
        raise SystemExit(f"No {YEAR} rows in feature matrix")

    test_x = apply_imputer(test, artifact["imputer"]) if artifact.get("imputer") else test
    X = test_x[feat_cols]
    y = test[config.TARGET_COLUMN].astype(int).to_numpy()
    clf = artifact["model"]
    proba = np.asarray(clf.predict_proba(X)[:, 1], dtype=float)
    _, _, ci_width = prediction_interval(proba, conformal_q=conformal_q)

    disagree = np.zeros(len(proba))
    if isinstance(clf, EnsembleClassifier):
        comps = clf.predict_proba_components(X)
        disagree = ensemble_disagreement(comps)

    f1o = test["f1_odds"].to_numpy() if "f1_odds" in test.columns else None
    f2o = test["f2_odds"].to_numpy() if "f2_odds" in test.columns else None

    margin = np.abs(proba - 0.5)
    return {
        "y": y,
        "proba": proba,
        "ci_width": np.asarray(ci_width, dtype=float),
        "disagreement": np.asarray(disagree, dtype=float),
        "margin": margin,
        "f1_odds": f1o,
        "f2_odds": f2o,
        "conformal_q": conformal_q,
        "n_features": len(feat_cols),
        "model_path": str(model_path),
        "n": len(y),
    }


def run() -> dict[str, Any]:
    features = _load_features()
    scored = _score_holdout(features)
    y = scored["y"]
    proba = scored["proba"]
    width = scored["ci_width"]
    disagree = scored["disagreement"]
    margin = scored["margin"]
    f1o, f2o = scored["f1_odds"], scored["f2_odds"]

    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    overall = {
        "n": float(scored["n"]),
        "accuracy": float(accuracy_score(y, (proba >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "conformal_q": scored["conformal_q"],
        "width_min": float(np.min(width)),
        "width_max": float(np.max(width)),
        "width_mean": float(np.mean(width)),
        "width_std": float(np.std(width)),
        "corr_width_vs_margin": float(np.corrcoef(width, margin)[0, 1]),
        "corr_width_vs_disagree": float(np.corrcoef(width, disagree)[0, 1])
        if np.std(disagree) > 1e-12
        else float("nan"),
        "model_path": scored["model_path"],
        "n_features": float(scored["n_features"]),
    }
    overall.update(
        {
            f"all_{k}": v
            for k, v in _flat_edge_pnl(y, proba, f1o, f2o).items()
        }
    )

    # Fixed thresholds used in product
    thresholds = {
        "live_skip_default": float(config.LIVE_INTERVAL_WIDTH_SKIP),
        "live_tighten": float(config.LIVE_INTERVAL_WIDTH_TIGHTEN),
        "display_wide": DISPLAY_WIDE,
        "paper_tighten": float(config.PAPER_INTERVAL_WIDTH_TIGHTEN),
        "paper_skip": float(config.PAPER_INTERVAL_WIDTH_SKIP),
    }
    width_threshold_table: dict[str, Any] = {}
    for name, thr in thresholds.items():
        wide = width >= thr
        narrow = ~wide
        width_threshold_table[name] = {
            "threshold": thr,
            "wide_share": float(wide.mean()),
            "wide": _bucket_metrics(y, proba, wide, f1_odds=f1o, f2_odds=f2o),
            "narrow": _bucket_metrics(y, proba, narrow, f1_odds=f1o, f2_odds=f2o),
            # Counterfactual: bet only narrow (skip wide)
            "bet_narrow_only": _flat_edge_pnl(y, proba, f1o, f2o, mask=narrow),
            "bet_wide_only": _flat_edge_pnl(y, proba, f1o, f2o, mask=wide),
        }

    tertiles: dict[str, Any] = {}
    for label, arr in (
        ("ci_width", width),
        ("disagreement", disagree),
        ("proba_margin", margin),
    ):
        masks = _tertile_masks(arr)
        tertiles[label] = {
            "cuts": {
                "p33": float(np.quantile(arr, 1 / 3)),
                "p66": float(np.quantile(arr, 2 / 3)),
            },
            "buckets": {
                b: _bucket_metrics(y, proba, m, f1_odds=f1o, f2_odds=f2o)
                for b, m in masks.items()
            },
        }

    paper_settings = {
        "enabled": True,
        "disagreement_skip": float(config.PAPER_DISAGREEMENT_SKIP),
        "disagreement_tighten": float(config.PAPER_DISAGREEMENT_TIGHTEN),
        "interval_width_skip": float(config.PAPER_INTERVAL_WIDTH_SKIP),
        "interval_width_tighten": float(config.PAPER_INTERVAL_WIDTH_TIGHTEN),
        "edge_bump": float(config.PAPER_UNCERTAINTY_EDGE_BUMP),
        "kelly_mult": float(config.PAPER_UNCERTAINTY_KELLY_MULT),
    }
    live_settings = {
        "enabled": True,
        "disagreement_skip": float(config.LIVE_DISAGREEMENT_SKIP),
        "disagreement_tighten": float(config.LIVE_DISAGREEMENT_TIGHTEN),
        "interval_width_skip": float(config.LIVE_INTERVAL_WIDTH_SKIP),
        "interval_width_tighten": float(config.LIVE_INTERVAL_WIDTH_TIGHTEN),
        "edge_bump": float(config.LIVE_UNCERTAINTY_EDGE_BUMP),
        "kelly_mult": float(config.LIVE_UNCERTAINTY_KELLY_MULT),
    }
    # Width-only gates (disagreement forced low) to isolate CI signal
    width_only_paper = dict(paper_settings)
    width_only_live = dict(live_settings)
    gate_summary = {
        "paper_full": _gate_counts(width, disagree, settings=paper_settings),
        "live_full": _gate_counts(width, disagree, settings=live_settings),
        "paper_width_only": _gate_counts(
            width, np.zeros_like(disagree), settings=width_only_paper
        ),
        "live_width_only": _gate_counts(
            width, np.zeros_like(disagree), settings=width_only_live
        ),
    }

    # Verdict helpers
    disp = width_threshold_table["display_wide"]
    paper = width_threshold_table["paper_skip"]
    note_parts = []
    if overall["width_min"] >= DISPLAY_WIDE - 1e-9:
        note_parts.append(
            f"All widths ≥ {overall['width_min']:.3f} (conformal_q={overall['conformal_q']:.3f}); "
            f"display badge @ {DISPLAY_WIDE} labels every fight 'wide'."
        )
    if abs(overall["corr_width_vs_margin"]) > 0.85:
        note_parts.append(
            f"|corr(width, |p-0.5|)|={abs(overall['corr_width_vs_margin']):.3f} — "
            "width mostly mirrors confidence (boundary clipping of ±q)."
        )
    # Does skipping wide help ROI at paper threshold?
    all_roi = overall.get("all_roi")
    narrow_roi = paper["bet_narrow_only"].get("roi")
    if (
        all_roi is not None
        and narrow_roi is not None
        and not np.isnan(all_roi)
        and not np.isnan(narrow_roi)
        and paper["bet_narrow_only"]["n_bets"] >= 10
    ):
        if narrow_roi > all_roi + 0.01:
            note_parts.append(
                f"Skipping paper-wide improves flat-edge ROI "
                f"({all_roi:+.3f} → {narrow_roi:+.3f})."
            )
        elif narrow_roi < all_roi - 0.01:
            note_parts.append(
                f"Skipping paper-wide hurts flat-edge ROI "
                f"({all_roi:+.3f} → {narrow_roi:+.3f})."
            )
        else:
            note_parts.append("Skipping paper-wide ≈ flat ROI vs betting all edges.")

    d_tert = tertiles["disagreement"]["buckets"]
    if (
        d_tert["low"]["n"] > 0
        and d_tert["high"]["n"] > 0
        and not np.isnan(d_tert["low"]["accuracy"])
        and not np.isnan(d_tert["high"]["accuracy"])
    ):
        note_parts.append(
            f"Disagreement tertiles acc: low={d_tert['low']['accuracy']:.3f} "
            f"vs high={d_tert['high']['accuracy']:.3f}."
        )

    result = {
        "year": YEAR,
        "overall": overall,
        "thresholds": thresholds,
        "by_width_threshold": width_threshold_table,
        "tertiles": tertiles,
        "gates": gate_summary,
        "notes": note_parts,
        "isolation": {
            "retrained_ensemble_winner": False,
            "enable_pathway_features": False,
            "enable_market_features": False,
            "live_ha_thresholds_changed": False,
        },
    }
    return result


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _fmt_f(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def write_report(result: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    o = result["overall"]
    lines = [
        f"# CI width calibration ({YEAR})",
        "",
        "UFC-only research. No retrain of `ensemble_winner.joblib`. "
        "Pathway/market flags left off. Live HA thresholds unchanged.",
        "",
        f"Model: `{o['model_path']}` (AB BASE / HV)",
        f"Holdout n={int(o['n'])}  acc={_fmt_f(o['accuracy'])}  "
        f"AUC={_fmt_f(o['auc'], 4)}  Brier={_fmt_f(o['brier'], 4)}",
        f"conformal_q={_fmt_f(o['conformal_q'], 4)}  "
        f"width mean/min/max={_fmt_f(o['width_mean'])}/"
        f"{_fmt_f(o['width_min'])}/{_fmt_f(o['width_max'])}",
        f"corr(width, |p-0.5|)={_fmt_f(o['corr_width_vs_margin'], 3)}  "
        f"corr(width, disagreement)={_fmt_f(o.get('corr_width_vs_disagree'), 3)}",
        f"Flat edge≥{EDGE_MIN:.0%}: n={int(o.get('all_n_bets') or 0)}  "
        f"ROI={_fmt_f(o.get('all_roi'))}  hit={_fmt_pct(o.get('all_hit_rate'))}",
        "",
        "## Notes",
        "",
    ]
    for n in result.get("notes") or ["(none)"]:
        lines.append(f"- {n}")

    lines += ["", "## Width thresholds (narrow vs wide)", ""]
    lines.append(
        "| Threshold | thr | wide% | narrow acc | wide acc | "
        "bet-narrow ROI (n) | bet-wide ROI (n) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, block in result["by_width_threshold"].items():
        lines.append(
            f"| {name} | {_fmt_f(block['threshold'], 2)} | "
            f"{_fmt_pct(block['wide_share'])} | "
            f"{_fmt_f(block['narrow']['accuracy'])} | "
            f"{_fmt_f(block['wide']['accuracy'])} | "
            f"{_fmt_f(block['bet_narrow_only']['roi'])} "
            f"({int(block['bet_narrow_only']['n_bets'])}) | "
            f"{_fmt_f(block['bet_wide_only']['roi'])} "
            f"({int(block['bet_wide_only']['n_bets'])}) |"
        )

    lines += ["", "## Tertiles", ""]
    for label, block in result["tertiles"].items():
        lines.append(
            f"### {label} (p33={_fmt_f(block['cuts']['p33'])}, "
            f"p66={_fmt_f(block['cuts']['p66'])})"
        )
        lines.append("| bucket | n | acc | Brier | margin | edge ROI (n) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for bname, b in block["buckets"].items():
            lines.append(
                f"| {bname} | {int(b['n'])} | {_fmt_f(b['accuracy'])} | "
                f"{_fmt_f(b['brier'], 4)} | {_fmt_f(b['mean_proba_margin'])} | "
                f"{_fmt_f(b['edge_roi'])} ({int(b['edge_n_bets'])}) |"
            )
        lines.append("")

    lines += ["## Gate action rates (core, no Paper wide override)", ""]
    lines.append("| profile | allow | tighten | skip | skip∶wide_interval |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, g in result["gates"].items():
        lines.append(
            f"| {name} | {_fmt_pct(g['allow_rate'])} | {_fmt_pct(g['tighten_rate'])} | "
            f"{_fmt_pct(g['skip_rate'])} | {_fmt_pct(g['skip_wide_rate'])} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Conformal intervals are `clip(p±q)`. When `q` is large (~0.5), "
        "width is high for almost all fights and mainly tracks how extreme `p` is. "
        "Prefer ensemble disagreement / proba margin for SKIP diagnostics if width "
        "does not separate accuracy.",
        "",
        f"Artifacts: `{REPORT_JSON.name}`, `{REPORT_MD.name}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s and %s", REPORT_MD, REPORT_JSON)


def main() -> int:
    result = run()
    write_report(result)
    print(REPORT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
