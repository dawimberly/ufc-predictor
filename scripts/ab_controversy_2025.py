"""2025 controversy backtest: messy fights / refs vs noise.

Does NOT retrain ensemble_winner. Uses cached HV BASE artifact when present,
otherwise trains one BASE arm (pathway/market off). Compares:

  BASE     — full 2025 holdout
  EXCL_M   — drop controversial methods (split/maj/DQ/doctor/overturn/CNC)
  EXCL_W   — drop watchlist-ref bouts (if referee join hits)
  CONTRO   — controversial-method subset only (is error above noise?)

Keep rule: exclude arm beats BASE by AUC≥+0.005 OR flat-edge ROI clearly better
with enough bets; else DROP (noise). No FEATURE_COLUMNS change.
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

import numpy as np
import pandas as pd

import config

config.refresh_runtime_env()

from src.controversy import (
    attach_referee_and_controversy,
    build_and_save_catalog,
)
from src.feature_engineering import apply_imputer
from src.model_trainer import train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ab_controversy")

YEAR = 2025
KEEP_AUC_DELTA = 0.005
EDGE_MIN = 0.03
BOOT_N = 400
NOISE_Z = 1.96  # ~95%

REPORTS = config.DATA_DIR / "reports"
REPORT_MD = REPORTS / "controversy_ab_2025.md"
REPORT_JSON = REPORTS / "controversy_ab_2025.json"
FLAG_TXT = REPORTS / "controversy_flag.txt"
BASE_MODEL = config.MODELS_DIR / "ab_decision_profile" / "base.joblib"


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


def _metrics(y, proba, *, f1_odds, f2_odds) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    pred = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")
    out = {
        "n": float(len(y)),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "auc": auc,
        "brier": float(brier_score_loss(y, proba)) if len(y) else float("nan"),
    }
    out.update(_flat_edge(y, proba, f1_odds, f2_odds))
    return out


def _binom_z(hits: int, n: int, p0: float) -> float:
    if n <= 0:
        return float("nan")
    phat = hits / n
    se = np.sqrt(max(p0 * (1.0 - p0), 1e-9) / n)
    return float((phat - p0) / se)


def _bootstrap_delta_auc(
    y: np.ndarray,
    proba: np.ndarray,
    mask_keep: np.ndarray,
    *,
    n_boot: int = BOOT_N,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap ΔAUC (kept subset − full)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 30 or int(mask_keep.sum()) < 30:
        return {"delta_auc_mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    def _auc(yy, pp):
        try:
            return float(roc_auc_score(yy, pp))
        except ValueError:
            return float("nan")

    base = _auc(y, proba)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, pp = y[idx], proba[idx]
        mk = mask_keep[idx]
        if mk.sum() < 20 or (~mk).sum() < 5:
            continue
        a_full = _auc(yy, pp)
        a_keep = _auc(yy[mk], pp[mk])
        if np.isfinite(a_full) and np.isfinite(a_keep):
            deltas.append(a_keep - a_full)
    if not deltas:
        return {"delta_auc_mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "base_auc": base}
    arr = np.asarray(deltas, dtype=float)
    return {
        "delta_auc_mean": float(arr.mean()),
        "ci_lo": float(np.quantile(arr, 0.025)),
        "ci_hi": float(np.quantile(arr, 0.975)),
        "base_auc": base,
        "n_boot_ok": float(len(arr)),
    }


def _load_features() -> pd.DataFrame:
    path = config.DATA_DIR / "processed" / "ab_feature_matrix_pathway_v2.parquet"
    if not path.is_file():
        path = config.DATA_DIR / "processed" / "ab_feature_matrix_v5.parquet"
    logger.info("Loading %s", path)
    feats = pd.read_parquet(path)
    feats = attach_referee_and_controversy(feats)
    return feats


def _feature_cols(features: pd.DataFrame) -> list[str]:
    path = set(getattr(config, "PATHWAY_FEATURE_COLUMNS", []) or [])
    mkt = set(getattr(config, "MARKET_FEATURE_COLUMNS", []) or [])
    return [
        c
        for c in config.FEATURE_COLUMNS
        if c in features.columns and c not in path and c not in mkt
    ]


def _predict_holdout(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Score 2025 with existing BASE artifact or train once."""
    import joblib

    cols = _feature_cols(features)
    dts = pd.to_datetime(features[config.DATE_COLUMN], errors="coerce")
    train = features.loc[dts.dt.year < YEAR].copy()
    test = features.loc[dts.dt.year == YEAR].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"Need pre-{YEAR} train and {YEAR} test")

    saved = list(config.FEATURE_COLUMNS)
    config.FEATURE_COLUMNS = cols
    try:
        if BASE_MODEL.is_file():
            logger.info("Using cached BASE model %s", BASE_MODEL)
            bundle = joblib.load(BASE_MODEL)
            model = bundle.get("model") if isinstance(bundle, dict) else bundle
            feat_cols = list(bundle.get("feature_columns") or cols) if isinstance(bundle, dict) else cols
            imputer = bundle.get("imputer") if isinstance(bundle, dict) else None
            X = test.reindex(columns=feat_cols)
            if imputer is not None:
                X = apply_imputer(X, imputer)
            else:
                X = X.apply(pd.to_numeric, errors="coerce")
            proba = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
            test = test.copy()
            test["prob_f1_win"] = proba
            return test, feat_cols

        logger.info("Training BASE arm (no cached artifact)")
        arm_dir = config.MODELS_DIR / "ab_controversy"
        arm_dir.mkdir(parents=True, exist_ok=True)
        result = train_model(
            train,
            model_path=arm_dir / "base.joblib",
            tune="none",
            run_backtest_hook=False,
        )
        model = result.model
        imputer = getattr(result, "imputer", None)
        X = test.reindex(columns=cols)
        if imputer is not None:
            X = apply_imputer(X, imputer)
        else:
            X = X.apply(pd.to_numeric, errors="coerce")
        test = test.copy()
        test["prob_f1_win"] = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
        return test, cols
    finally:
        config.FEATURE_COLUMNS = saved


def _slice_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
    p = pd.to_numeric(df["prob_f1_win"], errors="coerce")
    m = y.notna() & p.notna()
    yv = y[m].to_numpy(dtype=float)
    pv = p[m].to_numpy(dtype=float)
    o1 = df.loc[m, "f1_odds"].to_numpy() if "f1_odds" in df.columns else None
    o2 = df.loc[m, "f2_odds"].to_numpy() if "f2_odds" in df.columns else None
    return _metrics(yv, pv, f1_odds=o1, f2_odds=o2)


def _fmt(x: Any, pct: bool = False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.4f}"


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    catalog = build_and_save_catalog(min_bouts=40)
    features = _load_features()
    test, feat_cols = _predict_holdout(features)

    cont = test["is_controversial_fight"].fillna(False).astype(bool)
    watch_ref = test["ref_flagged"].fillna(False).astype(bool)
    # Watchlist refs from catalog names (attach only sets elevated flags; expand)
    watch_names = {
        str(x).casefold()
        for x in (catalog.get("watchlist_referees") or [])
    }
    if watch_names and "referee" in test.columns:
        from src.data_loader import clean_fighter_name

        watch_ref = test["referee"].fillna("").map(
            lambda r: clean_fighter_name(r).casefold() in watch_names if r else False
        )

    arms = {
        "BASE": test,
        "EXCL_METHOD": test.loc[~cont].copy(),
        "EXCL_WATCH_REF": test.loc[~watch_ref].copy(),
        "EXCL_METHOD_OR_WATCH": test.loc[~(cont | watch_ref)].copy(),
        "CONTRO_ONLY": test.loc[cont].copy(),
    }

    metrics = {name: _slice_metrics(df) for name, df in arms.items()}

    # Noise test: is CONTRO accuracy worse than BASE beyond binomial noise?
    base_acc = metrics["BASE"]["accuracy"]
    contro = arms["CONTRO_ONLY"]
    y_c = pd.to_numeric(contro[config.TARGET_COLUMN], errors="coerce")
    p_c = pd.to_numeric(contro["prob_f1_win"], errors="coerce")
    m_c = y_c.notna() & p_c.notna()
    hits_c = int(((p_c[m_c] >= 0.5).astype(int) == y_c[m_c].astype(int)).sum())
    n_c = int(m_c.sum())
    z_contro = _binom_z(hits_c, n_c, float(base_acc))

    y_all = pd.to_numeric(test[config.TARGET_COLUMN], errors="coerce")
    p_all = pd.to_numeric(test["prob_f1_win"], errors="coerce")
    m_all = (y_all.notna() & p_all.notna()).to_numpy()
    yv = y_all[m_all].to_numpy(dtype=float)
    pv = p_all[m_all].to_numpy(dtype=float)
    keep_method = (~cont.to_numpy())[m_all]
    boot = _bootstrap_delta_auc(yv, pv, keep_method)

    base = metrics["BASE"]
    excl = metrics["EXCL_METHOD"]
    d_auc = float(excl["auc"] - base["auc"]) if np.isfinite(excl["auc"]) and np.isfinite(base["auc"]) else float("nan")
    d_roi = float(excl["flat_edge_roi"] - base["flat_edge_roi"]) if (
        np.isfinite(excl.get("flat_edge_roi", np.nan)) and np.isfinite(base.get("flat_edge_roi", np.nan))
    ) else float("nan")

    ci_clear = (
        np.isfinite(boot.get("ci_lo", np.nan))
        and np.isfinite(boot.get("ci_hi", np.nan))
        and float(boot["ci_lo"]) > 0
    )
    keep = bool(
        (np.isfinite(d_auc) and d_auc >= KEEP_AUC_DELTA)
        or ci_clear
        or (np.isfinite(z_contro) and abs(z_contro) >= NOISE_Z and n_c >= 30)
    )
    if keep and np.isfinite(z_contro) and abs(z_contro) >= NOISE_Z and n_c >= 30:
        reason = f"contro_acc_z={z_contro:+.2f}_n={n_c}"
    elif keep and ci_clear:
        reason = f"boot_delta_auc_ci=({boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f})"
    elif keep:
        reason = f"excl_method_d_auc={d_auc:+.4f}"
    else:
        reason = (
            f"noise_d_auc={d_auc:+.4f}_boot_ci="
            f"({_fmt(boot.get('ci_lo'))},{_fmt(boot.get('ci_hi'))})"
            f"_contro_z={_fmt(z_contro)}_n_contro={n_c}"
        )

    decision = "keep_filter" if keep else "drop_noise"

    # Kind breakdown 2025
    kind_counts: dict[str, int] = {}
    for s in test.loc[cont, "controversy_kinds"].fillna(""):
        for k in str(s).split("|"):
            if k:
                kind_counts[k] = kind_counts.get(k, 0) + 1

    report = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "year": YEAR,
        "decision": decision,
        "reason": reason,
        "keep_auc_delta": KEEP_AUC_DELTA,
        "catalog": {
            "league_rate": catalog.get("league_controversy_rate"),
            "flagged_referees": catalog.get("flagged_referees"),
            "watchlist_referees": catalog.get("watchlist_referees"),
            "n_flagged_refs": len(catalog.get("flagged_referees") or []),
            "n_watchlist_refs": len(catalog.get("watchlist_referees") or []),
        },
        "2025_controversy": {
            "n_controversial": int(cont.sum()),
            "n_watch_ref": int(watch_ref.sum()),
            "kind_counts": kind_counts,
            "ref_join_rate": float(test["referee"].fillna("").ne("").mean())
            if "referee" in test.columns
            else 0.0,
        },
        "arms": metrics,
        "delta_excl_method": {"auc": d_auc, "flat_edge_roi": d_roi},
        "bootstrap_excl_method": boot,
        "contro_vs_base_acc": {
            "base_acc": base_acc,
            "contro_acc": metrics["CONTRO_ONLY"]["accuracy"],
            "n": n_c,
            "hits": hits_c,
            "z": z_contro,
            "above_noise": bool(np.isfinite(z_contro) and abs(z_contro) >= NOISE_Z and n_c >= 30),
        },
        "feature_fingerprint": _fingerprint_cols(feat_cols),
        "n_features": len(feat_cols),
        "notes": (
            "No 2σ elevated refs in Greco history — watchlist is 1σ only. "
            "Exclude filters are evaluation-only; no FEATURE_COLUMNS / live HA change. "
            "User Sutherland/Montanha flag remains in fighter_flags.json."
        ),
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Controversy A/B — {YEAR}",
        "",
        "UFC-only. Messy methods + referee EB rates vs noise. No ML retrain of production.",
        "",
        f"**Decision: {decision.upper()}** — {reason}",
        "",
        "## Referee screen (Greco history)",
        "",
        f"- League messy-outcome rate: {_fmt(catalog.get('league_controversy_rate'), pct=True)}",
        f"- Flagged refs (2σ): {catalog.get('flagged_referees') or '*(none)*'}",
        f"- Watchlist refs (1σ): {', '.join(catalog.get('watchlist_referees') or []) or '*(none)*'}",
        "",
        f"## {YEAR} controversial fights",
        "",
        f"- n controversial (method): {int(cont.sum())} / {len(test)}",
        f"- n watchlist-ref bouts: {int(watch_ref.sum())}",
        f"- Referee join hit rate: {_fmt(report['2025_controversy']['ref_join_rate'], pct=True)}",
        f"- Kinds: {kind_counts}",
        "",
        "## Side-by-side",
        "",
        "| Arm | n | Acc | AUC | Brier | ROI@3% | bets | hit | maxDD |",
        "|-----|--:|----:|----:|------:|-------:|-----:|----:|------:|",
    ]
    for name in ("BASE", "EXCL_METHOD", "EXCL_WATCH_REF", "EXCL_METHOD_OR_WATCH", "CONTRO_ONLY"):
        m = metrics[name]
        lines.append(
            f"| {name} | {int(m['n'])} | {_fmt(m['accuracy'])} | {_fmt(m['auc'])} | "
            f"{_fmt(m['brier'])} | {_fmt(m['flat_edge_roi'], pct=True)} | "
            f"{int(m['n_bets'])} | {_fmt(m['hit_rate'], pct=True)} | {_fmt(m['max_dd'])} |"
        )
    lines.extend(
        [
            "",
            f"ΔAUC (EXCL_METHOD − BASE) = {_fmt(d_auc)} (keep if ≥ +{KEEP_AUC_DELTA} or boot CI > 0)",
            f"Bootstrap ΔAUC mean={_fmt(boot.get('delta_auc_mean'))} "
            f"CI=[{_fmt(boot.get('ci_lo'))}, {_fmt(boot.get('ci_hi'))}]",
            "",
            "## Contro subset vs noise",
            "",
            f"- BASE acc={_fmt(base_acc, pct=True)} | CONTRO acc={_fmt(metrics['CONTRO_ONLY']['accuracy'], pct=True)} "
            f"(n={n_c}, z={_fmt(z_contro)})",
            f"- Above noise (|z|≥{NOISE_Z} & n≥30): "
            f"{report['contro_vs_base_acc']['above_noise']}",
            "",
            "## Recommendation",
            "",
        ]
    )
    if decision == "keep_filter":
        lines.append(
            "**KEEP filter (research)** — excluding messy fights / acting on contro error clears noise bar. "
            "Still do **not** add ref/method flags to FEATURE_COLUMNS. Optional evaluation filter only."
        )
    else:
        lines.append(
            "**DROP as signal** — controversial methods/refs do not beat noise for model filter. "
            "Keep user integrity flags (Sutherland/Montanha). Watchlist refs are informational only. "
            "No live HA / FEATURE_COLUMNS change."
        )
    lines.extend(
        [
            "",
            report["notes"],
            "",
            f"Artifacts: `{REPORT_JSON.name}`, `referee_controversy_rates.csv`, `controversial_catalog.json`",
            "",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    FLAG_TXT.write_text(
        "# controversy production flag\n"
        f"# {decision}: {reason}\n"
        "ADD_CONTROVERSY_FILTER_TO_FEATURES=false\n"
        "SKIP_CONTROVERSIAL_METHODS_IN_LIVE=false\n",
        encoding="utf-8",
    )

    # research_keep_drop
    kd = REPORTS / "research_keep_drop.md"
    if kd.is_file():
        text = kd.read_text(encoding="utf-8")
        row = (
            f"| Controversy methods / refs (2025) | **{'KEEP filter' if keep else 'DROP (noise)'}** | "
            f"{reason} |"
        )
        if "Controversy methods / refs" not in text:
            text = text.replace(
                "| Decision profile (dec/split/share) |",
                row + "\n| Decision profile (dec/split/share) |",
            )
            # If decision profile row format differs, append before Product priority
            if "Controversy methods / refs" not in text:
                text = text.replace(
                    "\n## Product priority",
                    f"\n{row}\n\n## Product priority",
                )
            kd.write_text(text, encoding="utf-8")

    print(REPORT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
