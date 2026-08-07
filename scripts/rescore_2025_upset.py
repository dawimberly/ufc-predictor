"""Re-score 2025 with current ensemble (disagreement + conformal CI) and run upset autopsy.

No model retrain — uses models/ensemble_winner.joblib.
Walk-forward: per-event imputer fit on past fights; frozen model weights.
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

from src.ensemble import EnsembleClassifier, ensemble_disagreement, prediction_interval
from src.feature_engineering import apply_imputer, apply_interaction_specs, fit_imputer
from src.predictor import FightPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rescore_2025")

DATE_COL = config.DATE_COLUMN
TARGET = config.TARGET_COLUMN
OUT_CSV = config.DATA_DIR / "backtest_2025_results_rescored.csv"
OUT_FULL = config.DATA_DIR / "reports" / "backtest_2025_rescored_full.csv"


def _load_features() -> pd.DataFrame:
    cache = config.PROCESSED_DIR / "ab_feature_matrix_v5.parquet"
    if cache.is_file():
        logger.info("Loading cached features %s", cache)
        df = pd.read_parquet(cache)
    else:
        from src.data_loader import load_processed_features

        logger.info("Loading %s", config.PROCESSED_FEATURES_CSV)
        df = load_processed_features(config.PROCESSED_FEATURES_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET])
    if "event" not in df.columns or df["event"].isna().all():
        if "event_name" in df.columns:
            df["event"] = df["event_name"]
        else:
            df["event"] = "Unknown"
    df["event"] = df["event"].fillna("Unknown").astype(str)
    return df


def walk_forward_score_2025(
    features: pd.DataFrame,
    predictor: FightPredictor,
    *,
    year: int = 2025,
) -> pd.DataFrame:
    """Event walk-forward with uncertainty columns attached."""
    interaction_specs = getattr(predictor, "interaction_specs", None) or []
    feature_cols = list(predictor.feature_columns)
    conformal_q = float(getattr(predictor, "conformal_q", 0.15) or 0.15)

    df = features.copy()
    df["event_key"] = (
        df["event"].astype(str) + "|" + df[DATE_COL].dt.normalize().astype(str)
    )
    year_mask = df[DATE_COL].dt.year == year
    if not year_mask.any():
        raise SystemExit(f"No fights for {year}. Max date={df[DATE_COL].max()}")

    events = (
        df.loc[year_mask, ["event_key", "event", DATE_COL]]
        .drop_duplicates("event_key")
        .sort_values(DATE_COL)
    )
    logger.info("Scoring %s events in %s…", len(events), year)

    rows: list[dict] = []
    model = predictor.model
    for _, ev in events.iterrows():
        ev_date = ev[DATE_COL]
        ev_key = ev["event_key"]
        train = df[df[DATE_COL] < ev_date]
        test = df[(df["event_key"] == ev_key) & year_mask]
        if train.empty or test.empty:
            continue

        train_x = apply_interaction_specs(train, interaction_specs) if interaction_specs else train
        test_x = apply_interaction_specs(test, interaction_specs) if interaction_specs else test
        imputer = fit_imputer(train_x)
        prepared = apply_imputer(test_x, imputer)
        missing = [c for c in feature_cols if c not in prepared.columns]
        if missing:
            raise SystemExit(f"Missing model features: {missing[:8]}…")
        prepared = prepared.dropna(subset=feature_cols)
        if prepared.empty:
            continue

        X = prepared[feature_cols]
        proba = model.predict_proba(X)[:, 1]
        ci_low, ci_high, ci_width = prediction_interval(proba, conformal_q=conformal_q)
        disagree = np.zeros(len(proba))
        prob_lgbm = proba
        prob_xgb = proba
        if isinstance(model, EnsembleClassifier):
            components = model.predict_proba_components(X)
            disagree = ensemble_disagreement(components)
            prob_lgbm = components.get("lgbm", proba)
            prob_xgb = components.get("xgb", proba)

        for i, (_, row) in enumerate(prepared.iterrows()):
            actual = int(row[TARGET])
            p1 = float(proba[i])
            f1 = str(row.get("fighter_1") or "")
            f2 = str(row.get("fighter_2") or "")
            pick = f1 if p1 >= 0.5 else f2
            winner = f1 if actual == 1 else f2
            record = row.to_dict()
            record["prob_f1_win"] = p1
            record["prob_f2_win"] = 1.0 - p1
            record["prob_ci_low"] = float(ci_low[i])
            record["prob_ci_high"] = float(ci_high[i])
            record["interval_width"] = float(ci_width[i])
            record["ensemble_disagreement"] = float(disagree[i])
            record["prob_lgbm"] = float(prob_lgbm[i]) if np.ndim(prob_lgbm) else float(p1)
            record["prob_xgb"] = float(prob_xgb[i]) if np.ndim(prob_xgb) else float(p1)
            record["predicted_winner"] = pick
            record["predicted_prob"] = p1 if p1 >= 0.5 else 1.0 - p1
            record["winner"] = winner
            record["correct"] = int((p1 >= 0.5) == bool(actual))
            record["wf_train_rows"] = int(len(train))
            record["event_name"] = ev["event"]
            rows.append(record)

    out = pd.DataFrame(rows)
    logger.info(
        "Scored %s fights | accuracy=%.3f | mean_width=%.3f | mean_disagree=%.4f",
        len(out),
        float(out["correct"].mean()) if len(out) else 0.0,
        float(out["interval_width"].mean()) if len(out) else 0.0,
        float(out["ensemble_disagreement"].mean()) if len(out) else 0.0,
    )
    return out


def _attach_market_fields(df: pd.DataFrame) -> pd.DataFrame:
    """De-vig implied probs + edges when odds present."""
    from ufc_betting_bot.modules.edge import compute_edge, market_probs

    out = df.copy()
    imp1, imp2, e1, e2, best, side = [], [], [], [], [], []
    for _, row in out.iterrows():
        market = market_probs(row)
        p1 = float(row["prob_f1_win"])
        if market:
            m1, m2 = market
            edges = compute_edge(p1, 1.0 - p1, m1, m2)
            imp1.append(m1)
            imp2.append(m2)
            e1.append(edges.get("edge_f1"))
            e2.append(edges.get("edge_f2"))
            best.append(edges.get("best_edge"))
            side.append(edges.get("bet_side"))
        else:
            imp1.append(np.nan)
            imp2.append(np.nan)
            e1.append(np.nan)
            e2.append(np.nan)
            best.append(np.nan)
            side.append("")
    out["implied_prob_f1"] = imp1
    out["implied_prob_f2"] = imp2
    out["edge_f1"] = e1
    out["edge_f2"] = e2
    out["best_edge"] = best
    out["bet_side"] = side
    return out


def main() -> int:
    features = _load_features()
    model_path = config.DEFAULT_MODEL_PATH
    if not Path(model_path).is_file():
        raise SystemExit(f"Missing model: {model_path}")
    logger.info("Loading predictor %s", model_path)
    predictor = FightPredictor(model_path)
    logger.info(
        "Model features=%s fingerprint=%s conformal_q=%s",
        len(predictor.feature_columns),
        (predictor.artifact or {}).get("features_fingerprint"),
        getattr(predictor, "conformal_q", None),
    )

    scored = walk_forward_score_2025(features, predictor, year=2025)
    scored = _attach_market_fields(scored)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    # Slim export for autopsy + keep full
    keep = [
        c
        for c in scored.columns
        if c
        in {
            "fight_id",
            "event_date",
            "event_name",
            "fighter_1",
            "fighter_2",
            "winner",
            "predicted_winner",
            "predicted_prob",
            "prob_f1_win",
            "prob_f2_win",
            "correct",
            "f1_odds",
            "f2_odds",
            "implied_prob_f1",
            "implied_prob_f2",
            "edge_f1",
            "edge_f2",
            "best_edge",
            "interval_width",
            "ensemble_disagreement",
            "prob_ci_low",
            "prob_ci_high",
            "prob_lgbm",
            "prob_xgb",
            "method",
            "weight_class",
            "elo_diff",
            "momentum_diff",
            "last5_winrate_diff",
            "win_rate_diff",
            "reach_diff",
            "age_diff",
            "striker_score_diff",
            "grappler_score_diff",
            "sig_strikes_per_min_diff",
            "td_defense_diff",
            "finish_rate_diff",
            "experience_diff",
            "sos_opp_win_rate_diff",
            "hv_short_notice_flag_diff",
            "hv_long_layoff_flag_diff",
            "first_fight_new_wc_flag_diff",
            "ko_losses_career_flag_diff",
            "finish_rate_l5_diff",
            "division_age_adj_diff",
            "hv_td_pressure_diff",
            "hv_control_clash",
            "wins_vs_better_record_l5_diff",
        }
        or c.startswith("hv_")
    ]
    # unique preserve order
    seen = set()
    slim_cols = []
    for c in keep:
        if c in scored.columns and c not in seen:
            seen.add(c)
            slim_cols.append(c)
    slim = scored[slim_cols].copy()
    slim.to_csv(OUT_CSV, index=False)
    scored.to_csv(OUT_FULL, index=False)

    meta = {
        "n_fights": int(len(scored)),
        "accuracy": float(scored["correct"].mean()),
        "mean_interval_width": float(scored["interval_width"].mean()),
        "mean_ensemble_disagreement": float(scored["ensemble_disagreement"].mean()),
        "n_with_odds": int(
            scored["f1_odds"].notna().sum() if "f1_odds" in scored.columns else 0
        ),
        "model_path": str(model_path),
        "features_fingerprint": (predictor.artifact or {}).get("features_fingerprint"),
        "out_csv": str(OUT_CSV),
    }
    (config.DATA_DIR / "reports" / "backtest_2025_rescore_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s (%s rows) meta=%s", OUT_CSV, len(slim), meta)

    # Run upset autopsy on rescored file
    os.environ["UPSET_AUTOPSY_SRC"] = str(OUT_CSV)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "upset_autopsy_backtest",
        ROOT / "scripts" / "upset_autopsy_backtest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
