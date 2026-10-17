"""Year-scoped event backtest with strict as-of prices and bankroll rules.

Default ``walk_forward=False`` scores a frozen production model
(``evaluation_mode=frozen_model``). Pass ``walk_forward=True`` to run the
simplified past-only trainer (``evaluation_mode=walk_forward_simplified``):
a small LightGBM + optional isotonic calibration — not the production
ensemble / discovery pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from typing import Any

import numpy as np
import pandas as pd

from ufc_betting_bot.backtester.metrics import (
    build_calibration_bins,
    evaluate_classification,
    segment_metrics,
)
from ufc_betting_bot.backtester.reports import (
    print_backtest_summary,
    save_calibration_plot,
    save_roi_plot,
)
from ufc_betting_bot.config.settings import (
    BACKTEST_2025_CSV,
    DATE_COLUMN,
    PLOTS_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    BankrollSettings,
    ensure_dirs,
    get_settings,
)
from ufc_betting_bot.modules.bankroll import (
    simulate_all_staking_modes,
    simulate_bankroll_bets,
    simulate_bankroll_bets_dynamic,
    sweep_bankroll_thresholds,
)
from ufc_betting_bot.modules.dynamic_thresholds import print_threshold_comparison_report
from ufc_betting_bot.modules.edge import compute_edge
from ufc_betting_bot.modules.model_bridge import get_predictor, load_features, load_fights, model_exists

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Replace undefined floats with ``\"N/A\"`` for valid JSON output."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "N/A"
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


@dataclass
class Backtest2025Result:
    predictions: pd.DataFrame
    overall_metrics: dict[str, float]
    segment_metrics: dict[str, dict[str, float]]
    per_event: pd.DataFrame
    bankroll_sweep: pd.DataFrame
    bankroll_trades: pd.DataFrame
    bankroll_summary: dict[str, float]
    staking_modes: pd.DataFrame
    calibration_bins: pd.DataFrame
    target_year: int
    fights_with_odds: int
    report_dir: Path
    monte_carlo: Any = None
    threshold_comparison: dict[str, Any] | None = None
    evaluation_mode: str = "frozen_model"
    as_of_eligibility: dict[str, int] | None = None
    trainer_identity: str = "production_artifact"


def _attach_event_names(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "event" in out.columns and out["event"].notna().any():
        return out

    fights = load_fights()
    if "fight_id" not in out.columns or "event" not in fights.columns:
        out["event"] = out.get("event_name", pd.Series("Unknown", index=out.index))
        return out

    meta = fights[["fight_id", "event"]].drop_duplicates("fight_id")
    out = out.merge(meta, on="fight_id", how="left")
    out["event"] = out["event"].fillna("Unknown")
    return out


def _interaction_names(predictor) -> set[str]:
    names: set[str] = set()
    for spec in getattr(predictor, "interaction_specs", None) or []:
        name = spec.get("name") if isinstance(spec, dict) else getattr(spec, "name", None)
        if name:
            names.add(str(name))
    return names


def _walk_forward_feature_columns(
    df: pd.DataFrame,
    predictor,
    *,
    refit_model: bool,
) -> list[str]:
    """Columns used for scoring.

    Walk-forward uses the fixed leakage-safe ``config.FEATURE_COLUMNS`` base
    list (never production-selected interaction columns). Frozen scoring may
    still use the artifact feature list after applying its interaction specs.
    """
    import config

    interactions = _interaction_names(predictor)
    if refit_model:
        base = [
            c
            for c in config.FEATURE_COLUMNS
            if c in df.columns and c not in interactions
        ]
        if base:
            return base
        # Synthetic fixtures often use a tiny non-production column set.
        return [
            c
            for c in (getattr(predictor, "feature_columns", None) or [])
            if c in df.columns and c not in interactions
        ]
    return [
        c
        for c in (getattr(predictor, "feature_columns", None) or [])
        if c in df.columns
    ]


def walk_forward_events(
    features: pd.DataFrame,
    predictor,
    *,
    target_year: int,
    refit_model: bool = False,
    trainer=None,
    min_train_rows: int = 80,
) -> pd.DataFrame:
    """Predict each event from information available before that event.

    ``refit_model=False`` scores a frozen artifact (``frozen_model``).
    ``refit_model=True`` with the default trainer uses the simplified
    past-only LightGBM path (``walk_forward_simplified``), not production
    ensemble training.
    """
    from src.feature_engineering import apply_imputer, apply_interaction_specs, fit_imputer
    from ufc_betting_bot.modules.edge import validated_as_of_price

    frozen_interaction_specs = getattr(predictor, "interaction_specs", None) or []

    df = _attach_event_names(features)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN, TARGET_COLUMN])
    df["event_key"] = (
        df["event"].astype(str) + "|" + df[DATE_COLUMN].dt.normalize().astype(str)
    )

    year_mask = df[DATE_COLUMN].dt.year == target_year
    if not year_mask.any():
        return pd.DataFrame()

    events = (
        df.loc[year_mask, ["event_key", "event", DATE_COLUMN]]
        .drop_duplicates("event_key")
        .sort_values(DATE_COLUMN)
    )

    feature_cols = _walk_forward_feature_columns(df, predictor, refit_model=refit_model)
    rows: list[dict] = []
    if not refit_model:
        mode = "frozen_model"
        trainer_identity = "production_artifact"
    elif trainer is None:
        mode = "walk_forward_simplified"
        trainer_identity = "simplified_lgbm_isotonic"
    else:
        mode = "walk_forward_simplified"
        trainer_identity = getattr(trainer, "__name__", type(trainer).__name__)

    for _, ev in events.iterrows():
        ev_date = ev[DATE_COLUMN]
        ev_key = ev["event_key"]
        train = df[df[DATE_COLUMN] < ev_date]
        test = df[(df["event_key"] == ev_key) & year_mask]
        if train.empty or test.empty:
            continue

        if not refit_model and frozen_interaction_specs:
            train = apply_interaction_specs(train, frozen_interaction_specs)
            test = apply_interaction_specs(test, frozen_interaction_specs)

        fold_cols = [c for c in feature_cols if c in train.columns and c in test.columns]
        if not fold_cols:
            continue

        if refit_model:
            if trainer is None and len(train) < min_train_rows:
                continue
            fit_model = trainer or _fit_past_only_model
            model = fit_model(train, fold_cols)
            prepared = test
            if trainer is None:
                imputer = fit_imputer(train)
                prepared = apply_imputer(test, imputer).dropna(subset=fold_cols)
            if prepared.empty:
                continue
            proba = model.predict_proba(prepared[fold_cols])[:, 1]
        else:
            imputer = fit_imputer(train)
            prepared = apply_imputer(test, imputer).dropna(subset=fold_cols)
            if prepared.empty:
                continue
            proba = predictor.model.predict_proba(prepared[fold_cols])[:, 1]

        for i, (_, row) in enumerate(prepared.iterrows()):
            actual = int(row[TARGET_COLUMN])
            p1 = float(proba[i])
            price, odds_timing = validated_as_of_price(row, as_of=ev_date)
            record = row.to_dict()
            record["prob_f1_win"] = p1
            record["prob_f2_win"] = 1.0 - p1
            record["predicted_winner"] = row.get("fighter_1") if p1 >= 0.5 else row.get("fighter_2")
            record["correct"] = int((p1 >= 0.5) == bool(actual))
            record["wf_train_rows"] = len(train)
            record["event_name"] = ev["event"]
            record["evaluation_mode"] = mode
            record["trainer_identity"] = trainer_identity
            record["odds_timing"] = odds_timing

            if price is not None:
                edges = compute_edge(p1, 1.0 - p1, price.market_p1, price.market_p2)
                record["implied_prob_f1"] = price.market_p1
                record["implied_prob_f2"] = price.market_p2
                record["as_of_decimal_f1"] = price.decimal_f1
                record["as_of_decimal_f2"] = price.decimal_f2
                record.update(edges)
            else:
                record["implied_prob_f1"] = np.nan
                record["implied_prob_f2"] = np.nan
                record["as_of_decimal_f1"] = np.nan
                record["as_of_decimal_f2"] = np.nan
                record["edge_f1"] = np.nan
                record["edge_f2"] = np.nan
                record["best_edge"] = np.nan
                record["bet_side"] = ""
                record["edge"] = np.nan
            rows.append(record)

    return pd.DataFrame(rows)


def _fit_past_only_model(train: pd.DataFrame, feature_cols: list[str]):
    """Simplified past-only trainer (not the production ensemble pipeline).

    Fits imputer + small LightGBM + optional isotonic calibration on the
    caller-provided past slice only. Does not rediscover interactions or
    run production Optuna / dual-model training.
    """
    from lightgbm import LGBMClassifier

    from src.feature_engineering import apply_imputer, fit_imputer

    cols = [col for col in feature_cols if col in train.columns]
    work = train.dropna(subset=[TARGET_COLUMN]).copy()
    if work.empty or not cols:
        raise ValueError("Walk-forward train slice has no usable rows")

    # Chronological order so a calibration holdout is strictly earlier than test.
    if DATE_COLUMN in work.columns:
        work = work.sort_values(DATE_COLUMN)

    imputer = fit_imputer(work)
    prepared = apply_imputer(work, imputer).dropna(subset=cols + [TARGET_COLUMN])
    if prepared.empty or not cols:
        raise ValueError("Walk-forward train slice has no usable rows")

    y = prepared[TARGET_COLUMN].astype(int)
    X = prepared[cols]
    cal_n = 0
    if len(prepared) >= 40:
        cal_n = max(8, min(int(len(prepared) * 0.15), len(prepared) // 5))
        if len(prepared) - cal_n < 12:
            cal_n = 0

    if cal_n:
        X_fit, y_fit = X.iloc[:-cal_n], y.iloc[:-cal_n]
        X_cal, y_cal = X.iloc[-cal_n:], y.iloc[-cal_n:]
    else:
        X_fit, y_fit = X, y
        X_cal = y_cal = None

    model = LGBMClassifier(
        n_estimators=40,
        num_leaves=8,
        learning_rate=0.1,
        verbose=-1,
        random_state=42,
    )
    model.fit(X_fit, y_fit)

    if X_cal is not None and len(X_cal) >= 8:
        try:
            from sklearn.calibration import CalibratedClassifierCV

            calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            calibrated.fit(X_cal, y_cal)
            return calibrated
        except Exception:
            pass
    return model


def _per_event_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "event_name" not in predictions.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for event, grp in predictions.groupby("event_name"):
        m = evaluate_classification(grp[TARGET_COLUMN], grp["prob_f1_win"])
        if "odds_timing" in grp.columns:
            eligible = int((grp["odds_timing"].astype(str) == "opening").sum())
        else:
            eligible = 0
        rows.append(
            {
                "event": event,
                "event_date": grp[DATE_COLUMN].iloc[0],
                "fights": len(grp),
                "fights_as_of_eligible": eligible,
                "fights_with_odds": eligible,
                "accuracy": m["accuracy"],
                "log_loss": m["log_loss"],
                "brier_score": m["brier_score"],
                "avg_best_edge": float(grp["best_edge"].mean()) if "best_edge" in grp else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("event_date")


def backtest_2025(
    features: pd.DataFrame | None = None,
    *,
    target_year: int | None = None,
    bankroll_settings: BankrollSettings | None = None,
    edge_thresholds: list[float] | None = None,
    save_outputs: bool = True,
    use_dynamic_thresholds: bool | None = None,
    compare_threshold_modes: bool | None = None,
    profile: str | None = None,
    walk_forward: bool = False,
) -> Backtest2025Result:
    ensure_dirs()
    settings = get_settings()
    year = target_year or settings.backtest_year
    bankroll = bankroll_settings or settings.bankroll
    thresholds = edge_thresholds or settings.edge_thresholds

    if not model_exists():
        raise FileNotFoundError("No ufc-predictor model. Run: cd ufc-predictor && python main.py --train")

    if features is None:
        features = load_features()

    if TARGET_COLUMN not in features.columns:
        raise ValueError(f"Features missing '{TARGET_COLUMN}'.")

    predictor = get_predictor()
    predictions = walk_forward_events(
        features,
        predictor,
        target_year=year,
        refit_model=walk_forward,
    )
    from ufc_betting_bot.modules.edge import as_of_eligibility_counts

    if walk_forward:
        evaluation_mode = "walk_forward_simplified"
        trainer_identity = "simplified_lgbm_isotonic"
    else:
        evaluation_mode = "frozen_model"
        trainer_identity = "production_artifact"
    if "evaluation_mode" in predictions.columns and not predictions.empty:
        evaluation_mode = str(predictions["evaluation_mode"].iloc[0])
    if "trainer_identity" in predictions.columns and not predictions.empty:
        trainer_identity = str(predictions["trainer_identity"].iloc[0])

    if predictions.empty:
        raise ValueError(
            f"No fights for {year}. Max date: {features[DATE_COLUMN].max()}. "
            "Refresh ufc-predictor data first."
        )

    overall = evaluate_classification(
        predictions[TARGET_COLUMN], predictions["prob_f1_win"]
    )
    main_mask = predictions.get("is_main_event", pd.Series(0, index=predictions.index)).astype(bool)
    title_mask = predictions.get("is_title_fight", pd.Series(0, index=predictions.index)).astype(bool)
    undercard_mask = ~main_mask & ~title_mask

    segments = {
        "main_events": segment_metrics(predictions, main_mask, TARGET_COLUMN, "prob_f1_win"),
        "title_fights": segment_metrics(predictions, title_mask, TARGET_COLUMN, "prob_f1_win"),
        "undercard": segment_metrics(predictions, undercard_mask, TARGET_COLUMN, "prob_f1_win"),
    }

    as_of_eligibility = as_of_eligibility_counts(predictions)
    fights_with_odds = int(as_of_eligibility.get("n_as_of_eligible", 0))

    active_profile = profile
    if active_profile is None:
        try:
            import config as _cfg

            active_profile = _cfg.UFC_PROFILE
        except ImportError:
            active_profile = "research"

    dynamic_enabled = use_dynamic_thresholds
    if dynamic_enabled is None:
        try:
            import config as _cfg

            dynamic_enabled = _cfg.DYNAMIC_THRESHOLDS_ENABLED
        except ImportError:
            dynamic_enabled = True

    compare_modes = compare_threshold_modes
    if compare_modes is None:
        compare_modes = dynamic_enabled

    if compare_modes:
        bankroll_trades, static_summary = simulate_bankroll_bets(predictions, bankroll)
        dyn_trades, dynamic_summary = simulate_bankroll_bets_dynamic(
            predictions, bankroll, profile=active_profile
        )
        bankroll_summary = dynamic_summary if dynamic_enabled else static_summary
        threshold_comparison = {
            "static": static_summary,
            "dynamic": dynamic_summary,
            "static_trades": len(bankroll_trades),
            "dynamic_trades": len(dyn_trades),
            "profile": active_profile,
        }
        print_threshold_comparison_report(
            static_summary,
            dynamic_summary,
            profile=active_profile,
            target_year=year,
        )
    elif dynamic_enabled:
        bankroll_trades, bankroll_summary = simulate_bankroll_bets_dynamic(
            predictions, bankroll, profile=active_profile
        )
        threshold_comparison = None
    else:
        bankroll_trades, bankroll_summary = simulate_bankroll_bets(predictions, bankroll)
        threshold_comparison = None
    bankroll_sweep = sweep_bankroll_thresholds(predictions, bankroll, thresholds)
    staking_modes = simulate_all_staking_modes(predictions, bankroll)

    monte_carlo = None
    try:
        from src.risk_manager import (
            enrich_staking_modes_with_mc,
            run_monte_carlo,
            save_monte_carlo_report,
        )

        monte_carlo = run_monte_carlo(
            predictions,
            random_seed=42,
            initial_bankroll=bankroll.initial_bankroll,
            min_edge=bankroll.min_edge,
        )
        staking_modes = enrich_staking_modes_with_mc(staking_modes, monte_carlo)
    except Exception as exc:
        logger.warning("Monte Carlo risk analysis skipped: %s", exc)

    calibration_bins = build_calibration_bins(
        predictions[TARGET_COLUMN], predictions["prob_f1_win"]
    )
    per_event = _per_event_breakdown(predictions)

    if as_of_eligibility.get("n_as_of_eligible", 0) == 0:
        # Do not present empty staking as break-even performance.
        for summary in (bankroll_summary,):
            summary["performance_available"] = False
            summary["roi_pct"] = float("nan")
            summary["hit_rate"] = float("nan")
        if isinstance(staking_modes, pd.DataFrame) and not staking_modes.empty:
            staking_modes = staking_modes.copy()
            staking_modes["performance_available"] = False
            staking_modes["roi_pct"] = np.nan
            staking_modes["hit_rate"] = np.nan
        if isinstance(bankroll_sweep, pd.DataFrame) and not bankroll_sweep.empty:
            bankroll_sweep = bankroll_sweep.copy()
            bankroll_sweep["performance_available"] = False
            bankroll_sweep["roi_pct"] = np.nan
            bankroll_sweep["hit_rate"] = np.nan

    result = Backtest2025Result(
        predictions=predictions,
        overall_metrics=overall,
        segment_metrics=segments,
        per_event=per_event,
        bankroll_sweep=bankroll_sweep,
        bankroll_trades=bankroll_trades,
        bankroll_summary=bankroll_summary,
        staking_modes=staking_modes,
        calibration_bins=calibration_bins,
        target_year=year,
        fights_with_odds=fights_with_odds,
        report_dir=REPORTS_DIR,
        monte_carlo=monte_carlo,
        threshold_comparison=threshold_comparison,
        evaluation_mode=evaluation_mode,
        as_of_eligibility=as_of_eligibility,
        trainer_identity=trainer_identity,
    )

    if save_outputs:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(BACKTEST_2025_CSV, index=False)
        per_event.to_csv(REPORTS_DIR / f"backtest_{year}_per_event.csv", index=False)
        # Undefined staking metrics must serialize as N/A, not blank/NaN.
        sweep_out = bankroll_sweep.copy()
        modes_out = staking_modes.copy()
        for frame in (sweep_out, modes_out):
            for col in ("roi_pct", "hit_rate", "avg_yield_pct"):
                if col in frame.columns:
                    frame[col] = frame[col].where(frame[col].notna(), other="N/A")
        sweep_out.to_csv(REPORTS_DIR / f"backtest_{year}_bankroll_sweep.csv", index=False)
        bankroll_trades.to_csv(REPORTS_DIR / f"backtest_{year}_bankroll_trades.csv", index=False)
        modes_out.to_csv(REPORTS_DIR / f"backtest_{year}_staking_modes.csv", index=False)
        calibration_bins.to_csv(REPORTS_DIR / f"backtest_{year}_calibration.csv", index=False)

        summary = {
            "target_year": year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fights_with_odds": fights_with_odds,
            "as_of_eligibility": as_of_eligibility,
            "bankroll_rules": bankroll.__dict__,
            **{f"overall_{k}": v for k, v in overall.items()},
            **{f"bankroll_{k}": v for k, v in bankroll_summary.items()},
            "dynamic_thresholds": dynamic_enabled,
            "threshold_comparison": threshold_comparison,
            "evaluation_mode": evaluation_mode,
            "trainer_identity": trainer_identity,
            "walk_forward": evaluation_mode.startswith("walk_forward"),
            "staking_modes": staking_modes.to_dict(orient="records"),
        }
        (REPORTS_DIR / f"backtest_{year}_summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2),
            encoding="utf-8",
        )
        save_calibration_plot(calibration_bins, PLOTS_DIR / f"calibration_{year}.png")
        save_roi_plot(bankroll_sweep, PLOTS_DIR / f"bankroll_roi_{year}.png")
        if monte_carlo is not None:
            save_monte_carlo_report(monte_carlo, REPORTS_DIR, year=year)
        logger.info("Backtest saved -> %s", BACKTEST_2025_CSV)

    return result
