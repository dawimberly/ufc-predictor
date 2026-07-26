"""
High-accuracy strategy backtest (last 12 months, rolling bankroll).

Uses the live decision stack (uncertainty gates, HA floors, ticket caps,
% card-budget allocation) on historical cards. Fail-closed when moneyline
odds are missing. Over 1.5 props are skipped without historical prop odds
(same as live HA: require live prop lines).

Modes:
  - Default: frozen production model (faster; may be in-sample optimistic)
  - ``--walk-forward``: retrain imputer + ensemble on fights **strictly before**
    each card date (true expanding-window walk-forward)

CLI::

    python -m main backtest --strategy high-accuracy --bankroll 100 --last-year
    python -m main backtest --strategy high-accuracy --bankroll 100 --walk-forward --last-year
"""

from __future__ import annotations

import html
import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Minimum labeled fights before an event date to fit a walk-forward model.
WF_MIN_TRAIN_ROWS = int(getattr(config, "HA_WF_MIN_TRAIN_ROWS", 400) or 400)
# Fast trees for per-event retrain (full production train is too slow for ~50 cards).
WF_N_ESTIMATORS = int(getattr(config, "HA_WF_N_ESTIMATORS", 120) or 120)


def _date_col(df: pd.DataFrame) -> str:
    for c in (config.DATE_COLUMN, "event_date", "date"):
        if c in df.columns:
            return c
    raise KeyError("No date column on features")


def _event_col(df: pd.DataFrame) -> str:
    for c in ("event_name", "event"):
        if c in df.columns:
            return c
    raise KeyError("No event column")


def _has_both_odds(row: pd.Series) -> bool:
    try:
        o1 = float(row.get("f1_odds"))
        o2 = float(row.get("f2_odds"))
    except (TypeError, ValueError):
        return False
    if not np.isfinite(o1) or not np.isfinite(o2):
        return False
    # American or decimal
    if abs(o1) >= 100 and abs(o2) >= 100:
        return True
    return o1 > 1.0 and o2 > 1.0


def filter_last_year(features: pd.DataFrame, *, as_of: datetime | None = None) -> pd.DataFrame:
    """Keep fights in the last 365 days with a valid date."""
    as_of = as_of or datetime.now()
    start = as_of - timedelta(days=365)
    dt = _date_col(features)
    out = features.copy()
    out["_bt_date"] = pd.to_datetime(out[dt], errors="coerce")
    out = out[out["_bt_date"].notna()].copy()
    out = out[(out["_bt_date"] >= start) & (out["_bt_date"] <= as_of)].copy()
    return out.sort_values("_bt_date").reset_index(drop=True)


def list_events_chrono(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Unique events oldest → newest (deduped by date + normalized name)."""
    ev = _event_col(features)
    g = (
        features.groupby(features[ev].astype(str), sort=False)
        .agg(event_date=("_bt_date", "min"), n_fights=("_bt_date", "size"))
        .reset_index()
        .rename(columns={ev: "event"})
    )
    g = g.sort_values("event_date")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for _, r in g.iterrows():
        name = str(r["event"]).strip()
        if not name:
            continue
        day = pd.Timestamp(r["event_date"]).strftime("%Y-%m-%d")
        key = f"{day}|{name.lower().replace(' ', '')}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"event": name, "event_date": day, "n_fights": int(r["n_fights"])})
    return out


def _card_pool_usd(bankroll: float, *, drawdown_pct: float = 0.0) -> float:
    """Per-card budget from profile risk rules, scaled down after drawdowns."""
    br = max(float(bankroll), 0.0)
    if br <= 0:
        return 0.0
    try:
        pool = float(config.max_card_stake_cap(br))
    except Exception:
        frac = float(config.profile_value("max_card_risk_fraction") or 0.15)
        pool = br * frac
    try:
        from src.high_accuracy_strategy import card_risk_drawdown_multiplier

        pool *= float(card_risk_drawdown_multiplier(drawdown_pct))
    except Exception:
        pass
    return max(0.0, min(pool, br))


def _pick_won(row: pd.Series, pick: str) -> bool | None:
    from src.replay import _actual_winner, _fighters_match

    actual = _actual_winner(row)
    if not actual:
        # Fall back to target
        try:
            y = row.get(config.TARGET_COLUMN)
            if pd.notna(y):
                f1 = str(row.get("fighter_1") or row.get("fighter1") or "")
                f2 = str(row.get("fighter_2") or row.get("fighter2") or "")
                actual = f1 if int(y) == 1 else f2
        except Exception:
            return None
    if not actual or not pick:
        return None
    return bool(_fighters_match(str(pick), str(actual)))


def _opening_odds(row: pd.Series, pick: str) -> float | None:
    from src.replay import _opening_odds_for_pick

    return _opening_odds_for_pick(row, pick)


def _settle_single(
    row: pd.Series,
    bet: dict[str, Any],
) -> dict[str, Any] | None:
    from src.settlement import compute_pnl

    pick = str(bet.get("pick") or "")
    if not _has_both_odds(row):
        return {
            "bet_type": "single",
            "status": "fail_closed_no_odds",
            "stake": float(bet.get("suggested_stake") or 0),
            "stake_pct": float(bet.get("stake_pct") or 0),
            "pnl": None,
            "won": None,
            "pick": pick,
            "fight": str(bet.get("fight") or ""),
        }
    odds = _opening_odds(row, pick)
    if odds is None or odds <= 1.0:
        return {
            "bet_type": "single",
            "status": "fail_closed_no_odds",
            "stake": float(bet.get("suggested_stake") or 0),
            "stake_pct": float(bet.get("stake_pct") or 0),
            "pnl": None,
            "won": None,
            "pick": pick,
            "fight": str(bet.get("fight") or ""),
        }
    won = _pick_won(row, pick)
    if won is None:
        return {
            "bet_type": "single",
            "status": "no_result",
            "stake": float(bet.get("suggested_stake") or 0),
            "stake_pct": float(bet.get("stake_pct") or 0),
            "pnl": None,
            "won": None,
            "pick": pick,
            "odds": odds,
            "fight": str(bet.get("fight") or ""),
        }
    stake = float(bet.get("suggested_stake") or 0)
    if stake <= 0:
        return None
    pnl = compute_pnl(correct=bool(won), stake=stake, opening_odds=odds)
    return {
        "bet_type": "single",
        "status": "settled",
        "stake": stake,
        "stake_pct": float(bet.get("stake_pct") or 0),
        "pnl": float(pnl) if pnl is not None else None,
        "won": int(bool(won)),
        "pick": pick,
        "odds": odds,
        "edge": float(bet.get("edge") or 0),
        "prob": bet.get("prob"),
        "confidence": bet.get("confidence"),
        "strength_score": bet.get("strength_score"),
        "sizing_mode": bet.get("sizing_mode") or "conf_odds",
        "sizing_target_pct": bet.get("sizing_target_pct"),
        "uncertainty_action": bet.get("uncertainty_action"),
        "uncertainty_penalty": bet.get("uncertainty_penalty"),
        "fight": str(bet.get("fight") or ""),
        "fight_id": str(bet.get("fight_id") or ""),
    }


def _to_decimal_odds(raw: Any) -> float | None:
    """Normalize American or decimal price to decimal odds (fail-closed if invalid)."""
    from src.strategy import sanitize_decimal_odds

    return sanitize_decimal_odds(raw)


def _settle_parlay(
    preds: pd.DataFrame,
    parlay: dict[str, Any],
) -> dict[str, Any] | None:
    from src.settlement import compute_pnl
    from src.replay import _fighters_match

    legs = parlay.get("legs") or []
    if len(legs) != 2:
        return {
            "bet_type": "parlay_2leg",
            "status": "skipped_not_2leg",
            "stake": float(parlay.get("suggested_stake") or 0),
            "stake_pct": float(parlay.get("stake_pct") or 0),
            "pnl": None,
            "won": None,
            "picks": str(parlay.get("picks") or ""),
        }

    fid_col = config.FIGHT_ID_COLUMN if config.FIGHT_ID_COLUMN in preds.columns else None
    by_id: dict[str, pd.Series] = {}
    for _, row in preds.iterrows():
        fid = str(row.get(fid_col) or "") if fid_col else ""
        f1 = str(row.get("fighter_1") or row.get("fighter1") or "")
        f2 = str(row.get("fighter_2") or row.get("fighter2") or "")
        fight = f"{f1} vs {f2}"
        if fid:
            by_id[fid] = row
        by_id[fight] = row

    combined_odds = 1.0
    all_won = True
    for leg in legs:
        fid = str(leg.get("fight_id") or "")
        pick = str(leg.get("pick_name") or leg.get("winner_name") or leg.get("pick") or "")
        row = by_id.get(fid)
        if row is None:
            for r in by_id.values():
                f1 = str(r.get("fighter_1") or r.get("fighter1") or "")
                f2 = str(r.get("fighter_2") or r.get("fighter2") or "")
                if _fighters_match(pick, f1) or _fighters_match(pick, f2):
                    row = r
                    break
        if row is None or not _has_both_odds(row):
            return {
                "bet_type": "parlay_2leg",
                "status": "fail_closed_no_odds",
                "stake": float(parlay.get("suggested_stake") or 0),
                "stake_pct": float(parlay.get("stake_pct") or 0),
                "pnl": None,
                "won": None,
                "picks": str(parlay.get("picks") or ""),
            }
        # Prefer sanitized opening odds from the fight row (never trust raw American as decimal)
        odds = _opening_odds(row, pick)
        if odds is None:
            odds = _to_decimal_odds(leg.get("odds") or leg.get("decimal_odds"))
        if odds is None or odds <= 1.0:
            return {
                "bet_type": "parlay_2leg",
                "status": "fail_closed_no_odds",
                "stake": float(parlay.get("suggested_stake") or 0),
                "stake_pct": float(parlay.get("stake_pct") or 0),
                "pnl": None,
                "won": None,
                "picks": str(parlay.get("picks") or ""),
            }
        won = _pick_won(row, pick)
        if won is None:
            return {
                "bet_type": "parlay_2leg",
                "status": "no_result",
                "stake": float(parlay.get("suggested_stake") or 0),
                "stake_pct": float(parlay.get("stake_pct") or 0),
                "pnl": None,
                "won": None,
                "picks": str(parlay.get("picks") or ""),
            }
        if not won:
            all_won = False
        combined_odds *= odds

    # Reject absurd combined prices (bad American→decimal leakage)
    if combined_odds > 40.0:
        return {
            "bet_type": "parlay_2leg",
            "status": "fail_closed_bad_odds",
            "stake": float(parlay.get("suggested_stake") or 0),
            "stake_pct": float(parlay.get("stake_pct") or 0),
            "pnl": None,
            "won": None,
            "odds": combined_odds,
            "picks": str(parlay.get("picks") or ""),
        }

    stake = float(parlay.get("suggested_stake") or 0)
    if stake <= 0:
        return None
    pnl = compute_pnl(correct=bool(all_won), stake=stake, opening_odds=combined_odds)
    return {
        "bet_type": "parlay_2leg",
        "status": "settled",
        "stake": stake,
        "stake_pct": float(parlay.get("stake_pct") or 0),
        "pnl": float(pnl) if pnl is not None else None,
        "won": int(bool(all_won)),
        "odds": combined_odds,
        "edge": float(parlay.get("edge") or parlay.get("expected_value") or 0),
        "prob": parlay.get("prob") or parlay.get("combined_prob"),
        "confidence": parlay.get("confidence") or parlay.get("min_leg_confidence"),
        "strength_score": parlay.get("strength_score"),
        "sizing_mode": parlay.get("sizing_mode") or "conf_odds",
        "sizing_target_pct": parlay.get("sizing_target_pct"),
        "uncertainty_action": parlay.get("uncertainty_action"),
        "uncertainty_penalty": parlay.get("uncertainty_penalty"),
        "picks": str(parlay.get("picks") or ""),
        "n_legs": 2,
    }


@dataclass
class WalkForwardPredictor:
    """Expanding-window scorer: imputer + LGBM/XGB ensemble trained on past-only rows."""

    model: Any
    feature_columns: list[str]
    imputer: Any
    conformal_q: float
    train_rows: int = 0
    train_end: str | None = None
    artifact: dict[str, Any] = field(default_factory=dict)
    _shap_explainer: Any = None
    _shap_cache_key: str = "walk_forward"

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        from src.feature_engineering import apply_imputer, build_interaction_candidates

        frame = features.copy()
        if config.INTERACTION_DISCOVERY_ENABLED:
            try:
                frame = build_interaction_candidates(frame)
            except Exception:
                pass
        frame = apply_imputer(frame, self.imputer)
        cols = [c for c in self.feature_columns if c in frame.columns]
        missing = [c for c in self.feature_columns if c not in frame.columns]
        for c in missing:
            frame[c] = 0.0
        cols = list(self.feature_columns)
        still = frame[cols].isna().any(axis=1)
        if still.any():
            frame = frame.loc[~still].copy()
        return frame

    def _attach_predictions(
        self,
        frame: pd.DataFrame,
        proba: np.ndarray,
        *,
        prepared: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        from src.ensemble import EnsembleClassifier, ensemble_disagreement, prediction_interval
        from src.predictor import _confidence_label, _confidence_score

        out = frame.copy()
        out["prob_f1_win"] = proba
        out["prob_f2_win"] = 1.0 - proba
        ci_low, ci_high, ci_width = prediction_interval(proba, conformal_q=self.conformal_q)
        out["prob_ci_low"] = ci_low
        out["prob_ci_high"] = ci_high
        out["interval_width"] = ci_width

        disagree = np.zeros(len(proba))
        score_frame = prepared if prepared is not None else frame
        if isinstance(self.model, EnsembleClassifier):
            components = self.model.predict_proba_components(score_frame[self.feature_columns])
            disagree = ensemble_disagreement(components)
            out["prob_lgbm"] = components.get("lgbm", proba)
            out["prob_xgb"] = components.get("xgb", proba)
        out["ensemble_disagreement"] = disagree
        out["confidence"] = [_confidence_score(p) for p in proba]
        out["confidence_label"] = [_confidence_label(c) for c in out["confidence"]]
        out["uncertainty_label"] = [
            (
                "high"
                if w >= config.UNCERTAINTY_HIGH_WIDTH or d >= 0.08
                else ("medium" if w >= config.UNCERTAINTY_HIGH_WIDTH * 0.6 or d >= 0.04 else "low")
            )
            for w, d in zip(ci_width, disagree)
        ]
        f1 = out.get("fighter_1", out.get("fighter1", pd.Series("Fighter 1", index=out.index)))
        f2 = out.get("fighter_2", out.get("fighter2", pd.Series("Fighter 2", index=out.index)))
        out["predicted_winner"] = np.where(out["prob_f1_win"] >= 0.5, f1, f2)
        out["predicted_prob"] = np.where(
            out["prob_f1_win"] >= 0.5, out["prob_f1_win"], out["prob_f2_win"]
        )
        pick_f1 = out["prob_f1_win"] >= 0.5
        out["predicted_ci_low"] = np.where(pick_f1, ci_low, 1.0 - ci_high)
        out["predicted_ci_high"] = np.where(pick_f1, ci_high, 1.0 - ci_low)
        return out

    def predict_batch(
        self,
        features: pd.DataFrame,
        *,
        apply_style_bonus: bool = True,
        explain: bool = False,
    ) -> pd.DataFrame:
        from src.predictor import apply_style_calibration

        prepared = self._prepare_features(features)
        if prepared.empty:
            return features.iloc[0:0].copy()
        proba = self.model.predict_proba(prepared[self.feature_columns])[:, 1]
        if apply_style_bonus:
            proba, bonuses = apply_style_calibration(prepared, proba)
            out = self._attach_predictions(prepared, proba, prepared=prepared)
            out["style_bonus"] = bonuses
        else:
            out = self._attach_predictions(prepared, proba, prepared=prepared)
        return out


def _production_feature_columns() -> list[str]:
    """Prefer feature list from the saved artifact so WF matches live columns."""
    try:
        from src.predictor import FightPredictor

        cols = list(FightPredictor().feature_columns or [])
        if cols:
            return cols
    except Exception as exc:
        logger.debug("Could not load production feature columns: %s", exc)
    return list(config.FEATURE_COLUMNS)


class _CalibratedEnsemble:
    """Apply isotonic calibration to ensemble P(f1 win) while keeping component probs."""

    def __init__(self, base: Any, iso: Any) -> None:
        self.base = base
        self.iso = iso
        self.classes_ = getattr(base, "classes_", np.array([0, 1]))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        cal = np.asarray(self.iso.predict(raw), dtype=float)
        cal = np.clip(cal, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - cal, cal])

    def predict_proba_components(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        comps = self.base.predict_proba_components(X)
        # Calibrate each member with the same isotonic map for disagreement signal.
        out: dict[str, np.ndarray] = {}
        for name, probs in comps.items():
            out[name] = np.clip(np.asarray(self.iso.predict(probs), dtype=float), 1e-6, 1.0 - 1e-6)
        return out

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_walk_forward_predictor(
    train: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    n_estimators: int = WF_N_ESTIMATORS,
) -> WalkForwardPredictor:
    """Fit imputer + LGBM/XGB ensemble + conformal q on past-only labeled fights."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    from src.ensemble import (
        EnsembleClassifier,
        conformal_quantile,
        fit_conformal_scores,
    )
    from src.feature_engineering import apply_imputer, build_interaction_candidates, fit_imputer

    work = train.dropna(subset=[config.TARGET_COLUMN]).copy()
    if work.empty:
        raise ValueError("Walk-forward train set is empty")
    if config.INTERACTION_DISCOVERY_ENABLED:
        try:
            work = build_interaction_candidates(work)
        except Exception:
            pass

    cols = feature_columns or _production_feature_columns()
    cols = [c for c in cols if c in work.columns] or [
        c for c in config.FEATURE_COLUMNS if c in work.columns
    ]
    if len(cols) < 5:
        raise ValueError(f"Too few feature columns for walk-forward ({len(cols)})")

    imputer = fit_imputer(work)
    prepared = apply_imputer(work, imputer)
    prepared = prepared.dropna(subset=cols + [config.TARGET_COLUMN])
    if len(prepared) < 80:
        raise ValueError(f"Insufficient rows after impute ({len(prepared)})")

    # Chronological calibration holdout for conformal bands
    cal_n = max(40, min(int(len(prepared) * 0.15), len(prepared) // 4))
    if len(prepared) - cal_n < 60:
        cal_n = max(20, len(prepared) // 5)
    fit_df = prepared.iloc[:-cal_n]
    cal_df = prepared.iloc[-cal_n:]
    X_fit = fit_df[cols]
    y_fit = fit_df[config.TARGET_COLUMN].astype(int)
    X_cal = cal_df[cols]
    y_cal = cal_df[config.TARGET_COLUMN].astype(int)

    lgbm = LGBMClassifier(
        n_estimators=int(n_estimators),
        num_leaves=31,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        verbose=-1,
        random_state=int(getattr(config, "RANDOM_STATE", 42) or 42),
    )
    xgb = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        verbosity=0,
        random_state=int(getattr(config, "RANDOM_STATE", 42) or 42),
        eval_metric="logloss",
    )
    lgbm.fit(X_fit, y_fit)
    xgb.fit(X_fit, y_fit)
    ensemble = EnsembleClassifier(
        [lgbm, xgb],
        weights=[0.55, 0.45],
        names=["lgbm", "xgb"],
    )
    # Isotonic calibration on chronological holdout so probs + conformal bands are usable.
    raw_cal = ensemble.predict_proba(X_cal)[:, 1]
    try:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_cal, y_cal.to_numpy())
        cal_proba = iso.predict(raw_cal)
        ensemble = _CalibratedEnsemble(ensemble, iso)
    except Exception as exc:
        logger.debug("WF isotonic calibration skipped: %s", exc)
        cal_proba = raw_cal

    scores = fit_conformal_scores(y_cal.to_numpy(), cal_proba)
    alpha = float(getattr(config, "CONFORMAL_ALPHA", 0.1) or 0.1)
    cq = float(conformal_quantile(scores, alpha))
    # Fast WF fits without full production calibration can yield huge q, which makes
    # interval_width trip HA skip gates on every fight. Cap to a paper-usable band.
    q_cap = float(getattr(config, "HA_WF_CONFORMAL_Q_CAP", 0.14) or 0.14)
    q_floor = float(getattr(config, "HA_WF_CONFORMAL_Q_FLOOR", 0.05) or 0.05)
    cq_raw = cq
    cq = float(min(max(cq, q_floor), q_cap))
    if abs(cq_raw - cq) > 1e-6:
        logger.info("WF conformal_q adjusted %.3f → %.3f", cq_raw, cq)

    dt = _date_col(prepared) if any(c in prepared.columns for c in (config.DATE_COLUMN, "event_date", "date")) else None
    train_end = None
    if dt:
        train_end = str(pd.to_datetime(prepared[dt], errors="coerce").max().date())

    return WalkForwardPredictor(
        model=ensemble,
        feature_columns=cols,
        imputer=imputer,
        conformal_q=cq,
        train_rows=len(fit_df),
        train_end=train_end,
        artifact={"conformal_q": cq, "walk_forward": True},
    )


def _ensure_bt_date(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "_bt_date" not in out.columns:
        dt = _date_col(out)
        out["_bt_date"] = pd.to_datetime(out[dt], errors="coerce")
    return out


def _train_slice_before(
    features: pd.DataFrame,
    event_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Labeled fights with date strictly before the card date."""
    work = _ensure_bt_date(features)
    cutoff = pd.Timestamp(event_date).normalize()
    past = work[work["_bt_date"].notna() & (work["_bt_date"] < cutoff)].copy()
    if config.TARGET_COLUMN in past.columns:
        past = past.dropna(subset=[config.TARGET_COLUMN])
    return past.sort_values("_bt_date").reset_index(drop=True)


def _record_wf_tickets_to_segments(
    tickets: list[dict[str, Any]],
    *,
    event_name: str,
    event_date: str,
    profile: str,
) -> None:
    """Feed settled WF tickets into strategy_metrics so rating uses past-only results."""
    try:
        from src.strategy_performance import (
            odds_bucket_from_decimal,
            record_closed_bet,
        )
    except Exception:
        return
    for i, t in enumerate(tickets):
        if t.get("status") != "settled" or t.get("pnl") is None:
            continue
        pid = (
            f"wf|{event_date}|{event_name}|{t.get('bet_type')}|{t.get('fight_id') or t.get('fight') or t.get('picks') or i}"
        )
        odds = t.get("odds")
        try:
            odds_f = float(odds) if odds is not None else None
        except (TypeError, ValueError):
            odds_f = None
        record_closed_bet(
            prediction_id=pid,
            settled_at=f"{event_date}T00:00:00+00:00",
            profile=profile,
            market_type="parlay" if "parlay" in str(t.get("bet_type")) else "moneyline",
            weight_class="unknown",
            odds_bucket=odds_bucket_from_decimal(odds_f) if odds_f else "unknown",
            confidence_label="unknown",
            prop_type="moneyline",
            correct=bool(int(t.get("won") or 0)),
            stake=float(t.get("stake") or 0),
            pnl=float(t.get("pnl")),
            decimal_odds=odds_f,
            settlement_complete=True,
            source="ha_walkforward",
        )


def backtest_event(
    event_name: str,
    features: pd.DataFrame,
    *,
    predictor: Any,
    bankroll: float,
    use_dynamic_thresholds: bool = True,
    recent_wins: list[bool] | None = None,
    drawdown_pct: float = 0.0,
    fixed_stake_usd: float | None = None,
) -> dict[str, Any]:
    """Score one past card with HA stack; settle tickets against results.

    When ``fixed_stake_usd`` is set, every selected ticket gets that flat stake
    (no % bankroll compounding). Ticket selection still uses HA gates.
    """
    from src.alerts import generate_alerts
    from src.replay import score_event_card
    from src.strategy import allocate_alerts_card_budget_pct

    scored = score_event_card(event_name, features, predictor=predictor, explain=False)
    # Fail-closed: do not fetch live odds in historical backtest
    odds_mask = scored.apply(_has_both_odds, axis=1)
    odds_coverage = float(odds_mask.mean()) if len(scored) else 0.0

    alerts = generate_alerts(
        scored,
        bankroll=bankroll,
        event_name=event_name,
        use_dynamic_thresholds=use_dynamic_thresholds,
        narrative_result=None,
        recent_wins=recent_wins,
    )
    pool = _card_pool_usd(bankroll, drawdown_pct=drawdown_pct)
    # Drop Over 1.5 from betting pool historically (no prop odds) — HA live-odds rule
    alerts = allocate_alerts_card_budget_pct(
        alerts,
        pool,
        profile=config.UFC_PROFILE,
        prop_singles=[],  # no historical prop lines
    )

    if fixed_stake_usd is not None and float(fixed_stake_usd) > 0:
        flat = float(fixed_stake_usd)
        for s in alerts.get("singles") or []:
            s["suggested_stake"] = flat
            s["stake_pct"] = 0.0
        for p in alerts.get("parlays") or []:
            p["suggested_stake"] = flat
            p["stake_pct"] = 0.0

    fid_col = config.FIGHT_ID_COLUMN if config.FIGHT_ID_COLUMN in scored.columns else None
    by_id: dict[str, pd.Series] = {}
    for _, row in scored.iterrows():
        f1 = str(row.get("fighter_1") or row.get("fighter1") or "")
        f2 = str(row.get("fighter_2") or row.get("fighter2") or "")
        fight = f"{f1} vs {f2}"
        fid = str(row.get(fid_col) or fight) if fid_col else fight
        by_id[fid] = row
        by_id[fight] = row

    tickets: list[dict[str, Any]] = []
    for s in alerts.get("singles") or []:
        key = str(s.get("fight_id") or s.get("fight") or "")
        row = by_id.get(key)
        if row is None:
            tickets.append(
                {
                    "bet_type": "single",
                    "status": "missing_row",
                    "stake": float(s.get("suggested_stake") or 0),
                    "stake_pct": float(s.get("stake_pct") or 0),
                    "pnl": None,
                    "won": None,
                    "pick": s.get("pick"),
                    "fight": s.get("fight"),
                }
            )
            continue
        settled = _settle_single(row, s)
        if settled:
            settled["event"] = event_name
            tickets.append(settled)

    for p in alerts.get("parlays") or []:
        settled = _settle_parlay(scored, p)
        if settled:
            settled["event"] = event_name
            tickets.append(settled)

    # Props explicitly noted as skipped (HA requires live odds)
    props_skipped = int(len(alerts.get("prop_singles") or []))

    settled = [t for t in tickets if t.get("status") == "settled" and t.get("pnl") is not None]
    card_pnl = float(sum(float(t["pnl"]) for t in settled))
    card_stake = float(sum(float(t["stake"]) for t in settled))
    return {
        "event": event_name,
        "n_fights": int(len(scored)),
        "odds_coverage": odds_coverage,
        "card_pool_usd": pool if fixed_stake_usd is None else float(card_stake),
        "n_tickets": len(tickets),
        "n_settled": len(settled),
        "card_pnl": card_pnl,
        "card_stake": card_stake,
        "tickets": tickets,
        "props_skipped_no_odds": props_skipped,
        "alerts_singles": len(alerts.get("singles") or []),
        "alerts_parlays": len(alerts.get("parlays") or []),
        "skipped_count": int(alerts.get("skipped_count") or 0),
        "fixed_stake_usd": fixed_stake_usd,
    }


def run_ha_backtest(
    *,
    bankroll_start: float = 100.0,
    last_year: bool = True,
    use_dynamic_thresholds: bool = True,
    profile: str = "paper",
    as_of: datetime | None = None,
    walk_forward: bool = False,
) -> dict[str, Any]:
    """
    Walk last-year UFC cards with high-accuracy strategy and rolling bankroll.

    When ``walk_forward=True``, refits imputer + ensemble on fights strictly
    before each card date (true expanding-window walk-forward).
    """
    if walk_forward:
        return run_ha_walkforward_backtest(
            bankroll_start=bankroll_start,
            last_year=last_year,
            use_dynamic_thresholds=use_dynamic_thresholds,
            profile=profile,
            as_of=as_of,
        )

    from src.predictor import FightPredictor
    from src.replay import load_replay_features

    config.UFC_PROFILE = config.normalize_profile(profile)
    config.apply_profile_overrides()

    features = load_replay_features()
    if last_year:
        features = filter_last_year(features, as_of=as_of)
    if features.empty:
        raise ValueError("No labeled fights in the requested window.")

    events = list_events_chrono(features)
    if not events:
        raise ValueError("No events found in window.")

    predictor = FightPredictor()
    return _run_ha_event_loop(
        events,
        features,
        predictor=predictor,
        bankroll_start=bankroll_start,
        use_dynamic_thresholds=use_dynamic_thresholds,
        last_year=last_year,
        walk_forward=False,
        wf_train_meta=None,
    )


def run_ha_walkforward_backtest(
    *,
    bankroll_start: float = 100.0,
    last_year: bool = True,
    use_dynamic_thresholds: bool = True,
    profile: str = "paper",
    as_of: datetime | None = None,
    min_train_rows: int = WF_MIN_TRAIN_ROWS,
    fixed_stake_usd: float | None = None,
) -> dict[str, Any]:
    """
    True walk-forward HA backtest: for each card date D, train only on fights < D.

    ``fixed_stake_usd``: if set, every selected ticket uses that flat stake
    (no compounding). Selection gates are unchanged.
    """
    from src.replay import load_replay_features

    config.UFC_PROFILE = config.normalize_profile(profile)
    config.apply_profile_overrides()

    features_all = _ensure_bt_date(load_replay_features())
    if features_all.empty:
        raise ValueError("No features available for walk-forward backtest.")

    eval_features = (
        filter_last_year(features_all, as_of=as_of) if last_year else features_all.copy()
    )
    eval_features = _ensure_bt_date(eval_features)
    if eval_features.empty:
        raise ValueError("No labeled fights in the evaluation window.")

    events = list_events_chrono(eval_features)
    if not events:
        raise ValueError("No events found in window.")

    feature_cols = _production_feature_columns()
    scorer_by_date: dict[str, WalkForwardPredictor] = {}
    wf_meta: list[dict[str, Any]] = []

    # Isolate strategy-rating DB so segment health only sees this WF's past tickets
    tmp = Path(tempfile.mkdtemp(prefix="ha_wf_metrics_"))
    prev_db = getattr(config, "STRATEGY_METRICS_DB", None)
    prev_json = getattr(config, "STRATEGY_PERFORMANCE_JSON", None)
    config.STRATEGY_METRICS_DB = tmp / "strategy_metrics.db"
    config.STRATEGY_PERFORMANCE_JSON = tmp / "strategy_performance.json"

    try:
        bankroll = float(bankroll_start)
        peak = bankroll
        max_dd = 0.0
        max_dd_usd = 0.0
        equity: list[dict[str, Any]] = []
        all_tickets: list[dict[str, Any]] = []
        per_event: list[dict[str, Any]] = []
        recent_wins: list[bool] = []

        for i, ev in enumerate(events, start=1):
            name = ev["event"]
            day = ev["event_date"]
            logger.info(
                "[%s/%s] HA walk-forward %s (%s) bankroll=$%.2f",
                i,
                len(events),
                name,
                day,
                bankroll,
            )
            try:
                if day not in scorer_by_date:
                    train = _train_slice_before(features_all, day)
                    if len(train) < int(min_train_rows):
                        raise ValueError(
                            f"Need ≥{min_train_rows} past fights before {day} "
                            f"(have {len(train)})"
                        )
                    logger.info(
                        "Fitting WF model on %s fights before %s…",
                        len(train),
                        day,
                    )
                    scorer_by_date[day] = fit_walk_forward_predictor(
                        train, feature_columns=feature_cols
                    )
                    wf_meta.append(
                        {
                            "event_date": day,
                            "train_rows": scorer_by_date[day].train_rows,
                            "train_end": scorer_by_date[day].train_end,
                            "conformal_q": scorer_by_date[day].conformal_q,
                        }
                    )
                predictor = scorer_by_date[day]
                # Score only this card's rows (features already pre-fight / leakage-safe)
                card_mask = eval_features[_event_col(eval_features)].astype(str) == str(name)
                card_feats = eval_features.loc[card_mask].copy()
                if card_feats.empty:
                    # fall back to full history filter by name (same as score_event_card)
                    card_feats = features_all

                dd_now = (peak - bankroll) / peak if peak > 0 else 0.0
                result = backtest_event(
                    name,
                    card_feats if not card_feats.empty else features_all,
                    predictor=predictor,
                    bankroll=bankroll,
                    use_dynamic_thresholds=use_dynamic_thresholds,
                    recent_wins=list(recent_wins[-10:]),
                    drawdown_pct=dd_now,
                    fixed_stake_usd=fixed_stake_usd,
                )
            except Exception as exc:
                logger.warning("Event %s failed: %s", name, exc)
                equity.append(
                    {
                        "event": name,
                        "date": day,
                        "bankroll": bankroll,
                        "card_pnl": 0.0,
                        "n_tickets": 0,
                        "error": str(exc),
                    }
                )
                continue

            card_pnl = float(result["card_pnl"])
            bankroll = bankroll + card_pnl
            peak = max(peak, bankroll)
            dd = (peak - bankroll) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            max_dd_usd = max(max_dd_usd, peak - bankroll)

            for t in result["tickets"]:
                t["event"] = name
                t["event_date"] = day
                t["bankroll_after"] = bankroll
                t["wf_train_rows"] = getattr(predictor, "train_rows", None)
                all_tickets.append(t)
                if t.get("status") == "settled" and t.get("bet_type") == "single":
                    recent_wins.append(bool(int(t.get("won") or 0)))

            _record_wf_tickets_to_segments(
                result["tickets"],
                event_name=name,
                event_date=day,
                profile=config.UFC_PROFILE,
            )

            row = {
                "event": name,
                "date": day,
                "n_fights": result["n_fights"],
                "odds_coverage": result["odds_coverage"],
                "n_tickets": result["n_settled"],
                "card_pnl": card_pnl,
                "card_stake": result["card_stake"],
                "card_pool_usd": result["card_pool_usd"],
                "bankroll": bankroll,
                "drawdown": dd,
                "wf_train_rows": getattr(predictor, "train_rows", None),
            }
            per_event.append(row)
            equity.append(row)

        report = _finalize_ha_report(
            bankroll_start=bankroll_start,
            bankroll=bankroll,
            max_dd=max_dd,
            all_tickets=all_tickets,
            per_event=per_event,
            equity=equity,
            events=events,
            last_year=last_year,
            walk_forward=True,
            wf_train_meta=wf_meta,
        )
        if report.get("summary") is not None:
            report["summary"]["max_drawdown_usd"] = float(max_dd_usd)
            if fixed_stake_usd is not None:
                report["summary"]["fixed_stake_usd"] = float(fixed_stake_usd)
                report["summary"]["staking"] = "fixed_stake"
                notes = list(report["summary"].get("notes") or [])
                notes.append(
                    f"Fixed stake ${float(fixed_stake_usd):.2f} per ticket "
                    "(no % bankroll compounding)."
                )
                report["summary"]["notes"] = notes
        return report
    finally:
        if prev_db is not None:
            config.STRATEGY_METRICS_DB = prev_db
        if prev_json is not None:
            config.STRATEGY_PERFORMANCE_JSON = prev_json


def _run_ha_event_loop(
    events: list[dict[str, Any]],
    features: pd.DataFrame,
    *,
    predictor: Any,
    bankroll_start: float,
    use_dynamic_thresholds: bool,
    last_year: bool,
    walk_forward: bool,
    wf_train_meta: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    bankroll = float(bankroll_start)
    peak = bankroll
    max_dd = 0.0
    equity: list[dict[str, Any]] = []
    all_tickets: list[dict[str, Any]] = []
    per_event: list[dict[str, Any]] = []
    recent_wins: list[bool] = []

    for i, ev in enumerate(events, start=1):
        name = ev["event"]
        logger.info(
            "[%s/%s] HA backtest %s (%s) bankroll=$%.2f",
            i,
            len(events),
            name,
            ev["event_date"],
            bankroll,
        )
        try:
            dd_now = (peak - bankroll) / peak if peak > 0 else 0.0
            result = backtest_event(
                name,
                features,
                predictor=predictor,
                bankroll=bankroll,
                use_dynamic_thresholds=use_dynamic_thresholds,
                recent_wins=list(recent_wins[-10:]),
                drawdown_pct=dd_now,
            )
        except Exception as exc:
            logger.warning("Event %s failed: %s", name, exc)
            equity.append(
                {
                    "event": name,
                    "date": ev["event_date"],
                    "bankroll": bankroll,
                    "card_pnl": 0.0,
                    "n_tickets": 0,
                    "error": str(exc),
                }
            )
            continue

        card_pnl = float(result["card_pnl"])
        bankroll = bankroll + card_pnl
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        for t in result["tickets"]:
            t["event"] = name
            t["event_date"] = ev["event_date"]
            t["bankroll_after"] = bankroll
            all_tickets.append(t)
            if t.get("status") == "settled" and t.get("bet_type") == "single":
                recent_wins.append(bool(int(t.get("won") or 0)))

        row = {
            "event": name,
            "date": ev["event_date"],
            "n_fights": result["n_fights"],
            "odds_coverage": result["odds_coverage"],
            "n_tickets": result["n_settled"],
            "card_pnl": card_pnl,
            "card_stake": result["card_stake"],
            "card_pool_usd": result["card_pool_usd"],
            "bankroll": bankroll,
            "drawdown": dd,
        }
        per_event.append(row)
        equity.append(row)

    return _finalize_ha_report(
        bankroll_start=bankroll_start,
        bankroll=bankroll,
        max_dd=max_dd,
        all_tickets=all_tickets,
        per_event=per_event,
        equity=equity,
        events=events,
        last_year=last_year,
        walk_forward=walk_forward,
        wf_train_meta=wf_train_meta,
    )


def _finalize_ha_report(
    *,
    bankroll_start: float,
    bankroll: float,
    max_dd: float,
    all_tickets: list[dict[str, Any]],
    per_event: list[dict[str, Any]],
    equity: list[dict[str, Any]],
    events: list[dict[str, Any]],
    last_year: bool,
    walk_forward: bool,
    wf_train_meta: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    settled = [t for t in all_tickets if t.get("status") == "settled" and t.get("pnl") is not None]
    wins = [t for t in settled if int(t.get("won") or 0) == 1]
    total_stake = float(sum(float(t["stake"]) for t in settled))
    total_pnl = float(sum(float(t["pnl"]) for t in settled))
    final = float(bankroll)
    roi = (final - bankroll_start) / bankroll_start if bankroll_start else 0.0
    roi_on_stake = total_pnl / total_stake if total_stake > 0 else None

    by_type: dict[str, dict[str, Any]] = {}
    for t in settled:
        k = str(t.get("bet_type") or "other")
        bucket = by_type.setdefault(k, {"n": 0, "wins": 0, "stake": 0.0, "pnl": 0.0})
        bucket["n"] += 1
        bucket["wins"] += int(t.get("won") or 0)
        bucket["stake"] += float(t["stake"])
        bucket["pnl"] += float(t["pnl"])
    for k, b in by_type.items():
        b["hit_rate"] = b["wins"] / b["n"] if b["n"] else None
        b["roi"] = b["pnl"] / b["stake"] if b["stake"] else None

    eq_df = pd.DataFrame(equity)
    monthly: list[dict[str, Any]] = []
    if not eq_df.empty and "date" in eq_df.columns:
        eq_df["month"] = pd.to_datetime(eq_df["date"], errors="coerce").dt.to_period("M").astype(str)
        for month, g in eq_df.groupby("month", sort=True):
            monthly.append(
                {
                    "period": month,
                    "end_bankroll": float(g["bankroll"].iloc[-1]),
                    "pnl": float(g["card_pnl"].sum()),
                    "tickets": int(g["n_tickets"].sum()),
                    "events": int(len(g)),
                }
            )

    cards_with_bets = sum(1 for e in per_event if int(e.get("n_tickets") or 0) > 0)
    avg_bets_per_card = len(settled) / len(per_event) if per_event else 0.0
    avg_bets_when_active = len(settled) / cards_with_bets if cards_with_bets else 0.0

    notes = [
        "Fail-closed on missing moneyline odds (no live odds fetch).",
        "Over 1.5 props excluded historically (HA requires live prop odds).",
        "Stakes = conf/odds strength → % of per-card budget (sum ≤ 100%; no inflate on weak/missing odds).",
        "2-leg parlays settled from sanitized leg odds only (rejects absurd combined prices).",
        "Decision stack: uncertainty gates, strategy rating, dynamic thresholds, HA ticket caps.",
        "Path-risk: conf/odds compounding, max parlay share of card budget, "
        "drawdown-scaled card risk, parlays require high-confidence legs.",
    ]
    if walk_forward:
        notes.insert(
            0,
            "TRUE WALK-FORWARD: imputer + LGBM/XGB ensemble refit on fights strictly before each card date.",
        )
        notes.append(
            "Strategy-rating segment DB isolated to this run (past WF tickets only; no future leakage)."
        )
    else:
        notes.append(
            "Model is the current trained artifact (possible in-sample optimism vs true walk-forward)."
        )

    summary = {
        "strategy": "high-accuracy",
        "mode": "walk_forward" if walk_forward else "frozen_model",
        "profile": config.UFC_PROFILE,
        "window": "last_12_months" if last_year else "full",
        "start_date": events[0]["event_date"] if events else None,
        "end_date": events[-1]["event_date"] if events else None,
        "n_events": len(per_event),
        "n_events_with_bets": cards_with_bets,
        "bankroll_start": float(bankroll_start),
        "bankroll_final": final,
        "total_pnl": total_pnl,
        "roi_pct": 100.0 * roi,
        "roi_on_stake_pct": (100.0 * roi_on_stake) if roi_on_stake is not None else None,
        "n_tickets": len(settled),
        "n_tickets_incomplete": sum(1 for t in all_tickets if t.get("status") != "settled"),
        "hit_rate": (len(wins) / len(settled)) if settled else None,
        "max_drawdown_pct": 100.0 * max_dd,
        "avg_bets_per_card": avg_bets_per_card,
        "avg_bets_per_active_card": avg_bets_when_active,
        "total_stake": total_stake,
        "by_bet_type": by_type,
        "notes": notes,
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "monthly": monthly,
        "per_event": per_event,
        "tickets": all_tickets,
        "equity": equity,
        "wf_train_meta": wf_train_meta or [],
    }


def write_ha_backtest_csv(report: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report.get("tickets") or []).to_csv(out, index=False, encoding="utf-8")
    return out


def write_ha_backtest_summary_csv(report: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = report.get("summary") or {}
    rows = [{"metric": k, "value": v} for k, v in s.items() if k != "by_bet_type" and k != "notes"]
    for bt, stats in (s.get("by_bet_type") or {}).items():
        for k, v in stats.items():
            rows.append({"metric": f"{bt}.{k}", "value": v})
    for note in s.get("notes") or []:
        rows.append({"metric": "note", "value": note})
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    return out


def analyze_ticket_gate_mix(tickets: list[dict[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    """Summarize which kinds of bets pass HA gates (singles odds/edge/prob buckets)."""
    if isinstance(tickets, pd.DataFrame):
        rows = tickets.to_dict("records")
    else:
        rows = list(tickets or [])
    settled = [t for t in rows if str(t.get("status") or "") == "settled"]
    singles = [t for t in settled if str(t.get("bet_type") or "") == "single"]
    parlays = [t for t in settled if "parlay" in str(t.get("bet_type") or "").lower()]

    def _bucket_counts(items: list[dict[str, Any]], key: str, edges: list[tuple[str, float, float]]) -> dict[str, int]:
        out = {label: 0 for label, _, _ in edges}
        out["missing"] = 0
        for t in items:
            try:
                v = float(t.get(key))
            except (TypeError, ValueError):
                out["missing"] += 1
                continue
            if not np.isfinite(v):
                out["missing"] += 1
                continue
            placed = False
            for label, lo, hi in edges:
                if lo <= v < hi:
                    out[label] += 1
                    placed = True
                    break
            if not placed:
                out["missing"] += 1
        return out

    odds_buckets = _bucket_counts(
        singles,
        "odds",
        [
            ("heavy_fav_<1.40", 1.0, 1.40),
            ("fav_1.40_1.80", 1.40, 1.80),
            ("near_even_1.80_2.20", 1.80, 2.20),
            ("dog_>=2.20", 2.20, 50.0),
        ],
    )
    edge_buckets = _bucket_counts(
        singles,
        "edge",
        [
            ("edge_<0.10", -1.0, 0.10),
            ("edge_0.10_0.15", 0.10, 0.15),
            ("edge_0.15_0.25", 0.15, 0.25),
            ("edge_>=0.25", 0.25, 5.0),
        ],
    )
    prob_buckets = _bucket_counts(
        singles,
        "prob",
        [
            ("prob_<0.70", 0.0, 0.70),
            ("prob_0.70_0.85", 0.70, 0.85),
            ("prob_0.85_0.95", 0.85, 0.95),
            ("prob_>=0.95", 0.95, 1.01),
        ],
    )

    def _mean(items: list[dict[str, Any]], key: str) -> float | None:
        vals = []
        for t in items:
            try:
                v = float(t.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    fight_ids: set[str] = set()
    for t in singles:
        fid = t.get("fight_id")
        key = ""
        if fid is not None and str(fid).strip() and str(fid).strip().lower() not in {"nan", "none"}:
            try:
                key = str(int(float(fid)))
            except (TypeError, ValueError):
                key = str(fid).strip()
        if not key:
            key = str(t.get("fight") or "").strip()
        if key:
            fight_ids.add(key)
    return {
        "n_settled": len(settled),
        "n_singles": len(singles),
        "n_parlays": len(parlays),
        "singles_share": (len(singles) / len(settled)) if settled else None,
        "parlay_share": (len(parlays) / len(settled)) if settled else None,
        "avg_single_odds": _mean(singles, "odds"),
        "avg_single_edge": _mean(singles, "edge"),
        "avg_single_prob": _mean(singles, "prob"),
        "odds_buckets": odds_buckets,
        "edge_buckets": edge_buckets,
        "prob_buckets": prob_buckets,
        "single_fight_keys": sorted(fight_ids),
    }


def compare_ticket_gate_mix(
    baseline_mix: dict[str, Any] | None,
    current_mix: dict[str, Any] | None,
) -> dict[str, Any]:
    """Diff gate-mix summaries; identify fights newly admitted / dropped."""
    base = baseline_mix or {}
    cur = current_mix or {}
    base_fights = set(base.get("single_fight_keys") or [])
    cur_fights = set(cur.get("single_fight_keys") or [])
    return {
        "baseline": {k: v for k, v in base.items() if k != "single_fight_keys"},
        "current": {k: v for k, v in cur.items() if k != "single_fight_keys"},
        "fights_added": sorted(cur_fights - base_fights)[:40],
        "fights_dropped": sorted(base_fights - cur_fights)[:40],
        "n_fights_added": int(len(cur_fights - base_fights)),
        "n_fights_dropped": int(len(base_fights - cur_fights)),
        "n_fights_shared": int(len(base_fights & cur_fights)),
    }


def write_ha_backtest_html(report: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = report.get("summary") or {}
    monthly = report.get("monthly") or []
    per_event = report.get("per_event") or []

    def esc(x: Any) -> str:
        return html.escape(str(x))

    by_type_rows = ""
    for bt, st in (s.get("by_bet_type") or {}).items():
        hr = st.get("hit_rate")
        roi = st.get("roi")
        hr_s = f"{100 * hr:.1f}%" if hr is not None else "—"
        roi_s = f"{100 * roi:.1f}%" if roi is not None else "—"
        by_type_rows += (
            f"<tr><td>{esc(bt)}</td><td>{st.get('n')}</td><td>{st.get('wins')}</td>"
            f"<td>{hr_s}</td><td>${float(st.get('stake') or 0):.2f}</td>"
            f"<td>${float(st.get('pnl') or 0):+.2f}</td><td>{roi_s}</td></tr>\n"
        )

    monthly_rows = ""
    for m in monthly:
        monthly_rows += (
            f"<tr><td>{esc(m.get('period'))}</td><td>{m.get('events')}</td>"
            f"<td>{m.get('tickets')}</td><td>${float(m.get('pnl') or 0):+.2f}</td>"
            f"<td>${float(m.get('end_bankroll') or 0):.2f}</td></tr>\n"
        )

    eq_rows = ""
    for e in per_event[-40:]:
        eq_rows += (
            f"<tr><td>{esc(e.get('date'))}</td><td>{esc(e.get('event'))[:40]}</td>"
            f"<td>{e.get('n_tickets')}</td><td>${float(e.get('card_pnl') or 0):+.2f}</td>"
            f"<td>${float(e.get('bankroll') or 0):.2f}</td></tr>\n"
        )

    notes = "".join(f"<li>{esc(n)}</li>" for n in (s.get("notes") or []))
    hit = s.get("hit_rate")
    hit_s = f"{100 * hit:.1f}%" if hit is not None else "—"

    baseline = report.get("baseline_summary") or {}
    compare_block = ""
    if baseline:
        compare_title = str(
            report.get("baseline_label")
            or "vs prior walk-forward (before path-risk controls)"
        )

        def _fmt_metric(label: str, key: str, *, pct: bool = False, money: bool = False) -> str:
            cur = s.get(key)
            base = baseline.get(key)
            if cur is None and base is None:
                return ""

            def _v(x: Any) -> str:
                if x is None:
                    return "—"
                if money:
                    return f"${float(x):.2f}"
                if pct:
                    if key == "hit_rate":
                        return f"{100 * float(x):.1f}%"
                    return f"{float(x):+.1f}%"
                return str(x)

            delta = ""
            try:
                if cur is not None and base is not None:
                    d = float(cur) - float(base)
                    if money:
                        delta = f" ({d:+.2f})"
                    elif key == "hit_rate":
                        delta = f" ({100 * d:+.1f}pp)"
                    elif pct:
                        delta = f" ({d:+.1f})"
                    else:
                        delta = f" ({d:+.0f})"
            except (TypeError, ValueError):
                delta = ""
            return (
                f"<tr><td>{esc(label)}</td><td>{_v(base)}</td>"
                f"<td><b>{_v(cur)}</b>{esc(delta)}</td></tr>\n"
            )

        def _bt_cell(bucket: dict[str, Any]) -> str:
            n = int(bucket.get("n") or 0)
            pnl = float(bucket.get("pnl") or 0)
            hr = bucket.get("hit_rate")
            hr_s = f"{100 * float(hr):.0f}%" if hr is not None else "—"
            return f"${pnl:+.2f} / {n} / {hr_s}"

        b_bt = baseline.get("by_bet_type") or {}
        c_bt = s.get("by_bet_type") or {}
        compare_block = f"""
<div class="card">
<h2>{esc(compare_title)}</h2>
<p style="color:#94a3b8;font-size:.85rem">Baseline: {esc(report.get('baseline_path') or 'prior walk-forward')}</p>
<table><thead><tr><th>Metric</th><th>Prior controlled WF</th><th>Post-enrich WF</th></tr></thead>
<tbody>
{_fmt_metric("Final bankroll", "bankroll_final", money=True)}
{_fmt_metric("ROI (bankroll)", "roi_pct", pct=True)}
{_fmt_metric("ROI on stake", "roi_on_stake_pct", pct=True)}
{_fmt_metric("Hit rate", "hit_rate", pct=True)}
{_fmt_metric("Max drawdown", "max_drawdown_pct", pct=True)}
{_fmt_metric("Tickets", "n_tickets")}
<tr><td>single PnL / n / hit</td>
<td>{_bt_cell(b_bt.get("single") or {})}</td>
<td><b>{_bt_cell(c_bt.get("single") or {})}</b></td></tr>
<tr><td>parlay_2leg PnL / n / hit</td>
<td>{_bt_cell(b_bt.get("parlay_2leg") or {})}</td>
<td><b>{_bt_cell(c_bt.get("parlay_2leg") or {})}</b></td></tr>
</tbody></table>
</div>
"""

    gate_mix = report.get("gate_mix_compare") or {}
    gate_block = ""
    if gate_mix:
        cur_m = gate_mix.get("current") or {}
        base_m = gate_mix.get("baseline") or {}

        def _bucket_table(title: str, key: str) -> str:
            b = base_m.get(key) or {}
            c = cur_m.get(key) or {}
            labels = sorted(set(b) | set(c), key=lambda x: (x == "missing", x))
            rows = ""
            for lab in labels:
                bv, cv = int(b.get(lab) or 0), int(c.get(lab) or 0)
                rows += f"<tr><td>{esc(lab)}</td><td>{bv}</td><td><b>{cv}</b> ({cv - bv:+d})</td></tr>\n"
            return f"""
<h3 style="margin-top:12px">{esc(title)}</h3>
<table><thead><tr><th>Bucket</th><th>Prior</th><th>Post-enrich</th></tr></thead>
<tbody>{rows or '<tr><td colspan="3">—</td></tr>'}</tbody></table>
"""

        def _avg_line(label: str, key: str, *, pct: bool = False) -> str:
            bv, cv = base_m.get(key), cur_m.get(key)
            if bv is None and cv is None:
                return ""
            def _fmt(x: Any) -> str:
                if x is None:
                    return "—"
                return f"{100 * float(x):.1f}%" if pct else f"{float(x):.3f}"
            return f"<tr><td>{esc(label)}</td><td>{_fmt(bv)}</td><td><b>{_fmt(cv)}</b></td></tr>\n"

        added = gate_mix.get("fights_added") or []
        dropped = gate_mix.get("fights_dropped") or []
        added_s = ", ".join(esc(x)[:48] for x in added[:12]) or "—"
        dropped_s = ", ".join(esc(x)[:48] for x in dropped[:12]) or "—"
        gate_block = f"""
<div class="card">
<h2>Gate mix — which fights get through</h2>
<p style="color:#94a3b8;font-size:.85rem">
Singles/parlays and odds·edge·prob buckets for settled singles.
Fights shared {int(gate_mix.get('n_fights_shared') or 0)} ·
added {int(gate_mix.get('n_fights_added') or 0)} ·
dropped {int(gate_mix.get('n_fights_dropped') or 0)}.
</p>
<table><thead><tr><th>Mix</th><th>Prior</th><th>Post-enrich</th></tr></thead>
<tbody>
{_avg_line("Singles share", "singles_share", pct=True)}
{_avg_line("Parlay share", "parlay_share", pct=True)}
{_avg_line("Avg single odds", "avg_single_odds")}
{_avg_line("Avg single edge", "avg_single_edge")}
{_avg_line("Avg single prob", "avg_single_prob")}
<tr><td>n singles / parlays</td>
<td>{int(base_m.get('n_singles') or 0)} / {int(base_m.get('n_parlays') or 0)}</td>
<td><b>{int(cur_m.get('n_singles') or 0)} / {int(cur_m.get('n_parlays') or 0)}</b></td></tr>
</tbody></table>
{_bucket_table("Singles by odds", "odds_buckets")}
{_bucket_table("Singles by edge", "edge_buckets")}
{_bucket_table("Singles by model prob", "prob_buckets")}
<p style="font-size:.85rem;margin-top:12px"><b>Newly admitted (sample):</b> {added_s}</p>
<p style="font-size:.85rem"><b>Dropped (sample):</b> {dropped_s}</p>
</div>
"""

    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>HA {"Walk-Forward" if s.get("mode") == "walk_forward" else "Backtest"} — ${s.get('bankroll_start')} start</title>
<style>
body {{ font-family: Segoe UI, system-ui, sans-serif; background:#0f172a; color:#e2e8f0; margin:24px; }}
h1,h2 {{ color:#f8fafc; }}
.card {{ background:#1e293b; border-radius:12px; padding:16px 20px; margin:12px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }}
.metric {{ background:#0f172a; border-radius:8px; padding:12px; }}
.metric .v {{ font-size:1.4rem; font-weight:700; color:#34d399; }}
.metric .l {{ font-size:.75rem; color:#94a3b8; text-transform:uppercase; }}
table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
th,td {{ padding:6px 8px; border-bottom:1px solid #334155; text-align:left; }}
th {{ color:#94a3b8; }}
ul {{ color:#cbd5e1; }}
</style></head><body>
<h1>High-Accuracy {"Walk-Forward " if s.get("mode") == "walk_forward" else ""}Backtest</h1>
<p>{esc(s.get('window'))} · mode <b>{esc(s.get('mode') or 'frozen_model')}</b> · profile <b>{esc(s.get('profile'))}</b> ·
{esc(s.get('start_date'))} → {esc(s.get('end_date'))} ·
generated {esc(report.get('generated_at'))}</p>

<div class="card grid">
  <div class="metric"><div class="l">Final bankroll</div><div class="v">${float(s.get('bankroll_final') or 0):.2f}</div></div>
  <div class="metric"><div class="l">ROI</div><div class="v">{float(s.get('roi_pct') or 0):+.1f}%</div></div>
  <div class="metric"><div class="l">Tickets</div><div class="v">{int(s.get('n_tickets') or 0)}</div></div>
  <div class="metric"><div class="l">Hit rate</div><div class="v">{hit_s}</div></div>
  <div class="metric"><div class="l">Max drawdown</div><div class="v">{float(s.get('max_drawdown_pct') or 0):.1f}%</div></div>
  <div class="metric"><div class="l">Avg bets / card</div><div class="v">{float(s.get('avg_bets_per_card') or 0):.2f}</div></div>
  <div class="metric"><div class="l">Events</div><div class="v">{int(s.get('n_events') or 0)}</div></div>
  <div class="metric"><div class="l">Events w/ bets</div><div class="v">{int(s.get('n_events_with_bets') or 0)}</div></div>
</div>

{compare_block}

{gate_block}

<div class="card">
<h2>By bet type</h2>
<table><thead><tr><th>Type</th><th>N</th><th>Wins</th><th>Hit</th><th>Stake</th><th>PnL</th><th>ROI</th></tr></thead>
<tbody>{by_type_rows or '<tr><td colspan="7">No settled tickets</td></tr>'}</tbody></table>
</div>

<div class="card">
<h2>Monthly equity</h2>
<table><thead><tr><th>Month</th><th>Events</th><th>Tickets</th><th>PnL</th><th>End bankroll</th></tr></thead>
<tbody>{monthly_rows or '<tr><td colspan="5">—</td></tr>'}</tbody></table>
</div>

<div class="card">
<h2>Recent cards (equity path)</h2>
<table><thead><tr><th>Date</th><th>Event</th><th>Tickets</th><th>PnL</th><th>Bankroll</th></tr></thead>
<tbody>{eq_rows}</tbody></table>
</div>

<div class="card">
<h2>Notes</h2>
<ul>{notes}</ul>
</div>
</body></html>
"""
    out.write_text(body, encoding="utf-8")
    return out


def format_ha_backtest_summary(report: dict[str, Any]) -> str:
    s = report.get("summary") or {}
    hit = s.get("hit_rate")
    hit_line = (
        f"ROI {float(s.get('roi_pct') or 0):+.1f}%  |  "
        f"tickets {int(s.get('n_tickets') or 0)}  |  "
        f"hit {100 * float(hit):.1f}%"
        if hit is not None
        else f"ROI {float(s.get('roi_pct') or 0):+.1f}%  |  tickets {int(s.get('n_tickets') or 0)}  |  hit n/a"
    )
    lines = [
        "HIGH-ACCURACY WALK-FORWARD BACKTEST"
        if s.get("mode") == "walk_forward"
        else "HIGH-ACCURACY BACKTEST",
        f"Window: {s.get('start_date')} → {s.get('end_date')}  |  "
        f"mode={s.get('mode') or 'frozen_model'}  |  profile={s.get('profile')}",
        f"Start ${float(s.get('bankroll_start') or 0):.2f}  →  Final ${float(s.get('bankroll_final') or 0):.2f}",
        hit_line,
        f"Max DD {float(s.get('max_drawdown_pct') or 0):.1f}%  |  "
        f"avg bets/card {float(s.get('avg_bets_per_card') or 0):.2f}  |  "
        f"events {int(s.get('n_events') or 0)} ({int(s.get('n_events_with_bets') or 0)} with bets)",
        "",
        "By type:",
    ]
    for bt, st in (s.get("by_bet_type") or {}).items():
        hr = st.get("hit_rate")
        hr_s = f"{100 * hr:.0f}%" if hr is not None else "n/a"
        lines.append(
            f"  {bt}: n={st.get('n')} hit={hr_s} pnl=${float(st.get('pnl') or 0):+.2f}"
        )
    if report.get("baseline_summary"):
        b = report["baseline_summary"]
        label = report.get("baseline_label") or "vs baseline"
        lines.append("")
        lines.append(f"{label}:")
        lines.append(
            f"  bankroll ${float(b.get('bankroll_final') or 0):.2f} → "
            f"${float(s.get('bankroll_final') or 0):.2f}  |  "
            f"ROI-stake {float(b.get('roi_on_stake_pct') or 0):.1f}% → "
            f"{float(s.get('roi_on_stake_pct') or 0):.1f}%  |  "
            f"hit {100 * float(b.get('hit_rate') or 0):.1f}% → "
            f"{100 * float(s.get('hit_rate') or 0):.1f}%  |  "
            f"maxDD {float(b.get('max_drawdown_pct') or 0):.1f}% → "
            f"{float(s.get('max_drawdown_pct') or 0):.1f}%  |  "
            f"tickets {int(b.get('n_tickets') or 0)} → {int(s.get('n_tickets') or 0)}"
        )
    gate = report.get("gate_mix_compare") or {}
    if gate:
        lines.append(
            f"  gate fights shared={gate.get('n_fights_shared')} "
            f"added={gate.get('n_fights_added')} dropped={gate.get('n_fights_dropped')}"
        )
    if report.get("html_path"):
        lines.append("")
        lines.append(f"HTML: {report['html_path']}")
    if report.get("csv_path"):
        lines.append(f"CSV:  {report['csv_path']}")
    return "\n".join(lines)


def save_ha_backtest_reports(
    report: dict[str, Any],
    *,
    stamp: str | None = None,
    prefix: str | None = None,
    baseline_path: Path | str | None = None,
    baseline_label: str | None = None,
    baseline_tickets_path: Path | str | None = None,
) -> dict[str, Path]:
    stamp = stamp or datetime.now().strftime("%Y%m%d")
    reports = config.ROOT_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    mode = str((report.get("summary") or {}).get("mode") or "")
    walk_forward = mode == "walk_forward"
    if prefix:
        use_prefix = prefix
    elif walk_forward:
        use_prefix = "ha_walkforward_drawdown_controls"
    else:
        use_prefix = "ha_backtest_1yr_100start"
    html_path = reports / f"{use_prefix}_{stamp}.html"
    csv_path = reports / f"{use_prefix}_{stamp}_tickets.csv"
    summary_path = reports / f"{use_prefix}_{stamp}_summary.csv"

    # Attach baseline comparison when available
    resolved_baseline: Path | None = None
    if baseline_path:
        resolved_baseline = Path(baseline_path)
    elif walk_forward and use_prefix == "ha_walkforward_post_enrich":
        cands = sorted(reports.glob("ha_walkforward_drawdown_controls_*_summary.json"))
        resolved_baseline = cands[-1] if cands else None
        baseline_label = baseline_label or (
            "vs prior controlled walk-forward (path-risk HA)"
        )
    elif walk_forward:
        bp = reports / f"ha_walkforward_1yr_100start_{stamp}_summary.json"
        if not bp.is_file():
            cands = sorted(reports.glob("ha_walkforward_1yr_100start_*_summary.json"))
            bp = cands[-1] if cands else bp
        resolved_baseline = bp if bp.is_file() else None

    if resolved_baseline and resolved_baseline.is_file() and walk_forward:
        try:
            baseline = json.loads(resolved_baseline.read_text(encoding="utf-8"))
            report["baseline_summary"] = baseline.get("summary") or baseline
            report["baseline_path"] = str(resolved_baseline)
            if baseline_label:
                report["baseline_label"] = baseline_label
        except Exception:
            pass

    # Gate-mix comparison vs baseline tickets CSV
    cur_mix = analyze_ticket_gate_mix(report.get("tickets") or [])
    report["gate_mix"] = {k: v for k, v in cur_mix.items() if k != "single_fight_keys"}
    base_tickets_path: Path | None = None
    if baseline_tickets_path:
        base_tickets_path = Path(baseline_tickets_path)
    elif resolved_baseline is not None:
        guess = Path(str(resolved_baseline).replace("_summary.json", "_tickets.csv"))
        if guess.is_file():
            base_tickets_path = guess
    if base_tickets_path and base_tickets_path.is_file():
        try:
            base_df = pd.read_csv(base_tickets_path)
            base_mix = analyze_ticket_gate_mix(base_df)
            report["gate_mix_compare"] = compare_ticket_gate_mix(base_mix, cur_mix)
        except Exception as exc:
            logger.info("Gate-mix compare skipped: %s", exc)

    if "post_enrich" in use_prefix or "schema" in use_prefix:
        notes = list((report.get("summary") or {}).get("notes") or [])
        notes.append(
            "Post-enrich features: Sherdog history, SOS, Greco/CompuBox-style striking, "
            "prior-sport tiers (schema v4) under the same path-risk HA controls."
        )
        if report.get("summary") is not None:
            report["summary"]["notes"] = notes

    write_ha_backtest_html(report, html_path)
    write_ha_backtest_csv(report, csv_path)
    write_ha_backtest_summary_csv(report, summary_path)
    json_path = reports / f"{use_prefix}_{stamp}_summary.json"
    gate_cmp = report.get("gate_mix_compare") or {}
    json_path.write_text(
        json.dumps(
            {
                "summary": report.get("summary"),
                "monthly": report.get("monthly"),
                "wf_train_meta": report.get("wf_train_meta"),
                "baseline_summary": report.get("baseline_summary"),
                "baseline_path": report.get("baseline_path"),
                "baseline_label": report.get("baseline_label"),
                "gate_mix": report.get("gate_mix"),
                "gate_mix_compare": gate_cmp,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    report["html_path"] = str(html_path)
    report["csv_path"] = str(csv_path)
    report["summary_csv_path"] = str(summary_path)
    report["json_path"] = str(json_path)
    return {
        "html": html_path,
        "csv": csv_path,
        "summary_csv": summary_path,
        "json": json_path,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.project_paths import bootstrap

    bootstrap(entry_file=config.ROOT_DIR / "main.py")
    p = argparse.ArgumentParser(description="High-accuracy 1-year backtest")
    p.add_argument("--strategy", default="high-accuracy", choices=["high-accuracy", "ha"])
    p.add_argument("--bankroll", type=float, default=100.0)
    p.add_argument("--last-year", action="store_true", default=True)
    p.add_argument("--no-last-year", action="store_true")
    p.add_argument(
        "--walk-forward",
        action="store_true",
        help="True expanding-window walk-forward (retrain before each card; no future leakage)",
    )
    p.add_argument("--profile", choices=["paper", "live", "research"], default="paper")
    p.add_argument("--dynamic-thresholds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--prefix",
        default=None,
        help="Report filename prefix (default: ha_walkforward_drawdown_controls)",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline *_summary.json for side-by-side comparison",
    )
    p.add_argument(
        "--baseline-label",
        default=None,
        help="HTML heading for the comparison block",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    last_year = not bool(args.no_last_year)
    report = run_ha_backtest(
        bankroll_start=float(args.bankroll),
        last_year=last_year,
        use_dynamic_thresholds=bool(args.dynamic_thresholds),
        profile=str(args.profile),
        walk_forward=bool(args.walk_forward),
    )
    save_ha_backtest_reports(
        report,
        prefix=args.prefix,
        baseline_path=args.baseline,
        baseline_label=args.baseline_label,
    )
    print(format_ha_backtest_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
