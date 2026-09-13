"""
Prop-only accuracy / ROI backtest — separate from ML moneyline backtest.

Uses method/round settlement + ``method_probs_from_row`` (L5 / decision_finish_share
when present). Does not touch FEATURE_COLUMNS or ensemble_winner.joblib.

Outputs under ``data/reports/`` only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

REPORTS = Path(config.DATA_DIR) / "reports"

# Markets scored for accuracy (display + research). HA betting path stays Over 1.5.
SCORE_MARKETS: tuple[str, ...] = (
    "over_1_5_rounds",
    "under_1_5_rounds",
    "goes_to_decision",
    "finish",
    "ko_tko",
    "submission",
    "round_1_finish",
    "fighter_ko",
    "fighter_sub",
)


def _year_mask(df: pd.DataFrame, year: int) -> pd.Series:
    date_col = config.DATE_COLUMN if config.DATE_COLUMN in df.columns else None
    if date_col is None:
        for c in ("date", "event_date", "fight_date"):
            if c in df.columns:
                date_col = c
                break
    if date_col is None:
        return pd.Series(True, index=df.index)
    dts = pd.to_datetime(df[date_col], errors="coerce")
    return dts.dt.year == int(year)


def _attach_props_engine_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Optional props-only enrichment; never enables FEATURE_COLUMNS flags."""
    out = df
    need_dec = any(
        c not in out.columns
        for c in (
            "f1_decision_finish_share_l5",
            "f1_dec_win_rate_l5",
            "f2_decision_finish_share_l5",
        )
    )
    if need_dec:
        try:
            from src.decision_profile import attach_decision_profile_to_wide

            out = attach_decision_profile_to_wide(out)
            logger.info("Attached decision_profile columns for prop engine (props-only)")
        except Exception as exc:
            logger.warning("decision_profile attach skipped: %s", exc)

    need_path = any(
        c not in out.columns or out[c].isna().all()
        for c in ("f1_ko_win_rate_l5", "f1_r1_finish_rate_l5", "f1_sub_win_rate_l5")
        if True
    )
    # Always try pathway attach when r1 missing (common on processed CSV).
    if "f1_r1_finish_rate_l5" not in out.columns or out["f1_r1_finish_rate_l5"].isna().all():
        need_path = True
    if need_path:
        try:
            from src.pathway_features import attach_pathway_rates_for_props

            out = attach_pathway_rates_for_props(out)
            logger.info("Attached pathway L5 rates for prop engine (props-only)")
        except Exception as exc:
            logger.warning("pathway props attach skipped: %s", exc)
    return out


def load_prop_holdout(*, year: int = 2025) -> pd.DataFrame:
    """Labeled fights for prop settlement in ``year`` (leakage-safe as-of rates)."""
    from src.prop_performance import load_labeled_prop_frame

    feats = load_labeled_prop_frame()
    if feats.empty:
        return feats
    feats = _attach_props_engine_fields(feats)
    mask = _year_mask(feats, year)
    out = feats.loc[mask].copy()
    if out.empty:
        logger.warning("No fights for year=%s; using full labeled frame", year)
        out = feats.copy()
    return out.reset_index(drop=True)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if dd.notna().any() else 0.0


def score_prop_singles(
    features: pd.DataFrame,
    *,
    markets: tuple[str, ...] | None = None,
    min_model_prob_bet: float | None = None,
    flat_stake: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Score every market: prediction hit rate + flat synthetic ROI when model_p >= floor.

    Returns (trades_df, by_market_summary, overall).
    """
    from src.props import (
        PROP_MARKET_LABELS,
        method_probs_from_row,
        prop_model_prob,
        settle_prop,
        synthetic_market_odds,
    )

    keys = list(markets or SCORE_MARKETS)
    keys = [k for k in keys if k in PROP_MARKET_LABELS]
    min_p = float(
        min_model_prob_bet
        if min_model_prob_bet is not None
        else max(float(getattr(config, "PROP_MIN_MODEL_PROB", 0.55) or 0.55), 0.55)
    )

    trade_rows: list[dict[str, Any]] = []
    bankroll = 1000.0
    pred_hits = 0
    pred_n = 0
    mkt_pred_n: dict[str, int] = {k: 0 for k in keys}
    mkt_pred_hits: dict[str, int] = {k: 0 for k in keys}

    for _, row in features.iterrows():
        probs = method_probs_from_row(row)
        for key in keys:
            actual = settle_prop(key, row)
            if actual is None:
                continue
            model_p = float(prop_model_prob(key, row, probs))
            pred_yes = model_p >= 0.5
            hit = int(pred_yes == bool(actual))
            pred_hits += hit
            pred_n += 1
            mkt_pred_n[key] += 1
            mkt_pred_hits[key] += hit

            if model_p < min_p:
                continue
            odds = synthetic_market_odds(model_p)
            won = bool(actual)
            pnl = flat_stake * (odds - 1.0) if won else -flat_stake
            bankroll += pnl
            trade_rows.append(
                {
                    "fight_id": row.get(config.FIGHT_ID_COLUMN, row.get("fight_id")),
                    "date": row.get(config.DATE_COLUMN, row.get("date")),
                    "fighter_1": row.get("fighter_1", row.get("fighter1")),
                    "fighter_2": row.get("fighter_2", row.get("fighter2")),
                    "prop_key": key,
                    "model_prob": model_p,
                    "odds": odds,
                    "odds_source": "synthetic",
                    "stake": flat_stake,
                    "won": int(won),
                    "pnl": pnl,
                    "equity": bankroll,
                    "pred_hit": hit,
                }
            )

    trades = pd.DataFrame(trade_rows)
    by_rows: list[dict[str, Any]] = []
    for key in keys:
        n_k = mkt_pred_n[key]
        hits_k = mkt_pred_hits[key]
        sub = trades[trades["prop_key"] == key] if not trades.empty else pd.DataFrame()
        n_bets = len(sub)
        bet_hr = float(sub["won"].mean()) if n_bets else None
        stake = float(sub["stake"].sum()) if n_bets else 0.0
        pnl = float(sub["pnl"].sum()) if n_bets else None
        if bet_hr is not None and not np.isfinite(bet_hr):
            bet_hr = None
        if pnl is not None and not np.isfinite(pnl):
            pnl = None
        by_rows.append(
            {
                "prop_key": key,
                "prop_label": PROP_MARKET_LABELS.get(key, key),
                "n_predictions": n_k,
                "pred_hit_rate": (hits_k / n_k) if n_k else None,
                "n_bets": n_bets,
                "bet_hit_rate": bet_hr,
                "pnl": pnl,
                "stake": stake if stake else None,
                "roi": (pnl / stake) if pnl is not None and stake > 0 else None,
                "odds_source": "synthetic",
            }
        )

    by_market = pd.DataFrame(by_rows)
    overall = {
        "n_fights": int(len(features)),
        "n_predictions": pred_n,
        "mean_pred_hit_rate": (pred_hits / pred_n) if pred_n else None,
        "n_bets": int(len(trades)),
        "bet_hit_rate": float(trades["won"].mean()) if not trades.empty else None,
        "pnl": float(trades["pnl"].sum()) if not trades.empty else None,
        "roi": (
            float(trades["pnl"].sum() / trades["stake"].sum())
            if not trades.empty and float(trades["stake"].sum()) > 0
            else None
        ),
        "max_dd": _max_drawdown(trades["equity"]) if not trades.empty else None,
        "min_model_prob_bet": min_p,
        "flat_stake": flat_stake,
        "odds_source": "synthetic",
    }
    return trades, by_market, overall


def score_mixed_parlays(
    features: pd.DataFrame,
    *,
    books: tuple[str, ...] = ("DraftKings", "MyBookie"),
    flat_stake: float = 10.0,
    max_parlays_per_card: int = 3,
    min_leg_prob: float = 0.55,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Lightweight research parlays for books that allow them (DK / MyBookie).

    Prop-only 2-leg same-card Over 1.5 combinations (cross-fight independence).
    Avoids full HA moneyline extraction (too slow / noisy for holdout reports).
    Live tickets still keep PROP_PARLAYS_ENABLED=False.
    """
    from itertools import combinations

    from src.props import method_probs_from_row, settle_prop, synthetic_market_odds

    event_col = next(
        (c for c in ("event_name", "event", "Event") if c in features.columns),
        None,
    )
    date_col = next(
        (
            c
            for c in (config.DATE_COLUMN, "event_date", "date", "fight_date")
            if c in features.columns
        ),
        None,
    )
    summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for book in books:
        rules = config.BOOK_PROP_RULES.get(book, {})
        if not rules.get("allow_prop_parlays"):
            summaries[book] = {"trades": 0.0, "skipped": "book_singles_only"}
            continue

        bankroll = 1000.0
        n_trades = 0
        wins = 0
        pnls: list[float] = []

        work = features
        if event_col is not None:
            groups = list(work.groupby(event_col, dropna=False))
        elif date_col is not None:
            dts = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
            groups = list(work.groupby(dts, dropna=False))
        else:
            groups = [("card", work)]

        for event_key, card_df in groups:
            legs: list[dict[str, Any]] = []
            for _, row in card_df.iterrows():
                probs = method_probs_from_row(row)
                p = float(probs.get("over_1_5_rounds", 0.0))
                if p < min_leg_prob:
                    continue
                settled = settle_prop("over_1_5_rounds", row)
                if settled is None:
                    continue
                legs.append(
                    {
                        "fight_id": str(row.get("fight_id", "")),
                        "prob": p,
                        "odds": synthetic_market_odds(p),
                        "won": bool(settled),
                    }
                )
            if len(legs) < 2:
                continue
            # Distinct fights only; take top-EV 2-leg combos
            scored: list[tuple[float, dict, dict]] = []
            for a, b in combinations(legs, 2):
                if a["fight_id"] == b["fight_id"] and a["fight_id"]:
                    continue
                combined_p = float(a["prob"] * b["prob"])  # cross-fight independence
                combined_odds = float(a["odds"] * b["odds"])
                ev = combined_p * (combined_odds - 1.0) - (1.0 - combined_p)
                scored.append((ev, a, b))
            scored.sort(key=lambda t: t[0], reverse=True)
            # Synthetic vig makes fair EV negative; take top-EV combos for sample metrics.
            for ev, a, b in scored[:max_parlays_per_card]:
                combined_odds = float(a["odds"] * b["odds"])
                combined_p = float(a["prob"] * b["prob"])
                won = bool(a["won"] and b["won"])
                pnl = flat_stake * (combined_odds - 1.0) if won else -flat_stake
                bankroll += pnl
                n_trades += 1
                wins += int(won)
                pnls.append(pnl)
                all_rows.append(
                    {
                        "event": event_key,
                        "book": book,
                        "market_type": "parlay",
                        "n_legs": 2,
                        "prop_keys": "over_1_5_rounds+over_1_5_rounds",
                        "combined_odds": combined_odds,
                        "combined_prob": combined_p,
                        "correlation_adjusted": 0,
                        "won": int(won),
                        "stake": flat_stake,
                        "pnl": pnl,
                        "equity": bankroll,
                        "odds_source": "synthetic",
                    }
                )

        summaries[book] = {
            "trades": float(n_trades),
            "hit_rate": (wins / n_trades) if n_trades else None,
            "total_pnl": float(sum(pnls)) if pnls else None,
            "roi_pct": (
                float(100.0 * sum(pnls) / (flat_stake * n_trades)) if n_trades else None
            ),
            "note": "prop-only 2-leg Over 1.5; cross-fight independence",
        }

    if not all_rows:
        return pd.DataFrame(), summaries
    return pd.DataFrame(all_rows), summaries


def coverage_decision_fields(features: pd.DataFrame) -> dict[str, float]:
    cols = [
        "f1_finish_rate_l5",
        "f1_r1_finish_rate_l5",
        "f1_ko_win_rate_l5",
        "f1_sub_win_rate_l5",
        "f1_dec_win_rate_l5",
        "f1_decision_finish_share_l5",
        "f1_distance_rate_l5",
        "f2_finish_rate_l5",
        "f2_r1_finish_rate_l5",
        "f2_ko_win_rate_l5",
        "f2_sub_win_rate_l5",
        "f2_dec_win_rate_l5",
        "f2_decision_finish_share_l5",
        "f2_distance_rate_l5",
    ]
    out: dict[str, float] = {}
    n = max(len(features), 1)
    for c in cols:
        if c in features.columns:
            out[c] = float(features[c].notna().mean())
        else:
            out[c] = 0.0
    out["n"] = float(n)
    return out


def run_props_accuracy_2025(
    *,
    year: int = 2025,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Full prop accuracy run; writes CSV/MD/JSON under data/reports/."""
    out = Path(out_dir) if out_dir else REPORTS
    out.mkdir(parents=True, exist_ok=True)

    # Master switch for prop engine paths that check ENABLE_PROPS
    prev = bool(config.ENABLE_PROPS)
    config.ENABLE_PROPS = True

    try:
        features = load_prop_holdout(year=year)
        trades, by_market, overall = score_prop_singles(features)
        parlay_trades, parlay_summary = score_mixed_parlays(features)
        coverage = coverage_decision_fields(features)

        trades_path = out / "prop_trades.csv"
        summary_csv = out / "prop_accuracy_by_market.csv"
        parlay_path = out / "mixed_parlay_trades.csv"
        md_path = out / f"props_accuracy_{year}.md"
        json_path = out / f"props_accuracy_{year}.json"

        if not trades.empty:
            trades.to_csv(trades_path, index=False)
        else:
            pd.DataFrame().to_csv(trades_path, index=False)
        by_market.to_csv(summary_csv, index=False)
        if not parlay_trades.empty:
            parlay_trades.to_csv(parlay_path, index=False)
        else:
            pd.DataFrame().to_csv(parlay_path, index=False)

        # Tuned vs keep-as-is note
        o15 = by_market[by_market["prop_key"] == "over_1_5_rounds"]
        o15_hr = float(o15["pred_hit_rate"].iloc[0]) if len(o15) and o15["pred_hit_rate"].notna().any() else None
        verdict = "tuned_props_engine"
        note = (
            "Prop engine uses pathway L5 method win/loss + R1/distance, finish_rate_l5, "
            "and decision_finish_share / dec_win|loss when present (props-only attach). "
            "Not added to LightGBM FEATURE_COLUMNS (decision_profile / pathway A/Bs were DROP). "
            "Live HA still bets Over 1.5 only; correlation adjustment unchanged. "
            "Live tickets keep PROP_PARLAYS_ENABLED=False. "
            "Research parlays: DK/MyBookie 2-leg Over 1.5 (cross-fight); BetNow singles-only."
        )

        flags = {
            "ENABLE_PROPS": "forced_true_for_report",
            "ADD_DECISION_PROFILE_TO_FEATURES": False,
            "ENABLE_PATHWAY_FEATURES": bool(getattr(config, "ENABLE_PATHWAY_FEATURES", False)),
            "ENABLE_MARKET_FEATURES": bool(getattr(config, "ENABLE_MARKET_FEATURES", False)),
            "ADD_OVERSEAS_FEATURES": bool(getattr(config, "ADD_OVERSEAS_FEATURES", False)),
            "ADD_HOME_COUNTRY_TO_FEATURES": False,
            "ENABLE_HIGH_VALUE_FEATURES": bool(getattr(config, "ENABLE_HIGH_VALUE_FEATURES", True)),
        }

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "year": year,
            "verdict": verdict,
            "note": note,
            "overall": overall,
            "by_market": by_market.to_dict(orient="records"),
            "parlay_summary": parlay_summary,
            "coverage": coverage,
            "flags": flags,
            "paths": {
                "prop_trades": str(trades_path),
                "by_market": str(summary_csv),
                "mixed_parlay_trades": str(parlay_path),
                "md": str(md_path),
                "json": str(json_path),
            },
            "over_1_5_pred_hit_rate": o15_hr,
            "book_rules": {
                k: dict(v) for k, v in config.BOOK_PROP_RULES.items()
            },
            "correlation_policy": (
                "same-fight ML+method joints via method_probs; "
                "multi-prop same fight discounted by PROP_CORRELATION_DISCOUNT; "
                "cross-fight independence"
            ),
        }
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        md_path.write_text(_format_md(payload), encoding="utf-8")
        logger.info("Wrote %s and %s", md_path, json_path)
        return payload
    finally:
        config.ENABLE_PROPS = prev


def _format_md(payload: dict[str, Any]) -> str:
    year = payload.get("year")
    ov = payload.get("overall") or {}
    lines = [
        f"# Props accuracy — {year}",
        "",
        "UFC-only. Separate from ML moneyline backtest. No FEATURE_COLUMNS / winner retrain.",
        "",
        f"**Verdict:** `{payload.get('verdict')}` — keep prop engine tuned; leave decision_profile out of ML features.",
        "",
        payload.get("note", ""),
        "",
        "## Overall (synthetic odds)",
        "",
        f"- Fights: {ov.get('n_fights')}",
        f"- Predictions scored: {ov.get('n_predictions')}",
        f"- Mean pred hit rate: {_pct(ov.get('mean_pred_hit_rate'))}",
        f"- Bets (model_p≥{100*float(ov.get('min_model_prob_bet') or 0):.0f}%): {ov.get('n_bets')}",
        f"- Bet hit rate: {_pct(ov.get('bet_hit_rate'))}",
        f"- Flat ROI: {_pct(ov.get('roi'))}",
        f"- Max DD: {_pct(ov.get('max_dd'))}",
        f"- Odds source: **synthetic** (historical live prop lines unavailable)",
        "",
        "## By market",
        "",
        "| Market | n_pred | pred hit | n_bets | bet hit | ROI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in payload.get("by_market") or []:
        lines.append(
            f"| {r.get('prop_label')} | {r.get('n_predictions')} | {_pct(r.get('pred_hit_rate'))} | "
            f"{r.get('n_bets')} | {_pct(r.get('bet_hit_rate'))} | {_pct(r.get('roi'))} |"
        )
    lines.extend(
        [
            "",
            "## Mixed parlays (research; DK / MyBookie book rules)",
            "",
        ]
    )
    for book, s in (payload.get("parlay_summary") or {}).items():
        lines.append(
            f"- **{book}**: trades={s.get('trades', 0)}, "
            f"hit={_pct(s.get('hit_rate'))}, "
            f"ROI={_fmt_roi_pct(s)}"
        )
    lines.extend(
        [
            "",
            "Live HA: `PROP_PARLAYS_ENABLED=false` (no parlays on tickets). BetNow remains singles-only.",
            "",
            "## Props-engine field coverage",
            "",
            "| Column | nonnull% |",
            "|---|---:|",
        ]
    )
    for k, v in (payload.get("coverage") or {}).items():
        if k == "n":
            continue
        lines.append(f"| `{k}` | {100*float(v):.1f}% |")
    lines.extend(
        [
            "",
            "## Flags (must stay off for ML)",
            "",
        ]
    )
    for k, v in (payload.get("flags") or {}).items():
        lines.append(f"- `{k}` = `{v}`")
    roi = ov.get("roi")
    o15_rows = [
        r for r in (payload.get("by_market") or []) if r.get("prop_key") == "over_1_5_rounds"
    ]
    o15_roi = o15_rows[0].get("roi") if o15_rows else None
    lines.extend(
        [
            "",
            "## Keep-as-is vs tuned",
            "",
            "- **Tuned (shipped):** `method_probs_from_row` blends pathway L5 KO/sub/dec win-loss, "
            "R1/distance rates, finish_rate_l5, and decision_finish_share for props + display.",
            "- **Keep out of ML:** pathway / decision_profile / home / overseas flags stay off "
            "(FEATURE_COLUMNS unchanged).",
            "- **Live betting:** Over 1.5 only (HA); live odds first, synthetic second; Source column labeled.",
            f"- **Synthetic ROI caveat:** flat {_pct(roi)} overall / {_pct(o15_roi)} Over 1.5 under "
            "vigged synthetic lines — not a live-edge claim. Prefer live book lines for staking.",
            "- **Rare markets** (fighter KO/sub) show high pred hit mainly from predicting No "
            "(majority class); few clear ≥78% bets.",
            "",
            "## Correlation",
            "",
            str(payload.get("correlation_policy") or ""),
            "",
            f"Artifacts: `{payload.get('paths', {}).get('prop_trades')}`, "
            f"`{payload.get('paths', {}).get('mixed_parlay_trades')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(x: Any) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{100 * v:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_roi_pct(summary: dict[str, Any]) -> str:
    """simulate_mixed_parlays returns roi_pct already in percent points."""
    if summary.get("roi_pct") is not None:
        try:
            return f"{float(summary['roi_pct']):+.1f}%"
        except (TypeError, ValueError):
            pass
    return _pct(summary.get("roi"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Prop-only accuracy report (data/reports/)")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("-o", "--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    payload = run_props_accuracy_2025(year=args.year, out_dir=args.out_dir)
    print(_format_md(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
