"""
Historical prop performance (Over/Under 1.5 rounds + other prop markets).

Sources (merged, de-duplicated):
  1. Settled rows in ``data/bet_journal.csv`` / ``data/prediction_bank.csv`` when
     ``prop_type`` is a prop market
  2. Labeled feature matrix fights with ``method`` + ``round`` (model probs via
     ``method_probs_from_row`` / ``settle_prop``)

Re-run::

    python scripts/run_prop_performance.py
    python -m src.prop_performance
    python -m main --prop-performance
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Focus markets always reported first
FOCUS_PROPS = ("over_1_5_rounds", "under_1_5_rounds")

_PROP_ALIASES: dict[str, str] = {
    "over_1.5": "over_1_5_rounds",
    "over 1.5": "over_1_5_rounds",
    "over_1_5": "over_1_5_rounds",
    "over15": "over_1_5_rounds",
    "o1.5": "over_1_5_rounds",
    "under_1.5": "under_1_5_rounds",
    "under 1.5": "under_1_5_rounds",
    "under_1_5": "under_1_5_rounds",
    "under15": "under_1_5_rounds",
    "u1.5": "under_1_5_rounds",
    "r1_finish": "round_1_finish",
    "round1_finish": "round_1_finish",
    "decision": "goes_to_decision",
    "goes_decision": "goes_to_decision",
    "inside_distance": "finish",
    "ko": "ko_tko",
    "tko": "ko_tko",
    "sub": "submission",
}


def normalize_prop_key(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text or text in ("nan", "none", "null", "moneyline", "ml", ""):
        return ""
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    if text in _PROP_ALIASES:
        return _PROP_ALIASES[text]
    # fuzzy contains
    if "over" in text and "1" in text and "5" in text:
        return "over_1_5_rounds"
    if "under" in text and "1" in text and "5" in text:
        return "under_1_5_rounds"
    return text


def prop_display_name(key: str) -> str:
    from src.props import PROP_MARKET_LABELS

    return PROP_MARKET_LABELS.get(key, key.replace("_", " ").title())


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if pd.isna(val):
            return None
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def load_settled_prop_rows_from_logs() -> pd.DataFrame:
    """Pull settled prop bets from bet_journal + prediction_bank when present."""
    rows: list[dict[str, Any]] = []

    journal_path = Path(getattr(config, "BET_JOURNAL_CSV", config.DATA_DIR / "bet_journal.csv"))
    if journal_path.is_file():
        try:
            j = pd.read_csv(journal_path, low_memory=False)
            for _, r in j.iterrows():
                key = normalize_prop_key(r.get("prop_type"))
                if not key:
                    # notes / pick sometimes encode prop
                    key = normalize_prop_key(r.get("notes")) or normalize_prop_key(r.get("pick"))
                if not key or key == "moneyline":
                    continue
                correct = r.get("correct")
                settled = r.get("settlement_complete")
                won = None
                if pd.notna(correct):
                    try:
                        won = bool(int(float(correct)))
                    except (TypeError, ValueError):
                        won = str(correct).strip().lower() in ("1", "true", "yes", "w", "win")
                elif pd.notna(settled) and str(settled).strip().lower() in ("1", "true", "yes"):
                    # settled but missing correct — skip fail-closed
                    continue
                else:
                    continue
                rows.append(
                    {
                        "source": "bet_journal",
                        "prop_key": key,
                        "fight": str(r.get("fight") or ""),
                        "event": str(r.get("event") or ""),
                        "model_prob": _safe_float(r.get("model_prob")),
                        "opening_odds": _safe_float(r.get("opening_odds")),
                        "stake": _safe_float(r.get("stake")) or float(getattr(config, "FLAT_STAKE", 10) or 10),
                        "won": int(won) if won is not None else None,
                        "pnl": _safe_float(r.get("pnl")),
                        "predicted_yes": 1,
                    }
                )
        except Exception as exc:
            logger.warning("bet_journal prop load failed: %s", exc)

    bank_path = Path(getattr(config, "PREDICTION_BANK_CSV", config.DATA_DIR / "prediction_bank.csv"))
    if bank_path.is_file():
        try:
            b = pd.read_csv(bank_path, low_memory=False)
            for _, r in b.iterrows():
                key = normalize_prop_key(r.get("prop_type") or r.get("prop_key"))
                if not key or key == "moneyline":
                    continue
                if pd.isna(r.get("correct")) and pd.isna(r.get("actual_winner")):
                    continue
                won = None
                if pd.notna(r.get("correct")):
                    try:
                        won = bool(int(float(r["correct"])))
                    except (TypeError, ValueError):
                        won = str(r["correct"]).strip().lower() in ("1", "true", "yes")
                rows.append(
                    {
                        "source": "prediction_bank",
                        "prop_key": key,
                        "fight": str(r.get("fight") or f"{r.get('fighter_1')} vs {r.get('fighter_2')}"),
                        "event": str(r.get("event") or ""),
                        "model_prob": _safe_float(r.get("pick_prob") or r.get("model_prob")),
                        "opening_odds": _safe_float(r.get("odds")),
                        "stake": float(getattr(config, "FLAT_STAKE", 10) or 10),
                        "won": int(won) if won is not None else None,
                        "pnl": None,
                        "predicted_yes": 1,
                    }
                )
        except Exception as exc:
            logger.warning("prediction_bank prop load failed: %s", exc)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_labeled_prop_frame() -> pd.DataFrame:
    """Labeled fights with method/round for prop settlement."""
    from src.data_loader import load_fights, load_processed_features

    try:
        feats = load_processed_features()
    except Exception:
        feats = pd.DataFrame()
    if feats is None or feats.empty:
        fights = load_fights()
        from src.feature_engineering import build_feature_matrix

        feats = build_feature_matrix(fights, keep_unlabeled=False)

    work = feats.copy()
    # Ensure method/round from fights when missing on many rows
    if "method" not in work.columns or work["method"].isna().mean() > 0.3:
        try:
            fights = load_fights()
            keep = [c for c in ("fight_id", "method", "round", "winner") if c in fights.columns]
            if "fight_id" in keep and "fight_id" in work.columns:
                meta = fights[keep].drop_duplicates("fight_id", keep="last")
                work = work.drop(columns=[c for c in ("method", "round") if c in work.columns], errors="ignore")
                work = work.merge(meta, on="fight_id", how="left", suffixes=("", "_f"))
        except Exception as exc:
            logger.debug("method merge skipped: %s", exc)

    if config.TARGET_COLUMN in work.columns:
        work = work.dropna(subset=[config.TARGET_COLUMN])
    # Need round for O/U settlement
    if "round" in work.columns:
        work = work[work["round"].notna()].copy()
    return work.reset_index(drop=True)


def _evaluate_history_prop(
    features: pd.DataFrame,
    prop_key: str,
    *,
    flat_stake: float,
    min_model_prob_bet: float,
) -> dict[str, Any]:
    """Score one prop market across labeled history."""
    from src.props import prop_model_prob, settle_prop, synthetic_market_odds

    n_pred = 0
    n_bet = 0
    hits_pred = 0
    hits_bet = 0
    actuals: list[int] = []
    model_ps: list[float] = []
    bet_pnls: list[float] = []
    bet_stakes: list[float] = []
    odds_used = 0

    for _, row in features.iterrows():
        actual = settle_prop(prop_key, row)
        if actual is None:
            continue
        model_p = float(prop_model_prob(prop_key, row))
        actual_i = int(bool(actual))
        actuals.append(actual_i)
        model_ps.append(model_p)
        n_pred += 1
        predicted_yes = model_p >= 0.5
        if predicted_yes == bool(actual):
            hits_pred += 1

        # Value / conviction bet: model leans this side strongly enough
        if model_p >= min_model_prob_bet:
            n_bet += 1
            odds = synthetic_market_odds(model_p)
            odds_used += 1
            won = bool(actual)
            if won:
                hits_bet += 1
                pnl = flat_stake * (odds - 1.0)
            else:
                pnl = -flat_stake
            bet_pnls.append(pnl)
            bet_stakes.append(flat_stake)

    base_rate = float(np.mean(actuals)) if actuals else None
    avg_model = float(np.mean(model_ps)) if model_ps else None
    total_pnl = float(sum(bet_pnls)) if bet_pnls else None
    total_stake = float(sum(bet_stakes)) if bet_stakes else 0.0
    return {
        "prop_key": prop_key,
        "prop_label": prop_display_name(prop_key),
        "source": "historical_features",
        "n_predictions": n_pred,
        "n_bets": n_bet,
        "pred_hit_rate": (hits_pred / n_pred) if n_pred else None,
        "bet_hit_rate": (hits_bet / n_bet) if n_bet else None,
        "actual_win_rate": base_rate,
        "avg_model_prob": avg_model,
        "calibration_gap": (avg_model - base_rate) if avg_model is not None and base_rate is not None else None,
        "pnl": total_pnl,
        "stake": total_stake if total_stake else None,
        "roi": (total_pnl / total_stake) if total_pnl is not None and total_stake > 0 else None,
        "odds_coverage": (odds_used / n_bet) if n_bet else None,
        "notes": f"Synthetic vig odds; bet when model_p>={min_model_prob_bet:.0%}",
    }


def _summarize_log_props(log_df: pd.DataFrame) -> list[dict[str, Any]]:
    if log_df is None or log_df.empty:
        return []
    out: list[dict[str, Any]] = []
    for key, grp in log_df.groupby("prop_key"):
        settled = grp[grp["won"].notna()].copy()
        if settled.empty:
            continue
        n = len(settled)
        hits = int(settled["won"].astype(int).sum())
        probs = settled["model_prob"].dropna().astype(float)
        pnls = []
        stakes = []
        for _, r in settled.iterrows():
            stake = float(r.get("stake") or getattr(config, "FLAT_STAKE", 10) or 10)
            stakes.append(stake)
            if r.get("pnl") is not None and pd.notna(r.get("pnl")):
                pnls.append(float(r["pnl"]))
            elif r.get("opening_odds") is not None and pd.notna(r.get("opening_odds")):
                odds = float(r["opening_odds"])
                if odds > 1.0:
                    pnls.append(stake * (odds - 1.0) if int(r["won"]) else -stake)
        total_pnl = float(sum(pnls)) if pnls else None
        total_stake = float(sum(stakes)) if stakes else 0.0
        actual_rate = hits / n
        avg_model = float(probs.mean()) if len(probs) else None
        out.append(
            {
                "prop_key": key,
                "prop_label": prop_display_name(str(key)),
                "source": "logs",
                "n_predictions": n,
                "n_bets": n,
                "pred_hit_rate": actual_rate,
                "bet_hit_rate": actual_rate,
                "actual_win_rate": actual_rate,
                "avg_model_prob": avg_model,
                "calibration_gap": (avg_model - actual_rate) if avg_model is not None else None,
                "pnl": total_pnl,
                "stake": total_stake if total_stake else None,
                "roi": (total_pnl / total_stake) if total_pnl is not None and total_stake > 0 else None,
                "odds_coverage": float(settled["opening_odds"].notna().mean()) if "opening_odds" in settled else None,
                "notes": "Settled bet_journal / prediction_bank rows",
            }
        )
    return out


def analyze_prop_performance(
    *,
    min_model_prob_bet: float | None = None,
    flat_stake: float | None = None,
    top_other_n: int = 8,
) -> dict[str, Any]:
    """
    Build prop performance summary.

    Returns dict with ``summary_rows``, ``focus``, ``generated_at``, paths helpers.
    """
    stake = float(flat_stake if flat_stake is not None else getattr(config, "FLAT_STAKE", 10) or 10)
    min_p = float(
        min_model_prob_bet
        if min_model_prob_bet is not None
        else getattr(config, "PROP_MIN_MODEL_PROB", 0.55) or 0.55
    )
    # For O/U sides, require a clearer lean than coin-flip for "bets"
    min_p = max(min_p, 0.55)

    log_df = load_settled_prop_rows_from_logs()
    log_summaries = _summarize_log_props(log_df)

    features = load_labeled_prop_frame()
    # Markets to score from history
    from src.props import PROP_MARKET_LABELS

    hist_keys = list(dict.fromkeys([*FOCUS_PROPS, *PROP_MARKET_LABELS.keys(), *list(getattr(config, "PROP_MARKETS", []))]))
    # Drop unfinished / unsupported keys quietly
    hist_rows: list[dict[str, Any]] = []
    for key in hist_keys:
        try:
            row = _evaluate_history_prop(
                features, key, flat_stake=stake, min_model_prob_bet=min_p
            )
        except Exception as exc:
            logger.debug("prop %s eval failed: %s", key, exc)
            continue
        if int(row.get("n_predictions") or 0) <= 0:
            continue
        hist_rows.append(row)

    # Prefer historical for focus metrics; append log rows as separate source
    by_key: dict[str, dict[str, Any]] = {r["prop_key"]: r for r in hist_rows}
    for lr in log_summaries:
        k = lr["prop_key"]
        if k not in by_key:
            by_key[k] = lr
        else:
            # Attach log overlay counts
            by_key[k]["log_n_bets"] = lr["n_bets"]
            by_key[k]["log_hit_rate"] = lr["bet_hit_rate"]
            by_key[k]["log_pnl"] = lr["pnl"]

    all_rows = list(by_key.values())
    focus = {k: by_key[k] for k in FOCUS_PROPS if k in by_key}

    others = sorted(
        [r for r in all_rows if r["prop_key"] not in FOCUS_PROPS],
        key=lambda r: int(r.get("n_predictions") or 0),
        reverse=True,
    )[: int(top_other_n)]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_fights_scored": int(len(features)),
        "min_model_prob_bet": min_p,
        "flat_stake": stake,
        "log_prop_rows": int(len(log_df)) if log_df is not None else 0,
        "focus": focus,
        "top_other": others,
        "summary_rows": all_rows,
        "focus_order": list(FOCUS_PROPS),
    }


def format_prop_performance_report(report: dict[str, Any]) -> str:
    lines = [
        "PROP PERFORMANCE REPORT",
        f"Generated: {report.get('generated_at')}",
        f"Labeled fights scored: {report.get('n_fights_scored')}",
        f"Log prop rows: {report.get('log_prop_rows')} | "
        f"Bet when model_p>={100*float(report.get('min_model_prob_bet') or 0):.0f}% | "
        f"stake ${float(report.get('flat_stake') or 0):.0f}",
        "",
        "=== Over / Under 1.5 Rounds ===",
    ]
    for key in report.get("focus_order") or FOCUS_PROPS:
        r = (report.get("focus") or {}).get(key)
        if not r:
            lines.append(f"{key}: no data")
            continue
        lines.append(_format_prop_block(r))
    lines.append("")
    lines.append("=== Top other prop types (by sample size) ===")
    for r in report.get("top_other") or []:
        lines.append(_format_prop_block(r, compact=True))
    if report.get("csv_path"):
        lines.append("")
        lines.append(f"CSV: {report['csv_path']}")
    return "\n".join(lines)


def _format_prop_block(r: dict[str, Any], *, compact: bool = False) -> str:
    label = r.get("prop_label") or r.get("prop_key")
    n_pred = r.get("n_predictions") or 0
    n_bet = r.get("n_bets") or 0
    pred_hr = r.get("pred_hit_rate")
    bet_hr = r.get("bet_hit_rate")
    actual = r.get("actual_win_rate")
    avg_p = r.get("avg_model_prob")
    pnl = r.get("pnl")
    roi = r.get("roi")
    gap = r.get("calibration_gap")
    pred_s = f"{100*pred_hr:.1f}%" if pred_hr is not None else "n/a"
    bet_s = f"{100*bet_hr:.1f}%" if bet_hr is not None else "n/a"
    act_s = f"{100*actual:.1f}%" if actual is not None else "n/a"
    avg_s = f"{100*avg_p:.1f}%" if avg_p is not None else "n/a"
    pnl_s = f"${pnl:+.1f}" if pnl is not None else "n/a"
    roi_s = f"{100*roi:+.1f}%" if roi is not None else "n/a"
    gap_s = f"{100*gap:+.1f}pp" if gap is not None else "n/a"
    if compact:
        return (
            f"- {label}: n={n_pred} | pred_hit {pred_s} | bets {n_bet} hit {bet_s} | "
            f"actual {act_s} vs model {avg_s} (gap {gap_s}) | PnL {pnl_s} ROI {roi_s}"
        )
    return (
        f"{label} ({r.get('prop_key')})\n"
        f"  Predictions: {n_pred} | pred hit rate {pred_s}\n"
        f"  Bets (model lean): {n_bet} | bet hit rate {bet_s}\n"
        f"  Actual win rate {act_s} | avg model prob {avg_s} | calibration gap {gap_s}\n"
        f"  PnL {pnl_s} | ROI {roi_s} | source={r.get('source')}"
    )


def write_prop_performance_csv(
    report: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    out = Path(path) if path else (config.ROOT_DIR / "reports" / f"prop_performance_{stamp}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = report.get("summary_rows") or []
    # Stable column order; focus props first
    focus = set(report.get("focus_order") or FOCUS_PROPS)
    ordered = sorted(
        rows,
        key=lambda r: (0 if r.get("prop_key") in focus else 1, -int(r.get("n_predictions") or 0)),
    )
    pd.DataFrame(ordered).to_csv(out, index=False, encoding="utf-8")
    report["csv_path"] = str(out)
    return out


def run_prop_performance(
    *,
    csv_path: Path | str | None = None,
    min_model_prob_bet: float | None = None,
    flat_stake: float | None = None,
) -> dict[str, Any]:
    report = analyze_prop_performance(
        min_model_prob_bet=min_model_prob_bet,
        flat_stake=flat_stake,
    )
    write_prop_performance_csv(report, csv_path)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.project_paths import bootstrap

    bootstrap(entry_file=config.ROOT_DIR / "main.py")
    p = argparse.ArgumentParser(description="Historical UFC prop performance report")
    p.add_argument("-o", "--output", type=Path, default=None, help="CSV path")
    p.add_argument(
        "--min-prob",
        type=float,
        default=None,
        help="Min model prob to count as a bet (default: max(PROP_MIN_MODEL_PROB, 0.55))",
    )
    p.add_argument("--stake", type=float, default=None, help="Flat stake for PnL sim")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    report = run_prop_performance(
        csv_path=args.output,
        min_model_prob_bet=args.min_prob,
        flat_stake=args.stake,
    )
    print(format_prop_performance_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
