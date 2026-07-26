"""Replay past cards through the current model + decision stack.

Compare picks vs actual winners, show bets that would have been taken,
opening-odds PnL, and skip reasons. Fail-closed when winners/odds missing.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _event_col(df: pd.DataFrame) -> str:
    if "event_name" in df.columns:
        return "event_name"
    if "event" in df.columns:
        return "event"
    raise ValueError("No event / event_name column in features")


def _date_col(df: pd.DataFrame) -> str:
    if config.DATE_COLUMN in df.columns:
        return config.DATE_COLUMN
    if "date" in df.columns:
        return "date"
    raise ValueError("No date / event_date column in features")


def _f1_col(df: pd.DataFrame) -> str:
    return "fighter_1" if "fighter_1" in df.columns else "fighter1"


def _f2_col(df: pd.DataFrame) -> str:
    return "fighter_2" if "fighter_2" in df.columns else "fighter2"


def _normalize_event_key(name: Any) -> str:
    text = str(name or "").strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fighters_match(a: str, b: str) -> bool:
    try:
        from src.prediction_bank import _fighters_match as _fm

        return _fm(a, b)
    except Exception:
        ca, cb = str(a or "").strip().lower(), str(b or "").strip().lower()
        if not ca or not cb:
            return False
        return ca == cb or ca in cb or cb in ca


def load_replay_features() -> pd.DataFrame:
    """Load labeled feature matrix with event names (rebuild from fights if needed)."""
    from src.data_loader import load_fights, load_processed_features
    from src.feature_engineering import build_feature_matrix

    try:
        features = load_processed_features()
    except Exception:
        features = pd.DataFrame()
    if features is None or features.empty:
        fights = load_fights()
        features = build_feature_matrix(fights, keep_unlabeled=False)
    if features is None or features.empty:
        raise ValueError("No features available for replay. Run --refresh-data / --train first.")
    if config.TARGET_COLUMN not in features.columns:
        raise ValueError(f"Missing target column {config.TARGET_COLUMN}")
    labeled = features.dropna(subset=[config.TARGET_COLUMN]).copy()
    if labeled.empty:
        raise ValueError("No labeled fights in feature matrix.")

    # Attach event names from fights.csv when the feature matrix lacks them
    if "event_name" not in labeled.columns and "event" not in labeled.columns:
        try:
            fights = load_fights()
            labeled = _attach_event_names(labeled, fights)
        except Exception as exc:
            logger.debug("event name attach failed: %s", exc)
    if "event_name" not in labeled.columns and "event" not in labeled.columns:
        # Last resort: bucket by date
        dt = _date_col(labeled)
        labeled = labeled.copy()
        labeled["event_name"] = pd.to_datetime(labeled[dt], errors="coerce").dt.strftime("%Y-%m-%d")
    return labeled


def _attach_event_names(features: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """Merge event / winner / odds from fights onto the feature matrix."""
    out = features.copy()
    fight_ev = "event_name" if "event_name" in fights.columns else "event"
    if fight_ev not in fights.columns:
        return out

    keep_cols = [c for c in ("fight_id", fight_ev, "event", "winner", "f1_odds", "f2_odds") if c in fights.columns]
    # Deduplicate fight_id
    meta = fights[keep_cols].copy()
    if "fight_id" in meta.columns and "fight_id" in out.columns:
        meta = meta.dropna(subset=["fight_id"]).drop_duplicates(subset=["fight_id"], keep="last")
        merged = out.merge(meta, on="fight_id", how="left", suffixes=("", "_fight"))
        if fight_ev in merged.columns:
            if "event_name" not in merged.columns:
                merged["event_name"] = merged[fight_ev]
            else:
                merged["event_name"] = merged["event_name"].fillna(merged[fight_ev])
        if "event" not in merged.columns and "event_name" in merged.columns:
            merged["event"] = merged["event_name"]
        # Prefer feature odds; fill from fights when missing
        for oc in ("f1_odds", "f2_odds"):
            src = f"{oc}_fight"
            if oc in merged.columns and src in merged.columns:
                merged[oc] = pd.to_numeric(merged[oc], errors="coerce").fillna(
                    pd.to_numeric(merged[src], errors="coerce")
                )
            elif src in merged.columns and oc not in merged.columns:
                merged[oc] = pd.to_numeric(merged[src], errors="coerce")
        if "winner" not in merged.columns and "winner" in meta.columns:
            merged["winner"] = meta.set_index("fight_id").reindex(merged["fight_id"])["winner"].values
        return merged

    # Fallback: date + fighter names
    f1c, f2c = _f1_col(out), _f2_col(out)
    ff1 = "fighter_1" if "fighter_1" in fights.columns else "fighter1"
    ff2 = "fighter_2" if "fighter_2" in fights.columns else "fighter2"
    dt_f = config.DATE_COLUMN if config.DATE_COLUMN in fights.columns else "date"
    dt_o = _date_col(out)
    tmp = fights[[ff1, ff2, fight_ev, dt_f]].copy()
    tmp["_d"] = pd.to_datetime(tmp[dt_f], errors="coerce").dt.normalize()
    out = out.copy()
    out["_d"] = pd.to_datetime(out[dt_o], errors="coerce").dt.normalize()
    out["_f1"] = out[f1c].astype(str).str.strip().str.lower()
    out["_f2"] = out[f2c].astype(str).str.strip().str.lower()
    tmp["_f1"] = tmp[ff1].astype(str).str.strip().str.lower()
    tmp["_f2"] = tmp[ff2].astype(str).str.strip().str.lower()
    m = out.merge(
        tmp[["_d", "_f1", "_f2", fight_ev]],
        on=["_d", "_f1", "_f2"],
        how="left",
    )
    m["event_name"] = m[fight_ev]
    m["event"] = m["event_name"]
    return m.drop(columns=[c for c in ("_d", "_f1", "_f2") if c in m.columns], errors="ignore")


def list_past_events(
    features: pd.DataFrame | None = None,
    *,
    last: int | None = None,
) -> list[dict[str, Any]]:
    """Chronological past events with fight counts (newest last)."""
    feats = features if features is not None else load_replay_features()
    ev_col = _event_col(feats)
    dt_col = _date_col(feats)
    work = feats.copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work = work.dropna(subset=[dt_col, ev_col])
    grouped = (
        work.groupby(ev_col, sort=False)
        .agg(event_date=(dt_col, "max"), fights=(ev_col, "size"))
        .reset_index()
        .rename(columns={ev_col: "event"})
    )
    grouped = grouped.sort_values("event_date")
    rows = [
        {
            "event": str(r["event"]),
            "event_date": pd.Timestamp(r["event_date"]).strftime("%Y-%m-%d"),
            "fights": int(r["fights"]),
        }
        for _, r in grouped.iterrows()
    ]
    if last is not None and last > 0:
        return rows[-int(last) :]
    return rows


def resolve_replay_events(
    *,
    event: str | None = None,
    date: str | None = None,
    last: int | None = None,
    features: pd.DataFrame | None = None,
) -> list[str]:
    """
    Resolve event names to replay.

    Prefer --event / --date; otherwise --last N (default 1 when nothing given).
    """
    feats = features if features is not None else load_replay_features()
    catalog = list_past_events(feats)
    if not catalog:
        raise ValueError("No past events found in features.")

    if event:
        q = _normalize_event_key(event)
        matches = [e for e in catalog if q in _normalize_event_key(e["event"])]
        if not matches:
            # try exact slug / number match
            matches = [
                e
                for e in catalog
                if q.replace(" ", "") in _normalize_event_key(e["event"]).replace(" ", "")
            ]
        if not matches:
            sample = ", ".join(e["event"] for e in catalog[-8:])
            raise ValueError(f"Past event not found: {event!r}. Recent: {sample}")
        # Prefer the most recent match if several contain the query
        return [matches[-1]["event"]]

    if date:
        target = pd.Timestamp(date).normalize()
        hits = [
            e
            for e in catalog
            if pd.Timestamp(e["event_date"]).normalize() == target
        ]
        if not hits:
            raise ValueError(f"No past event on date {date}")
        return [h["event"] for h in hits]

    n = int(last) if last is not None else 1
    n = max(1, n)
    return [e["event"] for e in catalog[-n:]]


def _actual_winner(row: pd.Series) -> str:
    f1c = "fighter_1" if "fighter_1" in row.index else "fighter1"
    f2c = "fighter_2" if "fighter_2" in row.index else "fighter2"
    f1 = str(row.get(f1c) or "")
    f2 = str(row.get(f2c) or "")
    if "winner" in row.index and str(row.get("winner") or "").strip():
        return str(row.get("winner")).strip()
    y = row.get(config.TARGET_COLUMN)
    try:
        if pd.notna(y):
            return f1 if int(y) == 1 else f2
    except (TypeError, ValueError):
        pass
    return ""


def score_event_card(
    event_name: str,
    features: pd.DataFrame,
    *,
    predictor: Any | None = None,
    explain: bool = False,
) -> pd.DataFrame:
    """Score one past event with the current model; attach historical edges."""
    from src.predictor import FightPredictor, attach_edge_columns

    ev_col = _event_col(features)
    subset = features[features[ev_col].astype(str) == str(event_name)].copy()
    if subset.empty:
        # fuzzy
        q = _normalize_event_key(event_name)
        mask = features[ev_col].map(lambda x: q in _normalize_event_key(x))
        subset = features[mask].copy()
    if subset.empty:
        raise ValueError(f"No fights for event {event_name!r}")

    pred = predictor or FightPredictor()
    scored = pred.predict_batch(subset, apply_style_bonus=True, explain=explain)
    scored = attach_edge_columns(scored)
    if "event_name" not in scored.columns:
        scored["event_name"] = event_name
    if "event" not in scored.columns:
        scored["event"] = event_name

    # Actual winners for compare
    f1c, f2c = _f1_col(scored), _f2_col(scored)
    actuals = []
    for _, row in scored.iterrows():
        actuals.append(_actual_winner(row))
    scored["actual_winner"] = actuals
    scored["correct"] = [
        1 if (a and p and _fighters_match(str(p), str(a))) else (0 if a else None)
        for p, a in zip(scored.get("predicted_winner", []), scored["actual_winner"])
    ]
    return scored.reset_index(drop=True)


def _opening_odds_for_pick(row: pd.Series, pick: str) -> float | None:
    from src.strategy import decimal_odds_for_pick

    odds = decimal_odds_for_pick(row, pick)
    if odds is not None and odds > 1.0:
        return float(odds)
    f1 = str(row.get("fighter_1") or row.get("fighter1") or "")
    f2 = str(row.get("fighter_2") or row.get("fighter2") or "")
    try:
        if _fighters_match(pick, f1):
            v = float(row.get("f1_odds"))
            return v if v > 1.0 else None
        if _fighters_match(pick, f2):
            v = float(row.get("f2_odds"))
            return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None
    return None


def build_compare_rows(
    preds: pd.DataFrame,
    alerts: dict[str, Any],
    *,
    stake: float | None = None,
) -> list[dict[str, Any]]:
    """Fight-level rows: pick vs winner, bet taken, PnL, skip reason."""
    from src.settlement import compute_pnl

    flat = float(stake if stake is not None else getattr(config, "FLAT_STAKE", 10.0) or 10.0)
    bet_by_fight: dict[str, dict[str, Any]] = {}
    for s in alerts.get("singles") or []:
        key = str(s.get("fight_id") or s.get("fight") or "")
        bet_by_fight[key] = s
    skip_by_fight: dict[str, dict[str, Any]] = {}
    for s in alerts.get("skipped") or []:
        key = str(s.get("fight_id") or s.get("fight") or "")
        skip_by_fight[key] = s

    f1c, f2c = _f1_col(preds), _f2_col(preds)
    fid = config.FIGHT_ID_COLUMN if config.FIGHT_ID_COLUMN in preds.columns else None
    rows: list[dict[str, Any]] = []
    for _, row in preds.iterrows():
        f1, f2 = str(row.get(f1c) or ""), str(row.get(f2c) or "")
        fight = f"{f1} vs {f2}"
        fight_id = str(row.get(fid) or fight) if fid else fight
        pick = str(row.get("predicted_winner") or "")
        actual = str(row.get("actual_winner") or "")
        correct_raw = row.get("correct")
        correct = None if correct_raw is None or (isinstance(correct_raw, float) and np.isnan(correct_raw)) else int(correct_raw)

        bet = bet_by_fight.get(fight_id) or bet_by_fight.get(fight)
        skip = skip_by_fight.get(fight_id) or skip_by_fight.get(fight)
        bet_taken = bool(bet)
        stake_used = None
        pnl = None
        opening_odds = None
        if bet_taken:
            try:
                stake_used = float(bet.get("suggested_stake") or flat)
            except (TypeError, ValueError):
                stake_used = flat
            if stake_used <= 0:
                stake_used = flat
            opening_odds = _opening_odds_for_pick(row, str(bet.get("pick") or pick))
            if correct is not None and opening_odds is not None:
                pnl = compute_pnl(
                    correct=bool(correct),
                    stake=stake_used,
                    opening_odds=opening_odds,
                )
            elif opening_odds is None:
                # Fail-closed: bet taken but no odds → no invented PnL
                pnl = None

        edge = row.get("best_edge")
        try:
            edge_f = float(edge) if edge is not None and pd.notna(edge) else None
        except (TypeError, ValueError):
            edge_f = None

        rows.append(
            {
                "event": str(row.get("event_name") or row.get("event") or ""),
                "date": str(row.get(config.DATE_COLUMN) or row.get("date") or "")[:10],
                "fight_id": fight_id,
                "fighter_1": f1,
                "fighter_2": f2,
                "predicted_winner": pick,
                "actual_winner": actual,
                "correct": correct,
                "prob": float(row["predicted_prob"]) if pd.notna(row.get("predicted_prob")) else None,
                "edge": edge_f,
                "f1_odds": row.get("f1_odds"),
                "f2_odds": row.get("f2_odds"),
                "bet_taken": bet_taken,
                "bet_pick": (bet or {}).get("pick", "") if bet_taken else "",
                "stake": stake_used if bet_taken else "",
                "opening_odds": opening_odds if bet_taken else "",
                "pnl": pnl if pnl is not None else "",
                "skip_reason": (skip or {}).get("skip_reason", "") if skip and not bet_taken else "",
                "uncertainty_action": row.get("uncertainty_action", ""),
            }
        )
    return rows


def summarize_compare(
    rows: list[dict[str, Any]],
    *,
    event_name: str = "",
    alerts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = [r for r in rows if r.get("correct") is not None]
    n = len(scored)
    correct_n = sum(1 for r in scored if r.get("correct") == 1)
    bets = [r for r in rows if r.get("bet_taken")]
    pnls = [float(r["pnl"]) for r in bets if r.get("pnl") not in ("", None)]
    stakes = [float(r["stake"]) for r in bets if r.get("stake") not in ("", None)]
    skip_counts: dict[str, int] = {}
    for r in rows:
        code = str(r.get("skip_reason") or "").strip()
        if code:
            skip_counts[code] = skip_counts.get(code, 0) + 1
    if alerts:
        for s in alerts.get("skipped") or []:
            code = str(s.get("skip_reason") or "unknown")
            # already counted from rows when present; fill gaps
            if code and code not in skip_counts:
                skip_counts[code] = skip_counts.get(code, 0)

    total_pnl = sum(pnls) if pnls else None
    total_stake = sum(stakes) if stakes else 0.0
    return {
        "event": event_name,
        "fights": len(rows),
        "scored": n,
        "correct": correct_n,
        "accuracy": (correct_n / n) if n else None,
        "bets_taken": len(bets),
        "bets_won": sum(1 for r in bets if r.get("correct") == 1),
        "bets_lost": sum(1 for r in bets if r.get("correct") == 0),
        "pnl": total_pnl,
        "stake": total_stake if bets else None,
        "roi": (total_pnl / total_stake) if total_pnl is not None and total_stake > 0 else None,
        "incomplete_pnl": any(r.get("bet_taken") and r.get("pnl") in ("", None) for r in rows),
        "skip_counts": dict(sorted(skip_counts.items(), key=lambda kv: -kv[1])),
        "top_skips": sorted(skip_counts.items(), key=lambda kv: -kv[1])[:5],
    }


def replay_event(
    event_name: str,
    *,
    features: pd.DataFrame | None = None,
    predictor: Any | None = None,
    bankroll: float | None = None,
    use_dynamic_thresholds: bool | None = None,
    explain: bool = False,
    narrative: bool = False,
) -> dict[str, Any]:
    """Run full decision stack on one past event and compare to winners."""
    from src.alerts import generate_alerts
    from src.predictor import FightPredictor

    feats = features if features is not None else load_replay_features()
    pred = predictor or FightPredictor()
    scored = score_event_card(event_name, feats, predictor=pred, explain=explain)

    # If historical opening odds are thin, try live book fallback (logged source).
    # Fail-closed for PnL when still unmatched after the chain.
    try:
        odds_cov = 0.0
        if "f1_odds" in scored.columns:
            odds_cov = float(pd.to_numeric(scored["f1_odds"], errors="coerce").notna().mean())
        if odds_cov < 0.5:
            from src.predictor import merge_predictions_with_odds

            scored = merge_predictions_with_odds(scored, fetch_if_missing=True)
    except Exception as exc:
        logger.debug("replay live odds fallback skipped: %s", exc)

    narrative_result = None
    if narrative:
        try:
            from src.grok_analysis import analyze_card_with_grok

            narrative_result = analyze_card_with_grok(scored, event_name=event_name)
        except Exception as exc:
            logger.debug("replay narrative skipped: %s", exc)
            narrative_result = None

    br = float(bankroll if bankroll is not None else config.INITIAL_BANKROLL)
    alerts = generate_alerts(
        scored,
        bankroll=br,
        event_name=event_name,
        use_dynamic_thresholds=use_dynamic_thresholds,
        narrative_result=narrative_result,
    )
    rows = build_compare_rows(scored, alerts)
    summary = summarize_compare(rows, event_name=event_name, alerts=alerts)
    return {
        "event": event_name,
        "predictions": scored,
        "alerts": alerts,
        "rows": rows,
        "summary": summary,
    }


def run_replay(
    *,
    event: str | None = None,
    date: str | None = None,
    last: int | None = None,
    csv_path: Path | str | None = None,
    bankroll: float | None = None,
    use_dynamic_thresholds: bool | None = None,
    explain: bool = False,
    narrative: bool = False,
) -> dict[str, Any]:
    """Replay one or more past events; optional CSV of fight-level compares."""
    from src.predictor import FightPredictor

    features = load_replay_features()
    events = resolve_replay_events(event=event, date=date, last=last, features=features)
    predictor = FightPredictor()

    per_event: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for ev in events:
        logger.info("Replaying %s …", ev)
        result = replay_event(
            ev,
            features=features,
            predictor=predictor,
            bankroll=bankroll,
            use_dynamic_thresholds=use_dynamic_thresholds,
            explain=explain,
            narrative=narrative,
        )
        per_event.append(result["summary"])
        all_rows.extend(result["rows"])

    # Aggregate
    scored = [r for r in all_rows if r.get("correct") is not None]
    n = len(scored)
    correct_n = sum(1 for r in scored if r.get("correct") == 1)
    bets = [r for r in all_rows if r.get("bet_taken")]
    pnls = [float(r["pnl"]) for r in bets if r.get("pnl") not in ("", None)]
    stakes = [float(r["stake"]) for r in bets if r.get("stake") not in ("", None)]
    skip_counts: dict[str, int] = {}
    for r in all_rows:
        code = str(r.get("skip_reason") or "").strip()
        if code:
            skip_counts[code] = skip_counts.get(code, 0) + 1

    total_pnl = sum(pnls) if pnls else None
    total_stake = sum(stakes) if stakes else 0.0
    overall = {
        "events": events,
        "n_events": len(events),
        "fights": len(all_rows),
        "scored": n,
        "correct": correct_n,
        "accuracy": (correct_n / n) if n else None,
        "bets_taken": len(bets),
        "bets_won": sum(1 for r in bets if r.get("correct") == 1),
        "bets_lost": sum(1 for r in bets if r.get("correct") == 0),
        "pnl": total_pnl,
        "stake": total_stake if bets else None,
        "roi": (total_pnl / total_stake) if total_pnl is not None and total_stake > 0 else None,
        "incomplete_pnl": any(r.get("bet_taken") and r.get("pnl") in ("", None) for r in all_rows),
        "skip_counts": dict(sorted(skip_counts.items(), key=lambda kv: -kv[1])),
        "per_event": per_event,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    out_path = None
    if csv_path:
        out_path = write_replay_csv(all_rows, csv_path)

    return {
        "overall": overall,
        "rows": all_rows,
        "csv_path": str(out_path) if out_path else None,
        "events": events,
    }


def write_replay_csv(rows: list[dict[str, Any]], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    return out


def format_replay_summary(report: dict[str, Any]) -> str:
    """Short human-readable summary for CLI."""
    o = report.get("overall") or report
    lines = [
        f"Replay {o.get('n_events', 1)} event(s) - {o.get('fights', 0)} fights",
        f"Events: {', '.join(str(e) for e in (o.get('events') or []))}",
    ]
    acc = o.get("accuracy")
    if acc is None:
        lines.append("Accuracy: n/a (fail-closed - no winners)")
    else:
        lines.append(
            f"Pick accuracy: {acc:.1%} ({o.get('correct', 0)}/{o.get('scored', 0)})"
        )
    lines.append(
        f"Bets taken: {o.get('bets_taken', 0)} "
        f"(W {o.get('bets_won', 0)} / L {o.get('bets_lost', 0)})"
    )
    pnl = o.get("pnl")
    if pnl is None:
        note = "incomplete odds" if o.get("incomplete_pnl") else "no bets"
        lines.append(f"PnL @ opening odds: n/a ({note})")
    else:
        roi = o.get("roi")
        roi_s = f" | ROI {roi:.1%}" if roi is not None else ""
        lines.append(f"PnL @ opening odds: ${pnl:+.2f}{roi_s}")
        if o.get("incomplete_pnl"):
            lines.append("  (some bets missing odds - those PnLs excluded, fail-closed)")

    skips = o.get("skip_counts") or {}
    if skips:
        top = ", ".join(f"{k} x{v}" for k, v in list(skips.items())[:5])
        lines.append(f"Skip reasons: {top}")
    else:
        lines.append("Skip reasons: (none logged)")

    for pe in o.get("per_event") or []:
        pe_acc = pe.get("accuracy")
        pe_acc_s = f"{pe_acc:.0%}" if pe_acc is not None else "n/a"
        pe_pnl = pe.get("pnl")
        pe_pnl_s = f"${pe_pnl:+.2f}" if pe_pnl is not None else "n/a"
        lines.append(
            f"  - {pe.get('event')}: acc {pe_acc_s} | "
            f"bets {pe.get('bets_taken', 0)} | pnl {pe_pnl_s}"
        )

    if report.get("csv_path"):
        lines.append(f"CSV: {report['csv_path']}")
    return "\n".join(lines)


def build_replay_parser() -> Any:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m main replay",
        description="Replay past cards through the current model + decision stack.",
    )
    p.add_argument("--event", metavar="NAME", help='Past event name (e.g. "UFC 329")')
    p.add_argument("--date", metavar="YYYY-MM-DD", help="Past event date")
    p.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Replay the last N completed events (default: 1 if no --event/--date)",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="CSV",
        type=Path,
        help="Write fight-level compare CSV",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List recent past events and exit",
    )
    p.add_argument(
        "--list-n",
        type=int,
        default=15,
        metavar="N",
        help="How many past events to show with --list (default: 15)",
    )
    p.add_argument("--explain", action="store_true", help="Attach SHAP (slower)")
    p.add_argument(
        "--narrative",
        action="store_true",
        help="Run Ollama/Grok narrative tilt (optional; off by default)",
    )
    p.add_argument(
        "--dynamic-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dynamic min-edge (default: on)",
    )
    p.add_argument(
        "--profile",
        choices=["live", "research", "paper"],
        default=None,
        help="Risk profile override",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


def run_replay_cli(argv: list[str] | None = None) -> int:
    """CLI entry for ``python -m main replay …``."""
    from src.project_paths import bootstrap

    bootstrap(entry_file=config.ROOT_DIR / "main.py")
    parser = build_replay_parser()
    args = parser.parse_args(argv)

    if args.profile:
        config.UFC_PROFILE = config.normalize_profile(args.profile)
        config.apply_profile_overrides()
    config.DYNAMIC_THRESHOLDS_ENABLED = args.dynamic_thresholds

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")

    if args.list:
        events = list_past_events(last=args.list_n)
        print(f"Past events (last {len(events)}):")
        for e in reversed(events):
            print(f"  {e['event_date']}  {e['event']}  ({e['fights']} fights)")
        return 0

    if not args.event and not args.date and args.last is None:
        args.last = 1

    try:
        report = run_replay(
            event=args.event,
            date=args.date,
            last=args.last,
            csv_path=args.output,
            use_dynamic_thresholds=args.dynamic_thresholds,
            explain=args.explain,
            narrative=args.narrative,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        logger.exception("replay failed")
        print(f"ERROR: {exc}")
        return 1

    print(format_replay_summary(report))
    return 0
