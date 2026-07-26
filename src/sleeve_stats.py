"""Per-sleeve performance reporting for settled bets and model picks.

Breaks down outcomes by bet type, weight class, odds bucket, model probability /
confidence bucket, and uncertainty level. Writes ``reports/sleeve_stats_YYYYMMDD.csv``
and a CLI / dashboard summary (top & bottom sleeves).

Re-run::

    python scripts/run_sleeve_stats.py
    python -m src.sleeve_stats
    python -m main --sleeve-stats
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

SLEEVE_DIMENSIONS: tuple[str, ...] = (
    "bet_type",
    "weight_class",
    "odds_bucket",
    "model_prob_bucket",
    "confidence_bucket",
    "uncertainty_level",
)

SLEEVE_LABELS: dict[str, str] = {
    "bet_type": "Bet type",
    "weight_class": "Weight class",
    "odds_bucket": "Odds bucket",
    "model_prob_bucket": "Model probability",
    "confidence_bucket": "Confidence",
    "uncertainty_level": "Uncertainty",
}

# Flat stake used when bank stake is missing but odds exist (ROI still meaningful).
_DEFAULT_UNIT_STAKE = 1.0

# Min n for ranking top/bottom sleeves in summaries / dashboard.
_MIN_N_RANK = 3


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        if isinstance(val, str) and not str(val).strip():
            return None
        if pd.isna(val):
            return None
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_bool(val: Any) -> bool | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    text = str(val).strip().lower()
    if text in ("1", "true", "yes", "y", "w", "win"):
        return True
    if text in ("0", "false", "no", "n", "l", "loss"):
        return False
    try:
        return bool(int(float(text)))
    except (TypeError, ValueError):
        return None


def normalize_bet_type(
    *,
    market_type: Any = None,
    prop_type: Any = None,
    notes: Any = None,
    pick: Any = None,
) -> str:
    """Map raw market/prop tags → single | 2leg_parlay | over_1_5 | other."""
    chunks = [
        str(market_type or "").strip().lower(),
        str(prop_type or "").strip().lower(),
        str(notes or "").strip().lower(),
        str(pick or "").strip().lower(),
    ]
    blob = " ".join(c for c in chunks if c)

    if any(x in blob for x in ("2leg", "2-leg", "2_leg", "two leg", "parlay")):
        return "2leg_parlay"
    if "over" in blob and "1" in blob and "5" in blob:
        return "over_1_5"
    if any(
        x in blob
        for x in (
            "under_1_5",
            "under 1.5",
            "round_1",
            "ko_tko",
            "submission",
            "decision",
            "finish",
            "prop",
        )
    ):
        # Keep Over 1.5 as its own sleeve; other props roll up.
        if "over" in blob and "1" in blob and "5" in blob:
            return "over_1_5"
        return "other_prop"

    mt = chunks[0]
    pt = chunks[1]
    if mt in ("", "moneyline", "ml", "winner", "single", "singles"):
        if pt in ("", "moneyline", "ml", "winner", "nan", "none"):
            return "single"
    if mt in ("parlay", "same_game_parlay", "sgp"):
        return "2leg_parlay"
    return "single"


def model_prob_bucket(prob: Any) -> str:
    p = _safe_float(prob)
    if p is None:
        return "unknown"
    if p > 1.0:
        p = p / 100.0
    p = max(0.0, min(1.0, p))
    if p < 0.55:
        return "lt_55"
    if p < 0.65:
        return "55_65"
    if p < 0.75:
        return "65_75"
    if p < 0.85:
        return "75_85"
    return "ge_85"


def confidence_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("high", "medium", "low"):
        return text
    if "high" in text:
        return "high"
    if "med" in text:
        return "medium"
    if "low" in text:
        return "low"
    return "unknown"


def uncertainty_level_from_row(row: pd.Series | dict[str, Any]) -> str:
    """
    low / medium / high / unknown from stored level, gate action, or metrics.
    """
    if isinstance(row, dict):
        row = pd.Series(row)

    stored = str(row.get("uncertainty_level") or row.get("uncertainty_label") or "").strip().lower()
    if stored in ("low", "medium", "high"):
        return stored
    if "high" in stored:
        return "high"
    if "med" in stored:
        return "medium"
    if "low" in stored:
        return "low"

    action = str(row.get("uncertainty_action") or "").strip().lower()
    if action == "skip":
        return "high"
    if action == "tighten":
        return "medium"
    if action == "allow":
        return "low"

    # Derive from disagreement / interval width when present.
    try:
        from src.uncertainty_gates import evaluate_uncertainty_gate, read_uncertainty_metrics

        d, w = read_uncertainty_metrics(row)
        if d is None and w is None:
            # Parse SKIP tags from brief/reason strings as a last resort.
            brief = str(row.get("reason_brief") or row.get("skip_reason") or "").lower()
            if "high_disagreement" in brief or "wide_interval" in brief or "skip:" in brief:
                return "high"
            return "unknown"
        gate = evaluate_uncertainty_gate(row)
        if gate.action == "skip":
            return "high"
        if gate.action == "tighten":
            return "medium"
        return "low"
    except Exception:
        return "unknown"


def classify_uncertainty_level(
    *,
    uncertainty_level: Any = None,
    uncertainty_action: Any = None,
    ensemble_disagreement: Any = None,
    interval_width: Any = None,
) -> str:
    return uncertainty_level_from_row(
        {
            "uncertainty_level": uncertainty_level,
            "uncertainty_action": uncertainty_action,
            "ensemble_disagreement": ensemble_disagreement,
            "interval_width": interval_width,
        }
    )


def _unit_pnl(correct: bool, decimal_odds: float | None, stake: float | None) -> float | None:
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    stake_f = stake if stake is not None and stake > 0 else _DEFAULT_UNIT_STAKE
    return stake_f * (decimal_odds - 1.0) if correct else -stake_f


def load_settled_sleeve_rows(
    *,
    bank_path: Path | str | None = None,
    include_journal: bool = True,
) -> pd.DataFrame:
    """Settled prediction / bet rows with sleeve tags attached."""
    rows: list[dict[str, Any]] = []

    try:
        from src.prediction_bank import load_bank

        bank = load_bank(bank_path)
    except Exception as exc:
        logger.debug("prediction bank load failed: %s", exc)
        bank = pd.DataFrame()

    if not bank.empty:
        settled = bank[bank["status"].astype(str).str.lower() == "settled"].copy()
        for _, r in settled.iterrows():
            correct = _safe_bool(r.get("correct"))
            if correct is None:
                continue
            odds = _safe_float(r.get("odds"))
            stake = _safe_float(r.get("stake"))
            pnl = _safe_float(r.get("pnl"))
            if pnl is None:
                pnl = _unit_pnl(correct, odds, stake)
            pick_prob = _safe_float(r.get("pick_prob"))
            bet_type = normalize_bet_type(
                market_type=r.get("market_type"),
                prop_type=r.get("prop_type"),
                pick=r.get("pick"),
            )
            unc = uncertainty_level_from_row(r)
            rows.append(
                {
                    "source": "prediction_bank",
                    "prediction_id": str(r.get("prediction_id") or ""),
                    "bet_type": bet_type,
                    "weight_class": _slug_wc(r.get("weight_class")),
                    "odds_bucket": str(r.get("odds_bucket") or "unknown").strip().lower()
                    or "unknown",
                    "model_prob": pick_prob,
                    "model_prob_bucket": model_prob_bucket(pick_prob),
                    "confidence_bucket": confidence_bucket(r.get("confidence")),
                    "uncertainty_level": unc,
                    "correct": int(correct),
                    "odds": odds,
                    "stake": stake,
                    "pnl": pnl,
                }
            )

    if include_journal:
        journal_path = Path(getattr(config, "BET_JOURNAL_CSV", config.DATA_DIR / "bet_journal.csv"))
        if journal_path.is_file():
            try:
                j = pd.read_csv(journal_path, low_memory=False)
            except Exception as exc:
                logger.debug("bet journal load failed: %s", exc)
                j = pd.DataFrame()
            seen = {r["prediction_id"] for r in rows if r.get("prediction_id")}
            for _, r in j.iterrows():
                correct = _safe_bool(r.get("correct"))
                if correct is None:
                    continue
                pid = str(r.get("prediction_id") or "").strip()
                if pid and pid in seen:
                    continue
                # Prefer settlement events; skip pure signal rows without outcome.
                et = str(r.get("event_type") or "").strip().lower()
                if et and et not in ("settlement", "settled", "result", "closed", ""):
                    # Still allow rows that have correct filled even if typed as signal.
                    if et in ("skip", "alert", "heartbeat", "card"):
                        continue
                odds = _safe_float(r.get("opening_odds")) or _safe_float(r.get("odds"))
                stake = _safe_float(r.get("stake"))
                pnl = _safe_float(r.get("pnl"))
                if pnl is None:
                    pnl = _unit_pnl(correct, odds, stake)
                model_prob = _safe_float(r.get("model_prob"))
                bet_type = normalize_bet_type(
                    market_type=r.get("event_type"),
                    prop_type=r.get("prop_type"),
                    notes=r.get("notes"),
                    pick=r.get("pick"),
                )
                rows.append(
                    {
                        "source": "bet_journal",
                        "prediction_id": pid,
                        "bet_type": bet_type,
                        "weight_class": _slug_wc(r.get("weight_class")),
                        "odds_bucket": str(r.get("odds_bucket") or "unknown").strip().lower()
                        or "unknown",
                        "model_prob": model_prob,
                        "model_prob_bucket": model_prob_bucket(model_prob),
                        "confidence_bucket": confidence_bucket(r.get("confidence")),
                        "uncertainty_level": uncertainty_level_from_row(r),
                        "correct": int(correct),
                        "odds": odds,
                        "stake": stake,
                        "pnl": pnl,
                    }
                )

    # Fill missing odds_bucket from decimal odds when possible.
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    try:
        from src.strategy_performance import odds_bucket_from_decimal

        mask = out["odds_bucket"].isin(("", "unknown", "nan", "none"))
        if mask.any():
            out.loc[mask, "odds_bucket"] = out.loc[mask, "odds"].map(
                lambda o: odds_bucket_from_decimal(o) if o is not None else "unknown"
            )
    except Exception:
        pass
    return out


def _slug_wc(value: Any) -> str:
    try:
        from src.strategy_performance import normalize_weight_class

        return normalize_weight_class(value)
    except Exception:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        return text or "unknown"


def _aggregate_sleeve(group: pd.DataFrame, *, dimension: str, value: str) -> dict[str, Any]:
    n = len(group)
    hits = int(group["correct"].sum()) if n else 0
    hit_rate = hits / n if n else None
    probs = group["model_prob"].dropna()
    avg_model_prob = float(probs.mean()) if len(probs) else None
    actual_win_rate = hit_rate

    with_odds = group[group["odds"].notna() & (group["odds"] > 1.0)]
    pnl_series = with_odds["pnl"].dropna() if not with_odds.empty else pd.Series(dtype=float)
    pnl_total = float(pnl_series.sum()) if len(pnl_series) else None
    stakes = with_odds["stake"].fillna(_DEFAULT_UNIT_STAKE)
    stakes = stakes.where(stakes > 0, _DEFAULT_UNIT_STAKE)
    stake_sum = float(stakes.sum()) if not with_odds.empty else None
    roi = (pnl_total / stake_sum) if pnl_total is not None and stake_sum and stake_sum > 0 else None

    cal_gap = None
    if avg_model_prob is not None and actual_win_rate is not None:
        cal_gap = actual_win_rate - avg_model_prob

    # Rank score: prefer ROI when odds exist, else hit rate vs model gap.
    if roi is not None and n >= _MIN_N_RANK:
        rank_score = 50.0 + min(40.0, max(-40.0, roi * 100.0))
        if hit_rate is not None:
            rank_score += min(10.0, max(-10.0, (hit_rate - 0.5) * 20.0))
    elif hit_rate is not None and n >= _MIN_N_RANK:
        rank_score = 50.0 + (hit_rate - 0.5) * 80.0
        if cal_gap is not None:
            rank_score += min(10.0, max(-10.0, cal_gap * 40.0))
    else:
        rank_score = 0.0

    return {
        "dimension": dimension,
        "dimension_label": SLEEVE_LABELS.get(dimension, dimension),
        "sleeve": value,
        "sleeve_key": f"{dimension}:{value}",
        "n_bets": n,
        "n_with_odds": int(len(with_odds)),
        "hits": hits,
        "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
        "pnl": round(pnl_total, 2) if pnl_total is not None else None,
        "roi": round(roi, 4) if roi is not None else None,
        "avg_model_prob": round(avg_model_prob, 4) if avg_model_prob is not None else None,
        "actual_win_rate": round(actual_win_rate, 4) if actual_win_rate is not None else None,
        "calibration_gap": round(cal_gap, 4) if cal_gap is not None else None,
        "rank_score": round(rank_score, 2),
    }


def compute_sleeve_stats(df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """Aggregate metrics for every sleeve across all dimensions."""
    work = df if df is not None else load_settled_sleeve_rows()
    if work is None or work.empty:
        return []

    stats: list[dict[str, Any]] = []
    for dim in SLEEVE_DIMENSIONS:
        if dim not in work.columns:
            continue
        for value, grp in work.groupby(work[dim].astype(str).fillna("unknown"), dropna=False):
            val = str(value or "unknown").strip() or "unknown"
            stats.append(_aggregate_sleeve(grp, dimension=dim, value=val))

    stats.sort(
        key=lambda r: (
            SLEEVE_DIMENSIONS.index(r["dimension"])
            if r["dimension"] in SLEEVE_DIMENSIONS
            else 99,
            -int(r.get("n_bets") or 0),
            str(r.get("sleeve") or ""),
        )
    )
    return stats


def rank_sleeves(
    stats: list[dict[str, Any]],
    *,
    min_n: int = _MIN_N_RANK,
    limit: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Top / bottom sleeves by rank_score (min sample size)."""
    eligible = [r for r in stats if int(r.get("n_bets") or 0) >= min_n]
    ranked = sorted(eligible, key=lambda r: float(r.get("rank_score") or 0), reverse=True)
    top = ranked[:limit]
    bottom = list(reversed(ranked[-limit:])) if ranked else []
    # Avoid duplicating the same sleeve in both when tiny set.
    top_keys = {r["sleeve_key"] for r in top}
    bottom = [r for r in bottom if r["sleeve_key"] not in top_keys] or bottom
    return top, bottom


def write_sleeve_stats_csv(
    stats: list[dict[str, Any]],
    path: Path | str | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    out = Path(path) if path else (config.ROOT_DIR / "reports" / f"sleeve_stats_{stamp}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dimension",
        "dimension_label",
        "sleeve",
        "sleeve_key",
        "n_bets",
        "n_with_odds",
        "hits",
        "hit_rate",
        "pnl",
        "roi",
        "avg_model_prob",
        "actual_win_rate",
        "calibration_gap",
        "rank_score",
    ]
    frame = pd.DataFrame(stats)
    for c in cols:
        if c not in frame.columns:
            frame[c] = None
    frame[cols].to_csv(out, index=False, encoding="utf-8")
    return out


def _fmt_pct(x: Any) -> str:
    f = _safe_float(x)
    if f is None:
        return "n/a"
    return f"{100.0 * f:.1f}%"


def _fmt_pnl(x: Any) -> str:
    f = _safe_float(x)
    if f is None:
        return "n/a"
    return f"${f:+.2f}"


def _fmt_roi(x: Any) -> str:
    f = _safe_float(x)
    if f is None:
        return "n/a"
    return f"{100.0 * f:+.1f}%"


def _sleeve_line(r: dict[str, Any]) -> str:
    label = f"{r.get('dimension_label') or r.get('dimension')}: {r.get('sleeve')}"
    return (
        f"- {label} | n={r.get('n_bets', 0)} | hit {_fmt_pct(r.get('hit_rate'))} | "
        f"ROI {_fmt_roi(r.get('roi'))} PnL {_fmt_pnl(r.get('pnl'))} | "
        f"model {_fmt_pct(r.get('avg_model_prob'))} vs actual {_fmt_pct(r.get('actual_win_rate'))}"
    )


def format_sleeve_stats_report(report: dict[str, Any]) -> str:
    lines = [
        "SLEEVE PERFORMANCE REPORT",
        f"Generated: {report.get('generated_at')}",
        f"Settled rows: {report.get('n_rows', 0)} "
        f"(bank={report.get('n_bank', 0)}, journal={report.get('n_journal', 0)})",
        "",
    ]

    top = report.get("top_sleeves") or []
    bottom = report.get("bottom_sleeves") or []
    lines.append("=== Top sleeves ===")
    if top:
        for r in top:
            lines.append(_sleeve_line(r))
    else:
        lines.append("(insufficient settled bets — need ≥3 per sleeve)")

    lines.append("")
    lines.append("=== Bottom sleeves ===")
    if bottom:
        for r in bottom:
            lines.append(_sleeve_line(r))
    else:
        lines.append("(insufficient settled bets)")

    lines.append("")
    lines.append("=== By dimension ===")
    by_dim: dict[str, list[dict[str, Any]]] = {}
    for r in report.get("sleeves") or []:
        by_dim.setdefault(str(r.get("dimension")), []).append(r)

    for dim in SLEEVE_DIMENSIONS:
        rows = by_dim.get(dim) or []
        if not rows:
            continue
        lines.append(f"\n{SLEEVE_LABELS.get(dim, dim)}")
        for r in sorted(rows, key=lambda x: -int(x.get("n_bets") or 0))[:12]:
            lines.append(f"  {_sleeve_line(r).lstrip('- ')}")

    if report.get("csv_path"):
        lines.append("")
        lines.append(f"CSV: {report['csv_path']}")
    return "\n".join(lines)


def format_sleeve_dashboard_lines(
    report: dict[str, Any] | None = None,
    *,
    limit: int = 3,
) -> list[str]:
    """Compact lines for the Risk tab (top / bottom)."""
    data = report or run_sleeve_stats(write_csv=False)
    lines = [
        f"Sleeve performance ({data.get('n_rows', 0)} settled):",
    ]
    top = (data.get("top_sleeves") or [])[:limit]
    bottom = (data.get("bottom_sleeves") or [])[:limit]
    if top:
        lines.append("Top:")
        for r in top:
            lines.append(
                f"  + {r.get('dimension_label')}:{r.get('sleeve')} "
                f"n={r.get('n_bets')} hit={_fmt_pct(r.get('hit_rate'))} "
                f"ROI={_fmt_roi(r.get('roi'))}"
            )
    else:
        lines.append("Top: (need ≥3 settled bets per sleeve)")
    if bottom:
        lines.append("Bottom:")
        for r in bottom:
            lines.append(
                f"  - {r.get('dimension_label')}:{r.get('sleeve')} "
                f"n={r.get('n_bets')} hit={_fmt_pct(r.get('hit_rate'))} "
                f"ROI={_fmt_roi(r.get('roi'))}"
            )
    return lines


def run_sleeve_stats(
    *,
    csv_path: Path | str | None = None,
    write_csv: bool = True,
    min_n_rank: int = _MIN_N_RANK,
    bank_path: Path | str | None = None,
) -> dict[str, Any]:
    df = load_settled_sleeve_rows(bank_path=bank_path)
    stats = compute_sleeve_stats(df)
    top, bottom = rank_sleeves(stats, min_n=min_n_rank)

    n_bank = int((df["source"] == "prediction_bank").sum()) if not df.empty else 0
    n_journal = int((df["source"] == "bet_journal").sum()) if not df.empty else 0

    report: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_rows": int(len(df)),
        "n_bank": n_bank,
        "n_journal": n_journal,
        "sleeves": stats,
        "top_sleeves": top,
        "bottom_sleeves": bottom,
        "csv_path": None,
    }
    if write_csv:
        out = write_sleeve_stats_csv(stats, csv_path)
        report["csv_path"] = str(out)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.project_paths import bootstrap

    bootstrap(entry_file=config.ROOT_DIR / "main.py")
    p = argparse.ArgumentParser(description="Per-sleeve UFC bet / pick performance")
    p.add_argument(
        "--csv",
        metavar="PATH",
        help="Output CSV (default: reports/sleeve_stats_YYYYMMDD.csv)",
    )
    p.add_argument("--min-n", type=int, default=_MIN_N_RANK, help="Min bets to rank top/bottom")
    args = p.parse_args(argv)
    report = run_sleeve_stats(csv_path=args.csv, min_n_rank=args.min_n)
    print(format_sleeve_stats_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
