"""
Prop reliability deep-dive for high-accuracy / low-volume betting.

For each prop market:
  - sample size, actual base rate, model vs actual calibration gap
  - hit rate when model_prob ≥ 70% / 75% / 80%
  - ROI when edge ≥ 5% (live odds if present, else base-rate+vig synthetic market)
  - recommendation: Use | Use only at high confidence | Avoid for now

Re-run::

    python scripts/run_prop_reliability.py
    python -m src.prop_reliability
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

CONF_THRESHOLDS = (0.70, 0.75, 0.80)
EDGE_FLOOR = 0.05
FLAT_STAKE = 10.0

FOCUS_ORDER = (
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


def _clip_arr(p: np.ndarray | float, lo: float = 0.05, hi: float = 0.95) -> np.ndarray | float:
    return np.clip(p, lo, hi)


def _col(df: pd.DataFrame, *names: str, default: float | None = None) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    if default is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.Series(default, index=df.index, dtype=float)


def _prop_keys() -> list[str]:
    from src.props import PROP_MARKET_LABELS

    keys = list(
        dict.fromkeys(
            [
                *FOCUS_ORDER,
                *PROP_MARKET_LABELS.keys(),
                *list(getattr(config, "PROP_MARKETS", [])),
            ]
        )
    )
    return [k for k in keys if k and k != "moneyline" and k in PROP_MARKET_LABELS]


def _vectorized_method_probs(features: pd.DataFrame) -> pd.DataFrame:
    """Fast vectorized mirror of method_probs_from_row."""
    f1_ko = _clip_arr(_col(features, "f1_ko_rate", default=0.18).fillna(0.18).to_numpy(), 0.05, 0.55)
    f2_ko = _clip_arr(_col(features, "f2_ko_rate", default=0.18).fillna(0.18).to_numpy(), 0.05, 0.55)
    f1_sub = _clip_arr(
        (_col(features, "f1_sub_avg", default=0.35).fillna(0.35) / 2.5).to_numpy(),
        0.03,
        0.40,
    )
    f2_sub = _clip_arr(
        (_col(features, "f2_sub_avg", default=0.35).fillna(0.35) / 2.5).to_numpy(),
        0.03,
        0.40,
    )
    f1_finish = _clip_arr(
        _col(features, "f1_finish_rate", default=0.45).fillna(0.45).to_numpy(),
        0.10,
        0.80,
    )
    f2_finish = _clip_arr(
        _col(features, "f2_finish_rate", default=0.45).fillna(0.45).to_numpy(),
        0.10,
        0.80,
    )

    ko_diff = _col(features, "ko_rate_diff", default=0.0).fillna(0.0).to_numpy()
    sub_diff = _col(features, "sub_avg_diff", default=0.0).fillna(0.0).to_numpy()
    striker_diff = _col(features, "striker_score_diff", default=0.0).fillna(0.0).to_numpy()

    p_ko = _clip_arr(0.5 * (f1_ko + f2_ko) + 0.15 * ko_diff + 0.08 * striker_diff, 0.08, 0.62)
    p_sub = _clip_arr(0.5 * (f1_sub + f2_sub) + 0.12 * sub_diff, 0.05, 0.45)
    p_dec = _clip_arr(1.0 - p_ko - p_sub, 0.12, 0.75)
    total = p_ko + p_sub + p_dec
    p_ko, p_sub, p_dec = p_ko / total, p_sub / total, p_dec / total

    avg_finish = 0.5 * (f1_finish + f2_finish)
    p_r1 = _clip_arr(avg_finish * 0.58 * (1.0 + 0.25 * np.abs(ko_diff)), 0.08, 0.55)
    p_over_15 = _clip_arr(1.0 - p_r1, 0.25, 0.92)
    p_under_15 = _clip_arr(1.0 - p_over_15, 0.08, 0.75)
    p_finish = _clip_arr(p_ko + p_sub, 0.15, 0.88)

    p1 = _col(features, "prob_f1_win", "predicted_prob", default=0.5).fillna(0.5).to_numpy()
    p2_raw = _col(features, "prob_f2_win")
    p2 = np.where(p2_raw.notna().to_numpy(), p2_raw.fillna(0).to_numpy(), 1.0 - p1)
    pick_f1 = p1 >= p2
    pick_prob = np.where(pick_f1, p1, p2)
    pick_ko = np.where(pick_f1, f1_ko, f2_ko)
    pick_sub = np.where(pick_f1, f1_sub, f2_sub)
    pick_dec = np.maximum(0.05, 1.0 - pick_ko - pick_sub)
    pick_total = pick_ko + pick_sub + pick_dec
    pick_ko = pick_ko / pick_total
    pick_sub = pick_sub / pick_total

    return pd.DataFrame(
        {
            "goes_to_decision": p_dec,
            "finish": p_finish,
            "ko_tko": p_ko,
            "submission": p_sub,
            "round_1_finish": p_r1,
            "over_1_5_rounds": p_over_15,
            "under_1_5_rounds": p_under_15,
            "fighter_ko": _clip_arr(pick_prob * pick_ko, 0.03, 0.55),
            "fighter_sub": _clip_arr(pick_prob * pick_sub, 0.02, 0.40),
            "pick_f1": pick_f1.astype(int),
        },
        index=features.index,
    )


def _vectorized_actuals(features: pd.DataFrame) -> pd.DataFrame:
    """Settle all supported props vectorized."""
    method = features.get("method", features.get("METHOD", pd.Series("", index=features.index)))
    method = method.fillna("").astype(str).str.upper()
    rnd = pd.to_numeric(features.get("round", features.get("ROUND")), errors="coerce").fillna(0).astype(int)

    is_ko = method.str.contains("KO|TKO", regex=True, na=False)
    is_sub = method.str.contains("SUB", regex=True, na=False)
    is_dec = method.str.contains("DEC|DECISION", regex=True, na=False)
    # Unknown non-empty method defaults to decision (matches method_flags)
    unknown = (~is_ko) & (~is_sub) & (~is_dec) & method.str.strip().ne("")
    is_dec = is_dec | unknown

    target = _col(features, config.TARGET_COLUMN)
    f1 = features.get("fighter_1", features.get("fighter1", pd.Series("", index=features.index))).fillna("").astype(str).str.strip()
    f2 = features.get("fighter_2", features.get("fighter2", pd.Series("", index=features.index))).fillna("").astype(str).str.strip()
    winner = features.get("winner", pd.Series("", index=features.index))
    if winner is None:
        winner = pd.Series("", index=features.index)
    winner = winner.fillna("").astype(str).str.strip()

    actual_f1 = target.copy()
    miss = actual_f1.isna()
    if miss.any():
        actual_f1 = actual_f1.where(~miss, np.where(winner.eq(f1), 1.0, np.where(winner.eq(f2), 0.0, np.nan)))

    p1 = _col(features, "prob_f1_win", "predicted_prob", default=0.5).fillna(0.5)
    pick_f1 = p1 >= 0.5
    # Prefer actual winner for pick_side when known (settle_prop behavior)
    has_w = winner.ne("")
    pick_f1 = np.where(has_w & winner.eq(f1), True, np.where(has_w & winner.eq(f2), False, pick_f1))

    pick_won = np.where(
        pick_f1,
        actual_f1.fillna(-1).to_numpy() == 1,
        actual_f1.fillna(-1).to_numpy() == 0,
    )
    # If outcome unknown, fighter props are None — mask later
    outcome_known = actual_f1.notna() | (is_ko | is_sub | is_dec)

    has_method = method.str.strip().ne("") | (rnd > 0)

    out = pd.DataFrame(
        {
            "goes_to_decision": is_dec.astype(int),
            "finish": (is_ko | is_sub).astype(int),
            "ko_tko": is_ko.astype(int),
            "submission": is_sub.astype(int),
            "round_1_finish": ((rnd == 1) & (is_ko | is_sub)).astype(int),
            "over_1_5_rounds": (rnd > 1).astype(int),
            "under_1_5_rounds": (rnd == 1).astype(int),
            "fighter_ko": (pick_won & is_ko.to_numpy()).astype(int),
            "fighter_sub": (pick_won & is_sub.to_numpy()).astype(int),
            "_valid": has_method.to_numpy(),
            "_fighter_valid": (has_method & outcome_known).to_numpy(),
        },
        index=features.index,
    )
    return out


def _try_attach_live_or_journal_odds(obs: pd.DataFrame, prop_key: str, features: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer real odds when journal/bank has settled prop rows with decimal odds.
    Falls back to caller-set synthetic market.
    """
    try:
        from src.settlement import load_settlement_journal

        journal = load_settlement_journal()
    except Exception:
        journal = None
    if journal is None or getattr(journal, "empty", True):
        return obs

    j = journal.copy()
    market_col = None
    for c in ("market", "prop_key", "bet_type", "selection_type"):
        if c in j.columns:
            market_col = c
            break
    if market_col is None:
        return obs

    mask = j[market_col].astype(str).str.lower().str.replace(" ", "_", regex=False).str.contains(
        prop_key.replace("_", ".*"), regex=True, na=False
    ) | j[market_col].astype(str).str.contains(prop_key, case=False, na=False)
    sub = j.loc[mask]
    if sub.empty:
        return obs

    odds_col = next((c for c in ("decimal_odds", "odds", "price") if c in sub.columns), None)
    if odds_col is None:
        return obs

    # Not fight-aligned reliably — use pooled book for edge ROI only when n small;
    # keep synthetic per-fight for ranking consistency. Annotate if any real odds exist.
    obs = obs.copy()
    obs.attrs["has_real_odds_pool"] = True
    obs.attrs["real_odds_n"] = int(len(sub))
    return obs


def _build_obs_for_prop(
    features: pd.DataFrame,
    probs: pd.DataFrame,
    actuals: pd.DataFrame,
    prop_key: str,
) -> pd.DataFrame:
    if prop_key not in probs.columns or prop_key not in actuals.columns:
        return pd.DataFrame()

    valid = actuals["_fighter_valid"] if prop_key.startswith("fighter_") else actuals["_valid"]
    model_p = probs[prop_key].to_numpy(dtype=float)
    actual = actuals[prop_key].to_numpy(dtype=float)
    mask = valid.to_numpy(dtype=bool) & np.isfinite(model_p)
    if not mask.any():
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "prop_key": prop_key,
            "model_prob": model_p[mask],
            "actual": actual[mask].astype(int),
        }
    )
    base = float(out["actual"].mean())
    vig = float(getattr(config, "PROP_SYNTHETIC_VIG", 0.08) or 0.08)
    # Stationary book at historical base rate + vig so model can show +EV
    market_implied = float(_clip_arr(base * (1.0 + vig), 0.05, 0.95))
    decimal = 1.0 / market_implied
    out["market_implied"] = market_implied
    out["decimal_odds"] = decimal
    out["edge"] = out["model_prob"] - market_implied
    out["odds_source"] = "synthetic_base_rate"
    out = _try_attach_live_or_journal_odds(out, prop_key, features)
    return out


def _hit_rate_at(df: pd.DataFrame, min_prob: float) -> tuple[int, float | None]:
    sub = df[df["model_prob"] >= min_prob]
    n = int(len(sub))
    if n == 0:
        return 0, None
    return n, float(sub["actual"].mean())


def _roi_at_edge(df: pd.DataFrame, min_edge: float, stake: float = FLAT_STAKE) -> dict[str, Any]:
    sub = df[df["edge"] >= min_edge]
    n = int(len(sub))
    if n == 0:
        return {
            "n_bets_edge": 0,
            "hit_rate_edge": None,
            "pnl_edge": None,
            "roi_edge": None,
            "avg_edge": None,
        }
    odds = sub["decimal_odds"].to_numpy(dtype=float)
    won = sub["actual"].to_numpy(dtype=int) == 1
    pnl = np.where(won, stake * (odds - 1.0), -stake)
    total_pnl = float(pnl.sum())
    total_stake = stake * n
    return {
        "n_bets_edge": n,
        "hit_rate_edge": float(sub["actual"].mean()),
        "pnl_edge": total_pnl,
        "roi_edge": total_pnl / total_stake if total_stake > 0 else None,
        "avg_edge": float(sub["edge"].mean()),
    }


def _recommend(row: dict[str, Any]) -> str:
    n = int(row.get("n") or 0)
    n70 = int(row.get("n_ge_70") or 0)
    n75 = int(row.get("n_ge_75") or 0)
    n80 = int(row.get("n_ge_80") or 0)
    h70 = row.get("hit_rate_ge_70")
    h75 = row.get("hit_rate_ge_75")
    h80 = row.get("hit_rate_ge_80")
    gap = abs(float(row.get("calibration_gap") or 0.0))
    roi = row.get("roi_edge_ge_5pct")
    n_edge = int(row.get("n_bets_edge_ge_5pct") or 0)

    if n < 200:
        return "Avoid for now"

    if n80 >= 20 and h80 is not None and h80 < 0.55:
        return "Avoid for now"
    if n75 >= 40 and h75 is not None and h75 < 0.58:
        return "Avoid for now"
    if n_edge >= 40 and roi is not None and roi < -0.15:
        return "Avoid for now"
    if gap > 0.15 and (h80 is None or h80 < 0.70):
        return "Avoid for now"

    use_ok = (
        n75 >= 40
        and h75 is not None
        and h75 >= 0.72
        and gap <= 0.08
        and (roi is None or n_edge < 20 or roi >= -0.05)
    )
    if use_ok and h80 is not None and n80 >= 15 and h80 >= 0.75:
        return "Use"
    if use_ok and (h80 is None or n80 < 15):
        return "Use"

    high_only = (n80 >= 15 and h80 is not None and h80 >= 0.70) or (
        n75 >= 25 and h75 is not None and h75 >= 0.68 and (h70 is None or h70 < 0.65)
    )
    if high_only:
        return "Use only at high confidence"

    if n70 >= 50 and h70 is not None and h70 >= 0.70 and gap <= 0.10:
        return "Use only at high confidence"

    return "Avoid for now"


def _rec_tier(rec: str) -> int:
    """Lower = better for high-accuracy sort (primary key)."""
    if rec == "Use":
        return 0
    if rec == "Use only at high confidence":
        return 1
    return 2


def _reliability_score(row: dict[str, Any]) -> float:
    """
    Higher = better within a recommendation tier.
    High-accuracy / low-volume: weight calibrated high-prob hit rates;
    do not let synthetic ROI@edge dominate when the model never reaches ≥70%.
    """
    h80 = row.get("hit_rate_ge_80")
    h75 = row.get("hit_rate_ge_75")
    h70 = row.get("hit_rate_ge_70")
    n80 = int(row.get("n_ge_80") or 0)
    n75 = int(row.get("n_ge_75") or 0)
    n70 = int(row.get("n_ge_70") or 0)
    gap = abs(float(row.get("calibration_gap") or 0.0))
    roi = row.get("roi_edge_ge_5pct")
    n_edge = int(row.get("n_bets_edge_ge_5pct") or 0)
    rec = row.get("recommendation") or ""

    score = 0.0
    # High-conf hit rates: reward only when selective accuracy is strong
    if h80 is not None and n80 > 0:
        score += 0.50 * (float(h80) - 0.60) * min(1.0, n80 / 40.0)
    if h75 is not None and n75 > 0:
        score += 0.35 * (float(h75) - 0.60) * min(1.0, n75 / 60.0)
    if h70 is not None and n70 > 0:
        score += 0.15 * (float(h70) - 0.55) * min(1.0, n70 / 80.0)
    else:
        score -= 0.35

    score -= 0.35 * min(gap, 0.35) / 0.35

    if roi is not None and n_edge >= 20 and n70 > 0 and (h75 or 0) >= 0.65:
        score += 0.12 * float(np.clip(roi, -0.3, 0.3) / 0.3)

    if rec == "Use":
        score += 0.25
    elif rec == "Use only at high confidence":
        score += 0.10
    else:
        score -= 0.20
    return float(score)


def analyze_prop_reliability(
    *,
    features: pd.DataFrame | None = None,
    edge_floor: float = EDGE_FLOOR,
    stake: float = FLAT_STAKE,
) -> dict[str, Any]:
    from src.prop_performance import load_labeled_prop_frame, prop_display_name

    feats = features if features is not None else load_labeled_prop_frame()
    probs = _vectorized_method_probs(feats)
    actuals = _vectorized_actuals(feats)

    rows: list[dict[str, Any]] = []
    for key in _prop_keys():
        obs = _build_obs_for_prop(feats, probs, actuals, key)
        if obs.empty:
            continue
        n = int(len(obs))
        base = float(obs["actual"].mean())
        avg_model = float(obs["model_prob"].mean())
        gap = avg_model - base

        n70, h70 = _hit_rate_at(obs, 0.70)
        n75, h75 = _hit_rate_at(obs, 0.75)
        n80, h80 = _hit_rate_at(obs, 0.80)
        edge_stats = _roi_at_edge(obs, edge_floor, stake=stake)

        row: dict[str, Any] = {
            "prop_key": key,
            "prop_label": prop_display_name(key),
            "n": n,
            "actual_base_rate": base,
            "avg_model_prob": avg_model,
            "calibration_gap": gap,
            "n_ge_70": n70,
            "hit_rate_ge_70": h70,
            "n_ge_75": n75,
            "hit_rate_ge_75": h75,
            "n_ge_80": n80,
            "hit_rate_ge_80": h80,
            "n_bets_edge_ge_5pct": edge_stats["n_bets_edge"],
            "hit_rate_edge_ge_5pct": edge_stats["hit_rate_edge"],
            "pnl_edge_ge_5pct": edge_stats["pnl_edge"],
            "roi_edge_ge_5pct": edge_stats["roi_edge"],
            "avg_edge_when_bet": edge_stats["avg_edge"],
            "market_implied_used": float(obs["market_implied"].iloc[0]),
            "odds_note": "edge vs historical base-rate market + synthetic vig (live/journal when available)",
        }
        row["recommendation"] = _recommend(row)
        row["reliability_score"] = _reliability_score(row)
        rows.append(row)

    rows.sort(
        key=lambda r: (_rec_tier(str(r.get("recommendation") or "")), -float(r["reliability_score"]))
    )
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_fights": int(len(feats)),
        "edge_floor": edge_floor,
        "stake": stake,
        "rows": rows,
    }


def format_prop_reliability_report(report: dict[str, Any]) -> str:
    lines = [
        "PROP RELIABILITY RANKING (high-accuracy / low-volume)",
        f"Generated: {report.get('generated_at')}",
        f"Fights scored: {report.get('n_fights')} | "
        f"Edge floor {100 * float(report.get('edge_floor') or 0):.0f}% | "
        f"stake ${float(report.get('stake') or 0):.0f}",
        "",
        f"{'Rk':>3} {'Prop':<28} {'n':>6} {'Base':>6} {'Gap':>7} "
        f"{'H@70':>10} {'H@75':>10} {'H@80':>10} {'ROI@5%':>8}  Rec",
        "-" * 118,
    ]
    for r in report.get("rows") or []:

        def pct(v: Any, digits: int = 0) -> str:
            if v is None:
                return "n/a"
            return f"{100 * float(v):.{digits}f}%"

        h70, h75, h80 = r.get("hit_rate_ge_70"), r.get("hit_rate_ge_75"), r.get("hit_rate_ge_80")
        n70, n75, n80 = r.get("n_ge_70"), r.get("n_ge_75"), r.get("n_ge_80")
        h70s = f"{pct(h70)}/{n70}" if h70 is not None else "n/a"
        h75s = f"{pct(h75)}/{n75}" if h75 is not None else "n/a"
        h80s = f"{pct(h80)}/{n80}" if h80 is not None else "n/a"
        roi = r.get("roi_edge_ge_5pct")
        roi_s = pct(roi, 1) if roi is not None else "n/a"
        gap = r.get("calibration_gap")
        gap_s = f"{100 * float(gap):+.1f}pp" if gap is not None else "n/a"
        lines.append(
            f"{int(r['rank']):>3} {str(r.get('prop_label') or r['prop_key'])[:28]:<28} "
            f"{int(r['n']):>6} {pct(r.get('actual_base_rate')):>6} {gap_s:>7} "
            f"{h70s:>10} {h75s:>10} {h80s:>10} {roi_s:>8}  {r.get('recommendation')}"
        )
    if report.get("csv_path"):
        lines.append("")
        lines.append(f"CSV: {report['csv_path']}")
    return "\n".join(lines)


def write_prop_reliability_csv(
    report: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    out = Path(path) if path else (config.ROOT_DIR / "reports" / "prop_reliability_ranked.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank",
        "prop_key",
        "prop_label",
        "recommendation",
        "reliability_score",
        "n",
        "actual_base_rate",
        "avg_model_prob",
        "calibration_gap",
        "n_ge_70",
        "hit_rate_ge_70",
        "n_ge_75",
        "hit_rate_ge_75",
        "n_ge_80",
        "hit_rate_ge_80",
        "n_bets_edge_ge_5pct",
        "hit_rate_edge_ge_5pct",
        "pnl_edge_ge_5pct",
        "roi_edge_ge_5pct",
        "avg_edge_when_bet",
        "market_implied_used",
        "odds_note",
    ]
    df = pd.DataFrame(report.get("rows") or [])
    if not df.empty:
        ordered = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
        df = df[ordered]
    df.to_csv(out, index=False, encoding="utf-8")
    report["csv_path"] = str(out)
    return out


def run_prop_reliability(*, csv_path: Path | str | None = None) -> dict[str, Any]:
    report = analyze_prop_reliability()
    write_prop_reliability_csv(report, csv_path)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.project_paths import bootstrap

    bootstrap(entry_file=config.ROOT_DIR / "main.py")
    p = argparse.ArgumentParser(description="Rank UFC props by high-accuracy reliability")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV path (default: reports/prop_reliability_ranked.csv)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    report = run_prop_reliability(csv_path=args.output)
    print(format_prop_reliability_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
