"""Read-only fight context for dashboard overlay — display only, not model inputs.

Uses columns already present on prediction rows / feature cache, plus optional
mmadecisions cache for judge names. Never trains or mutates pathway/market flags
or ensemble artifacts.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.data_loader import clean_fighter_name
from src.judge_geography import format_panel_geography_note, judge_country
from src.home_country import location_to_country

logger = logging.getLogger(__name__)

# 2025 holdout tertiles (ci_width_calibration_2025): low acc~0.91, high~0.62.
_DISAGREE_LOW_MAX = 0.008
_DISAGREE_HIGH_MIN = 0.037


def _f(row: pd.Series | dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    get = row.get if isinstance(row, dict) else row.get
    for k in keys:
        try:
            v = get(k)
        except Exception:
            continue
        if v is None:
            continue
        try:
            if isinstance(v, float) and pd.isna(v):
                continue
            if pd.isna(v):
                continue
        except Exception:
            pass
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return default


def _pct(v: float | None) -> str:
    if v is None:
        return ""
    return f"{100.0 * float(v):.0f}%"


def _get(row: pd.Series | dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        return row.get(key, default) if hasattr(row, "get") else default
    except Exception:
        return default


def _method_snapshot(row: pd.Series | dict[str, Any]) -> str | None:
    """KO/sub/dec win-rate snapshot if pathway or legacy columns exist."""
    parts: list[str] = []
    ko = _f(row, "ko_win_rate_l5_diff", "f1_ko_win_rate_l5", "ko_rate_diff", "f1_ko_rate")
    sub = _f(row, "sub_win_rate_l5_diff", "f1_sub_win_rate_l5", "sub_avg_diff", "f1_sub_avg")
    dec = _f(row, "dec_win_rate_l5_diff", "f1_dec_win_rate_l5", "dec_win_rate_l5_diff")
    if ko is not None:
        parts.append(f"KO {_pct(abs(ko)) if abs(ko) <= 1 else f'{ko:.0f}%'}")
    if sub is not None:
        if abs(sub) <= 1.5:
            parts.append(f"Sub {_pct(abs(sub))}")
        else:
            parts.append(f"Sub {sub:.2f}")
    if dec is not None and abs(dec) <= 1.5:
        parts.append(f"Dec {_pct(abs(dec))}")
    if not parts:
        return None
    return "Method: " + " | ".join(parts)


def _decision_profile_line(row: pd.Series | dict[str, Any]) -> str | None:
    """Judge-agnostic decision profile for the pick (or f1) when columns exist."""
    pick = str(
        _get(row, "predicted_winner") or _get(row, "pick") or ""
    ).strip()
    f1 = str(_get(row, "fighter_1") or _get(row, "fighter1") or "").strip()
    use_f2 = bool(pick and f1 and pick != f1)

    def side(*keys: str) -> float | None:
        pref = "f2_" if use_f2 else "f1_"
        expanded: list[str] = []
        for k in keys:
            if k.startswith("f1_") or k.startswith("f2_"):
                expanded.append(k)
            else:
                expanded.append(pref + k)
                expanded.append(k)
        return _f(row, *expanded)

    dec_w = side("dec_win_rate_l5", "dec_win_rate_career")
    split_w = side("split_dec_win_rate_l5", "split_dec_win_rate_career")
    share = side("decision_finish_share_l5", "decision_finish_share_career")
    # Diffs as fallback (signed toward f1)
    if dec_w is None:
        dec_w = _f(row, "dec_win_rate_l5_diff", "dec_win_rate_career_diff")
    if split_w is None:
        split_w = _f(row, "split_dec_win_rate_l5_diff", "split_dec_win_rate_career_diff")
    if share is None:
        share = _f(
            row,
            "decision_finish_share_l5_diff",
            "decision_finish_share_career_diff",
        )

    parts: list[str] = []
    if dec_w is not None and abs(dec_w) <= 1.5:
        parts.append(f"dec-win {_pct(abs(dec_w)) if abs(dec_w) <= 1 else f'{dec_w:.0f}%'}")
    if split_w is not None and abs(split_w) <= 1.5:
        parts.append(f"split-win {_pct(abs(split_w))}")
    if share is not None and abs(share) <= 1.5:
        parts.append(f"dec-share {_pct(abs(share))}")
    if not parts:
        return None
    who = pick or ("f2" if use_f2 else "f1")
    return f"Decision profile ({who}): " + " | ".join(parts)


def _style_clash_line(row: pd.Series | dict[str, Any]) -> str | None:
    """One-liner TD pressure / style clash from existing stats."""
    td = _f(row, "path_opp_td_att_x_own_td_def", "hv_td_pressure_diff", "td_defense_diff")
    ko_clash = _f(row, "path_opp_ko_x_own_ko_loss")
    stance = _get(row, "path_stance_mismatch")
    if stance is None:
        stance = _get(row, "stance_matchup")
    bits: list[str] = []
    if td is not None and abs(td) > 1e-6:
        bits.append("TD pressure favors " + ("f1" if td > 0 else "f2"))
    if ko_clash is not None and abs(ko_clash) > 1e-6:
        bits.append("KO threat vs chin mismatch")
    try:
        if stance is not None and float(stance) >= 0.5:
            bits.append("stance mismatch (SP vs Orth)")
    except (TypeError, ValueError):
        pass
    style = _get(row, "striker_vs_grappler")
    try:
        if style is not None and float(style) >= 0.5:
            bits.append("striker vs grappler")
    except (TypeError, ValueError):
        pass
    if not bits:
        return None
    return "Style: " + "; ".join(bits)


def _market_line(row: pd.Series | dict[str, Any]) -> str:
    """Implied vs model when odds exist."""
    mkt = _f(row, "mkt_implied_prob", "implied_prob_f1")
    pick = str(_get(row, "predicted_winner") or _get(row, "pick") or "").strip()
    f1 = str(_get(row, "fighter_1") or "").strip()
    model = _f(row, "predicted_prob", "prob_f1_win")
    if pick and f1 and pick != f1:
        model = _f(row, "prob_f2_win", "predicted_prob")
        if mkt is not None:
            mkt = 1.0 - mkt
    has_odds = bool(_get(row, "odds_matched")) or (
        _get(row, "f1_odds") is not None
        and not (
            isinstance(_get(row, "f1_odds"), float) and pd.isna(_get(row, "f1_odds"))
        )
    )
    if not has_odds and mkt is None:
        return "Market: n/a"
    if mkt is None:
        return "Market: n/a"
    model_txt = _pct(model) if model is not None else "—"
    return f"Market: {_pct(mkt)} implied | model {model_txt}"


def disagreement_band(disagreement: float | None) -> str | None:
    """Map ensemble disagreement to low / mid / high (display bands)."""
    if disagreement is None:
        return None
    d = float(disagreement)
    if d <= _DISAGREE_LOW_MAX:
        return "low"
    if d >= _DISAGREE_HIGH_MIN:
        return "high"
    return "mid"


def _disagreement_badge(row: pd.Series | dict[str, Any]) -> str | None:
    d = _f(row, "ensemble_disagreement", "disagreement", "model_disagreement")
    band = disagreement_band(d)
    if band is None or d is None:
        return None
    return f"Disagree: {band} ({d:.3f})"


def _last(name: Any) -> str:
    parts = clean_fighter_name(name).split()
    return parts[-1].lower() if parts else ""


def _pair_key(a: Any, b: Any) -> tuple[str, str]:
    return tuple(sorted([_last(a), _last(b)]))


@lru_cache(maxsize=2)
def _judge_index(cache_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = Path(cache_path)
    if not path.is_file():
        return {}
    idx: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _pair_key(row.get("fighter_1"), row.get("fighter_2"))
            if not key[0] or not key[1]:
                continue
            judges = row.get("judges") or []
            names = [str(j.get("judge_name") or "") for j in judges if j.get("judge_name")]
            if not names:
                continue
            ec = str(row.get("event_country") or "") or location_to_country(
                row.get("event_location")
            )
            j_countries = [judge_country(n) for n in names]
            share = (
                sum(1 for c in j_countries if c and ec and c == ec) / len(names)
                if names
                else None
            )
            idx[key] = {
                "judge_names": names,
                "event_country": ec,
                "panel_event_country_share": share,
                "decision_id": row.get("decision_id"),
            }
    except Exception as exc:
        logger.debug("judge index load failed: %s", exc)
        return {}
    return idx


def _judges_line(row: pd.Series | dict[str, Any]) -> str | None:
    """Show assigned judges when known (cache or row fields). Display only."""
    # Explicit row fields first (upcoming cards may attach later)
    names_raw = _get(row, "judge_names") or _get(row, "judges")
    if isinstance(names_raw, str) and names_raw.strip():
        names = [n.strip() for n in names_raw.replace("|", ";").split(";") if n.strip()]
        share = _f(row, "panel_event_country_share")
        ec = str(_get(row, "event_country") or "") or location_to_country(
            _get(row, "location") or _get(row, "event_location")
        )
        note = format_panel_geography_note(
            names, panel_event_country_share=share, event_country=ec or None
        )
        return note or None

    f1 = _get(row, "fighter_1") or _get(row, "fighter1")
    f2 = _get(row, "fighter_2") or _get(row, "fighter2")
    if not f1 or not f2:
        return None

    enriched = config.CACHE_DIR / "mmadecisions" / "decisions_with_location.jsonl"
    raw = config.CACHE_DIR / "mmadecisions" / "decisions.jsonl"
    path = enriched if enriched.is_file() else raw
    if not path.is_file():
        return None
    hit = _judge_index(str(path)).get(_pair_key(f1, f2))
    if not hit:
        return None
    return format_panel_geography_note(
        hit["judge_names"],
        panel_event_country_share=hit.get("panel_event_country_share"),
        event_country=hit.get("event_country") or None,
    )


def build_fight_context(row: pd.Series | dict[str, Any] | None) -> dict[str, str]:
    """
    Return display sections (omit empty).

    Keys: method, decision, style, market, judges, disagree, title, summary.
    """
    if row is None:
        return {}
    series: pd.Series | dict[str, Any] = row

    out: dict[str, str] = {}
    method = _method_snapshot(series)
    if method:
        out["method"] = method
    decision = _decision_profile_line(series)
    if decision:
        out["decision"] = decision
    style = _style_clash_line(series)
    if style:
        out["style"] = style
    out["market"] = _market_line(series)
    judges = _judges_line(series)
    if judges:
        out["judges"] = judges
    disagree = _disagreement_badge(series)
    if disagree:
        out["disagree"] = disagree

    f1 = str(_get(series, "fighter_1") or "")
    f2 = str(_get(series, "fighter_2") or "")
    try:
        from src.fighter_flags import format_flag_badge

        badge = format_flag_badge(f1, f2)
        if badge:
            out["integrity"] = badge
    except Exception:
        pass
    try:
        from src.weigh_in import format_weigh_in_line

        event = str(_get(series, "event_name") or _get(series, "event") or "")
        date = str(_get(series, "event_date") or _get(series, "date") or "")
        weigh = format_weigh_in_line(f1, f2, event=event or None, date=date or None)
        if weigh:
            out["weigh_in"] = weigh
    except Exception:
        pass
    pick = str(_get(series, "predicted_winner") or _get(series, "pick") or "")
    title = f"{f1} vs {f2}" if f1 and f2 else "Fight"
    if pick:
        title = f"{title}  |  pick {pick}"
    out["title"] = title
    bits = [
        out[k]
        for k in (
            "integrity",
            "weigh_in",
            "method",
            "decision",
            "style",
            "market",
            "judges",
            "disagree",
        )
        if k in out
    ]
    out["summary"] = "  ·  ".join(bits) if bits else out.get("market", "No context")
    return out


def format_fight_context_lines(ctx: dict[str, str]) -> list[str]:
    """Ordered lines for a detail strip / overlay."""
    if not ctx:
        return ["Select a fight row for context."]
    lines: list[str] = []
    if ctx.get("title"):
        lines.append(ctx["title"])
    for key in (
        "integrity",
        "weigh_in",
        "method",
        "decision",
        "style",
        "market",
        "judges",
        "disagree",
    ):
        if ctx.get(key):
            lines.append(ctx[key])
    return lines or [ctx.get("summary", "No context")]
