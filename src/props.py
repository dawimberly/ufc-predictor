"""UFC prop betting: method, rounds, decision markets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

import config
from src.strategy import BetCandidate, StrategyConfig, bet_expected_value


PROP_MARKET_LABELS: dict[str, str] = {
    "goes_to_decision": "Goes to Decision",
    "finish": "Fight Ends Inside Distance",
    "ko_tko": "KO/TKO",
    "submission": "Submission",
    "round_1_finish": "Round 1 Finish",
    "over_1_5_rounds": "Over 1.5 Rounds",
    "under_1_5_rounds": "Under 1.5 Rounds",
    "over_2_5_rounds": "Over 2.5 Rounds",
    "under_2_5_rounds": "Under 2.5 Rounds",
    "over_3_5_rounds": "Over 3.5 Rounds",
    "under_3_5_rounds": "Under 3.5 Rounds",
    "over_4_5_rounds": "Over 4.5 Rounds",
    "under_4_5_rounds": "Under 4.5 Rounds",
    "fighter_ko": "Fighter Wins by KO/TKO",
    "fighter_sub": "Fighter Wins by Submission",
    "fighter_decision": "Fighter Wins by Decision",
}

_FIGHTER_METHOD_KEYS = frozenset({"fighter_ko", "fighter_sub", "fighter_decision"})
_METHOD_KEY_TO_SIDE = {
    "fighter_ko": "ko",
    "fighter_sub": "sub",
    "fighter_decision": "dec",
}

_OVER_UNDER_POINT_RE = re.compile(r"^(over|under)_(\d+)_(\d+)_rounds$")


def method_flags(method: Any) -> tuple[int, int, int]:
    """Return (is_ko, is_sub, is_dec) from a method string."""
    text = str(method or "").upper()
    is_ko = int("KO" in text or "TKO" in text)
    is_sub = int("SUB" in text)
    is_dec = int("DEC" in text or "DECISION" in text)
    if not is_ko and not is_sub and not is_dec and text.strip():
        is_dec = 1
    return is_ko, is_sub, is_dec


def _clip_prob(p: float, lo: float = 0.03, hi: float = 0.97) -> float:
    if not np.isfinite(p):
        return lo
    return float(np.clip(p, lo, hi))


def _over_line_from_anchors(
    p_over_15: float,
    p_dec: float,
    *,
    point: float,
    max_line: float,
) -> float:
    """P(Over point) between Over 1.5 and go-the-distance (decision)."""
    if point <= 1.5 + 1e-9:
        return _clip_prob(p_over_15, 0.20, 0.95)
    if max_line <= 1.5 + 1e-9 or point >= max_line - 1e-9:
        return _clip_prob(p_dec, 0.10, 0.95)
    t = (point - 1.5) / (max_line - 1.5)
    t = float(np.clip(t, 0.0, 1.0))
    # Keep deeper overs from exceeding shallower when models disagree
    p_deep = min(float(p_over_15), float(p_dec))
    return _clip_prob(float(p_over_15) * (1.0 - t) + p_deep * t, 0.10, 0.95)


def _fighter_rate(row: pd.Series, prefix: str, key: str, default: float) -> float:
    for col in (f"{prefix}_{key}", f"f1_{key}" if prefix == "f1" else f"f2_{key}"):
        if col in row.index and pd.notna(row.get(col)):
            try:
                return float(row[col])
            except (TypeError, ValueError):
                continue
    return default


def _fighter_rate_first(
    row: pd.Series,
    prefix: str,
    keys: tuple[str, ...],
    default: float | None,
) -> float | None:
    """First available fighter rate among keys; None if all missing and default is None."""
    for key in keys:
        for col in (f"{prefix}_{key}",):
            if col in row.index and pd.notna(row.get(col)):
                try:
                    return float(row[col])
                except (TypeError, ValueError):
                    continue
    return default


def method_probs_from_row(row: pd.Series) -> dict[str, float]:
    """
    Estimate fight-level method probabilities from rolling fighter stats.

    Prefers L5 pathway method win/loss + finish / R1 / decision-profile fields when
    present (props-only; not LightGBM FEATURE_COLUMNS). Returns keys: ko, sub, dec,
    round_1_finish, over_1_5_rounds, fighter_ko, fighter_sub (conditional on model pick).
    """
    # Prefer pathway L5 method win rates; fall back to legacy ko_rate / sub_avg.
    f1_ko = _fighter_rate_first(
        row, "f1", ("ko_win_rate_l5", "ko_win_rate_career", "ko_rate"), None
    )
    f2_ko = _fighter_rate_first(
        row, "f2", ("ko_win_rate_l5", "ko_win_rate_career", "ko_rate"), None
    )
    f1_ko = _clip_prob(float(f1_ko if f1_ko is not None else 0.18), 0.05, 0.55)
    f2_ko = _clip_prob(float(f2_ko if f2_ko is not None else 0.18), 0.05, 0.55)

    f1_sub_raw = _fighter_rate_first(
        row, "f1", ("sub_win_rate_l5", "sub_win_rate_career"), None
    )
    f2_sub_raw = _fighter_rate_first(
        row, "f2", ("sub_win_rate_l5", "sub_win_rate_career"), None
    )
    if f1_sub_raw is None:
        f1_sub = _clip_prob(_fighter_rate(row, "f1", "sub_avg", 0.35) / 2.5, 0.03, 0.40)
    else:
        f1_sub = _clip_prob(float(f1_sub_raw), 0.03, 0.40)
    if f2_sub_raw is None:
        f2_sub = _clip_prob(_fighter_rate(row, "f2", "sub_avg", 0.35) / 2.5, 0.03, 0.40)
    else:
        f2_sub = _clip_prob(float(f2_sub_raw), 0.03, 0.40)

    f1_finish = _clip_prob(
        float(
            _fighter_rate_first(row, "f1", ("finish_rate_l5", "finish_rate"), 0.45) or 0.45
        ),
        0.10,
        0.80,
    )
    f2_finish = _clip_prob(
        float(
            _fighter_rate_first(row, "f2", ("finish_rate_l5", "finish_rate"), 0.45) or 0.45
        ),
        0.10,
        0.80,
    )

    ko_diff = float(row.get("ko_rate_diff", 0) or 0)
    if "ko_win_rate_l5_diff" in row.index and pd.notna(row.get("ko_win_rate_l5_diff")):
        try:
            ko_diff = float(row["ko_win_rate_l5_diff"])
        except (TypeError, ValueError):
            pass
    sub_diff = float(row.get("sub_avg_diff", 0) or 0)
    if "sub_win_rate_l5_diff" in row.index and pd.notna(row.get("sub_win_rate_l5_diff")):
        try:
            sub_diff = float(row["sub_win_rate_l5_diff"])
        except (TypeError, ValueError):
            pass
    striker_diff = float(row.get("striker_score_diff", 0) or 0)

    p_ko = _clip_prob(0.5 * (f1_ko + f2_ko) + 0.15 * ko_diff + 0.08 * striker_diff, 0.08, 0.62)
    p_sub = _clip_prob(0.5 * (f1_sub + f2_sub) + 0.12 * sub_diff, 0.05, 0.45)
    p_dec = _clip_prob(1.0 - p_ko - p_sub, 0.12, 0.75)

    # Decision-profile / pathway: tilt toward distance when both fighters finish via decisions
    f1_share = _fighter_rate_first(
        row, "f1", ("decision_finish_share_l5", "decision_finish_share_career"), None
    )
    f2_share = _fighter_rate_first(
        row, "f2", ("decision_finish_share_l5", "decision_finish_share_career"), None
    )
    f1_dec_w = _fighter_rate_first(row, "f1", ("dec_win_rate_l5", "dec_win_rate_career"), None)
    f2_dec_w = _fighter_rate_first(row, "f2", ("dec_win_rate_l5", "dec_win_rate_career"), None)
    f1_dec_l = _fighter_rate_first(row, "f1", ("dec_loss_rate_l5", "dec_loss_rate_career"), None)
    f2_dec_l = _fighter_rate_first(row, "f2", ("dec_loss_rate_l5", "dec_loss_rate_career"), None)
    f1_dist = _fighter_rate_first(row, "f1", ("distance_rate_l5", "distance_rate_career"), None)
    f2_dist = _fighter_rate_first(row, "f2", ("distance_rate_l5", "distance_rate_career"), None)

    dec_tilt = 0.0
    share_vals = [v for v in (f1_share, f2_share) if v is not None]
    if share_vals:
        # share ~ dec_wins/(dec_wins+finish_wins); high → more distance
        dec_tilt += 0.18 * (float(np.mean(share_vals)) - 0.45)
    dec_w_vals = [v for v in (f1_dec_w, f2_dec_w) if v is not None]
    if dec_w_vals:
        dec_tilt += 0.10 * (float(np.mean(dec_w_vals)) - 0.40)
    dec_l_vals = [v for v in (f1_dec_l, f2_dec_l) if v is not None]
    if dec_l_vals:
        # Fighters who often lose on cards also push fights deep
        dec_tilt += 0.06 * (float(np.mean(dec_l_vals)) - 0.25)
    dist_vals = [v for v in (f1_dist, f2_dist) if v is not None]
    if dist_vals:
        dec_tilt += 0.12 * (float(np.mean(dist_vals)) - 0.45)
    if abs(dec_tilt) > 1e-9:
        p_dec = _clip_prob(p_dec + dec_tilt, 0.12, 0.78)
        finish_mass = max(1e-6, p_ko + p_sub)
        scale = max(0.05, 1.0 - p_dec) / finish_mass
        p_ko *= scale
        p_sub *= scale

    total = p_ko + p_sub + p_dec
    p_ko, p_sub, p_dec = p_ko / total, p_sub / total, p_dec / total

    f1_r1 = _fighter_rate_first(
        row, "f1", ("r1_finish_rate_l5", "r1_finish_rate_career"), None
    )
    f2_r1 = _fighter_rate_first(
        row, "f2", ("r1_finish_rate_l5", "r1_finish_rate_career"), None
    )
    avg_finish = 0.5 * (f1_finish + f2_finish)
    if f1_r1 is not None or f2_r1 is not None:
        r1_vals = [
            float(v if v is not None else avg_finish * 0.55) for v in (f1_r1, f2_r1)
        ]
        p_r1 = _clip_prob(
            0.5 * sum(r1_vals) * (1.0 + 0.20 * abs(ko_diff)) * (1.0 - 0.35 * max(0.0, dec_tilt)),
            0.08,
            0.55,
        )
    else:
        p_r1 = _clip_prob(
            avg_finish * 0.58 * (1.0 + 0.25 * abs(ko_diff)) * (1.0 - 0.35 * max(0.0, dec_tilt)),
            0.08,
            0.55,
        )
    p_over_15 = _clip_prob(1.0 - p_r1, 0.25, 0.92)
    p_finish = _clip_prob(p_ko + p_sub, 0.15, 0.88)

    try:
        scheduled = int(float(row.get("scheduled_rounds", 3) or 3))
    except (TypeError, ValueError):
        scheduled = 3
    scheduled = 5 if scheduled >= 5 else 3
    # Deeper overs interpolate toward decision (go-the-distance ≈ Over (N-0.5))
    max_line = float(scheduled) - 0.5
    p_over_25 = _over_line_from_anchors(p_over_15, p_dec, point=2.5, max_line=max_line)
    p_over_35 = _over_line_from_anchors(p_over_15, p_dec, point=3.5, max_line=max_line)
    p_over_45 = _over_line_from_anchors(p_over_15, p_dec, point=4.5, max_line=max_line)

    p1 = float(row.get("prob_f1_win", row.get("predicted_prob", 0.5)) or 0.5)
    if pd.notna(row.get("prob_f2_win")):
        p2 = float(row["prob_f2_win"])
    else:
        p2 = 1.0 - p1
    pick_side = "f1" if p1 >= p2 else "f2"
    pick_prob = p1 if pick_side == "f1" else p2
    pick_ko = f1_ko if pick_side == "f1" else f2_ko
    pick_sub = f1_sub if pick_side == "f1" else f2_sub
    pick_dec = max(0.05, 1.0 - pick_ko - pick_sub)
    pick_total = pick_ko + pick_sub + pick_dec
    pick_ko /= pick_total
    pick_sub /= pick_total
    pick_dec /= pick_total

    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
    pick_name = f1 if pick_side == "f1" else f2

    def _cond(ko_r: float, sub_r: float) -> tuple[float, float]:
        dec_r = max(0.05, 1.0 - ko_r - sub_r)
        tot = ko_r + sub_r + dec_r
        return ko_r / tot, sub_r / tot

    f1_ko_c, f1_sub_c = _cond(f1_ko, f1_sub)
    f2_ko_c, f2_sub_c = _cond(f2_ko, f2_sub)

    return {
        "ko": p_ko,
        "sub": p_sub,
        "dec": p_dec,
        "finish": p_finish,
        "round_1_finish": p_r1,
        "over_1_5_rounds": p_over_15,
        "under_1_5_rounds": _clip_prob(1.0 - p_over_15, 0.08, 0.75),
        "over_2_5_rounds": p_over_25,
        "under_2_5_rounds": _clip_prob(1.0 - p_over_25, 0.05, 0.90),
        "over_3_5_rounds": p_over_35,
        "under_3_5_rounds": _clip_prob(1.0 - p_over_35, 0.05, 0.90),
        "over_4_5_rounds": p_over_45,
        "under_4_5_rounds": _clip_prob(1.0 - p_over_45, 0.05, 0.90),
        "fighter_ko": _clip_prob(pick_prob * pick_ko, 0.03, 0.55),
        "fighter_sub": _clip_prob(pick_prob * pick_sub, 0.02, 0.40),
        "fighter_decision": _clip_prob(pick_prob * pick_dec, 0.05, 0.70),
        "pick_side": pick_side,
        "pick_name": pick_name,
        "pick_prob": pick_prob,
        "f1_name": f1,
        "f2_name": f2,
        "f1_ko_cond": f1_ko_c,
        "f1_sub_cond": f1_sub_c,
        "f2_ko_cond": f2_ko_c,
        "f2_sub_cond": f2_sub_c,
        "p1": p1,
        "p2": p2,
    }


def side_method_joint_probs(row: pd.Series, probs: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """P(fighter wins by ko/sub/dec) for both sides (research / live method lines)."""
    p = probs or method_probs_from_row(row)
    f1 = str(p.get("f1_name") or row.get("fighter_1", row.get("fighter1", ""))).strip()
    f2 = str(p.get("f2_name") or row.get("fighter_2", row.get("fighter2", ""))).strip()
    p1 = float(p.get("p1", row.get("prob_f1_win", row.get("predicted_prob", 0.5)) or 0.5))
    p2 = float(p.get("p2", 1.0 - p1))
    out: dict[str, dict[str, float]] = {}
    for name, win_p, ko_c, sub_c in (
        (f1, p1, float(p.get("f1_ko_cond", 0.33)), float(p.get("f1_sub_cond", 0.2))),
        (f2, p2, float(p.get("f2_ko_cond", 0.33)), float(p.get("f2_sub_cond", 0.2))),
    ):
        if not name:
            continue
        dec_c = max(0.05, 1.0 - ko_c - sub_c)
        total = ko_c + sub_c + dec_c
        ko_c, sub_c, dec_c = ko_c / total, sub_c / total, dec_c / total
        out[name] = {
            "ko": _clip_prob(win_p * ko_c, 0.01, 0.70),
            "sub": _clip_prob(win_p * sub_c, 0.005, 0.55),
            "dec": _clip_prob(win_p * dec_c, 0.02, 0.75),
            "win": win_p,
        }
    return out


def selection_fighter_name(selection: str) -> str:
    """Parse fighter from MyBookie-style selection ('Islam Makhachev Yes')."""
    text = " ".join(str(selection or "").split()).strip()
    if not text:
        return ""
    if text.lower().endswith(" yes"):
        text = text[: -4].strip()
    for suffix in (" by ko/tko", " by ko", " by submission", " by decision"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    return text


def fighter_method_model_prob(
    prop_key: str,
    row: pd.Series,
    fighter_name: str,
    *,
    probs: dict[str, float] | None = None,
) -> float:
    """Joint model prob for a named fighter method market."""
    side = _METHOD_KEY_TO_SIDE.get(str(prop_key))
    if not side:
        return prop_model_prob(prop_key, row, probs)
    joints = side_method_joint_probs(row, probs)
    from src.predictor import _names_match

    for name, vals in joints.items():
        if _names_match(fighter_name, name) or fighter_name.lower() in name.lower():
            return float(vals[side])
    # Fall back to model-pick joint if name unresolved
    p = probs or method_probs_from_row(row)
    return float(p.get(prop_key, 0.0) or 0.0)


def prop_model_prob(prop_key: str, row: pd.Series, probs: dict[str, float] | None = None) -> float:
    """Model probability for a supported prop market."""
    p = probs or method_probs_from_row(row)
    mapping = {
        "goes_to_decision": p["dec"],
        "finish": p["finish"],
        "ko_tko": p["ko"],
        "submission": p["sub"],
        "round_1_finish": p["round_1_finish"],
        "over_1_5_rounds": p["over_1_5_rounds"],
        "under_1_5_rounds": p.get("under_1_5_rounds", 1.0 - float(p["over_1_5_rounds"])),
        "over_2_5_rounds": p["over_2_5_rounds"],
        "under_2_5_rounds": p["under_2_5_rounds"],
        "over_3_5_rounds": p["over_3_5_rounds"],
        "under_3_5_rounds": p["under_3_5_rounds"],
        "over_4_5_rounds": p["over_4_5_rounds"],
        "under_4_5_rounds": p["under_4_5_rounds"],
        "fighter_ko": p["fighter_ko"],
        "fighter_sub": p["fighter_sub"],
        "fighter_decision": p.get("fighter_decision", p["pick_prob"] * max(0.05, 1.0 - float(p.get("f1_ko_cond", 0.3)) - float(p.get("f1_sub_cond", 0.2)))),
    }
    if prop_key in mapping:
        return float(mapping[prop_key])
    m = _OVER_UNDER_POINT_RE.match(str(prop_key))
    if m:
        side, whole, frac = m.group(1), m.group(2), m.group(3)
        point = float(f"{whole}.{frac}")
        try:
            scheduled = int(float(row.get("scheduled_rounds", 3) or 3))
        except (TypeError, ValueError):
            scheduled = 3
        scheduled = 5 if scheduled >= 5 else 3
        over_p = _over_line_from_anchors(
            float(p["over_1_5_rounds"]),
            float(p["dec"]),
            point=point,
            max_line=float(scheduled) - 0.5,
        )
        return float(over_p if side == "over" else 1.0 - over_p)
    return 0.0


def synthetic_market_odds(model_prob: float, *, vig: float | None = None) -> float:
    """Fair decimal odds with book vig when live prop lines are unavailable."""
    v = vig if vig is not None else config.PROP_SYNTHETIC_VIG
    implied = max(model_prob, 0.02) * (1.0 + v)
    return float(max(1.05, 1.0 / implied))


def is_live_prop_odds_source(source: str | None) -> bool:
    """True for real book / Odds API lines (not synthetic)."""
    return str(source or "").strip().lower() in {"live", "the_odds_api"}


def fetch_live_prop_odds(
    book: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load live prop odds for a book (Odds API / DraftKings / BetNow / MyBookie)."""
    if not config.ENABLE_PROPS:
        from src.odds_providers.prop_odds_common import empty_prop_odds_df

        return empty_prop_odds_df()
    if book == "MyBookie" and not config.MYBOOKIE_ENABLED:
        from src.odds_providers.prop_odds_common import empty_prop_odds_df

        return empty_prop_odds_df()
    if book == "BetNow.eu" and not getattr(config, "BETNOW_ENABLED", False):
        from src.odds_providers.prop_odds_common import empty_prop_odds_df

        return empty_prop_odds_df()
    if book == "DraftKings" and not getattr(config, "DRAFTKINGS_ENABLED", False):
        # Do not hit The Odds API for DK props when DK is disabled (huge credit burn).
        from src.odds_providers.prop_odds_common import empty_prop_odds_df

        return empty_prop_odds_df()
    try:
        if book in ("Odds API", "the_odds_api", "OddsAPI"):
            from src.odds_providers.the_odds_api import fetch_the_odds_api_prop_odds

            return fetch_the_odds_api_prop_odds(force_refresh=force_refresh)
        if book == "DraftKings":
            from src.odds_providers.draftkings_props import fetch_draftkings_prop_odds

            return fetch_draftkings_prop_odds(force_refresh=force_refresh)
        if book == "BetNow.eu":
            from src.odds_providers.betnow_scraper import fetch_betnow_prop_odds

            return fetch_betnow_prop_odds(force_refresh=force_refresh)
        if book == "MyBookie":
            from src.odds_providers.mybookie_scraper import fetch_mybookie_prop_odds

            return fetch_mybookie_prop_odds(force_refresh=force_refresh)
    except Exception as exc:
        logger = __import__("logging").getLogger(__name__)
        logger.warning("%s live prop odds unavailable: %s", book, exc)
    from src.odds_providers.prop_odds_common import empty_prop_odds_df

    return empty_prop_odds_df()


def resolve_prop_quote(
    row: pd.Series,
    prop_key: str,
    *,
    book: str,
    prop_odds: pd.DataFrame | None = None,
    probs: dict[str, float] | None = None,
    fighter_name: str | None = None,
) -> dict[str, Any]:
    """
    Resolve market quote for a prop: prefer live book line, else synthetic.

    Returns decimal_odds, implied_prob, edge, odds_source, american_odds.
    """
    from src.odds_providers.prop_odds_common import lookup_prop_odds_row

    p = probs or method_probs_from_row(row)
    named = str(fighter_name or "").strip()
    if named and prop_key in _FIGHTER_METHOD_KEYS:
        model_p = fighter_method_model_prob(prop_key, row, named, probs=p)
    else:
        model_p = prop_model_prob(prop_key, row, p)
    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()

    live_row = None
    if prop_odds is not None and not prop_odds.empty:
        sel_fighter = named or (str(p.get("pick_name", "")) if prop_key in _FIGHTER_METHOD_KEYS else "")
        live_row = lookup_prop_odds_row(
            f1,
            f2,
            prop_key,
            prop_odds,
            fighter_name=sel_fighter or None,
        )

    if live_row is not None and float(live_row.get("decimal_odds", 0) or 0) > 1:
        decimal = float(live_row["decimal_odds"])
        implied = float(live_row.get("implied_prob") or (1.0 / decimal))
        edge = model_p - implied
        src = str(live_row.get("odds_source") or "live")
        if is_live_prop_odds_source(src):
            # Keep labeled source (the_odds_api) while counting as live for HA
            pass
        else:
            src = "live"
        return {
            "decimal_odds": decimal,
            "implied_prob": implied,
            "edge": edge,
            "odds_source": src,
            "american_odds": live_row.get("american_odds"),
            "selection": live_row.get("selection", ""),
            "book": book,
        }

    decimal = synthetic_market_odds(model_p)
    implied = model_p * (1.0 + config.PROP_SYNTHETIC_VIG)
    edge = model_p - implied
    return {
        "decimal_odds": decimal,
        "implied_prob": implied,
        "edge": edge,
        "odds_source": "synthetic",
        "american_odds": None,
        "selection": "",
        "book": book,
    }


def _method_short_label(prop_key: str, fighter: str) -> str:
    if prop_key == "fighter_ko":
        return f"{fighter} by KO/TKO"
    if prop_key == "fighter_sub":
        return f"{fighter} by Submission"
    if prop_key == "fighter_decision":
        return f"{fighter} by Decision"
    return PROP_MARKET_LABELS.get(prop_key, prop_key.replace("_", " ").title())


def prop_short_label(
    prop_key: str,
    row: pd.Series,
    probs: dict[str, float] | None = None,
    *,
    fighter_name: str | None = None,
) -> str:
    """Compact prop title for dashboard tables (e.g. 'Pereira by KO/TKO')."""
    if prop_key in _FIGHTER_METHOD_KEYS:
        named = str(fighter_name or "").strip()
        if not named:
            p = probs or method_probs_from_row(row)
            named = str(p.get("pick_name", "")).strip()
        if named:
            return _method_short_label(prop_key, named)
    return PROP_MARKET_LABELS.get(prop_key, prop_key.replace("_", " ").title())


def prop_display_label(
    prop_key: str,
    row: pd.Series,
    probs: dict[str, float] | None = None,
    *,
    fighter_name: str | None = None,
) -> str:
    """Full prop label for slips and parlay legs."""
    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
    fight = f"{f1} vs {f2}"
    return f"{prop_short_label(prop_key, row, probs, fighter_name=fighter_name)} ({fight})"


def settle_prop(prop_key: str, row: pd.Series) -> bool | None:
    """Whether a prop bet wins given actual fight outcome. None if outcome unknown."""
    method = row.get("method", row.get("METHOD"))
    rnd = row.get("round", row.get("ROUND"))
    if pd.isna(method) and pd.isna(rnd):
        return None

    ko, sub, dec = method_flags(method)
    try:
        round_num = int(float(rnd)) if pd.notna(rnd) else 0
    except (TypeError, ValueError):
        round_num = 0

    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
    winner = str(row.get("winner", row.get("predicted_winner", ""))).strip()
    actual_f1_win = row.get(config.TARGET_COLUMN)
    if pd.isna(actual_f1_win) and winner:
        if winner == f1:
            actual_f1_win = 1
        elif winner == f2:
            actual_f1_win = 0
    # Model pick only (do not use actual winner — that leaks outcome into fighter props)
    pick_side = "f1" if float(row.get("prob_f1_win", row.get("predicted_prob", 0.5)) or 0.5) >= 0.5 else "f2"
    if pd.notna(row.get("prob_f2_win")):
        pick_side = "f1" if float(row["prob_f1_win"] or 0.5) >= float(row["prob_f2_win"]) else "f2"

    if prop_key == "goes_to_decision":
        return bool(dec)
    if prop_key == "finish":
        return bool(ko or sub)
    if prop_key == "ko_tko":
        return bool(ko)
    if prop_key == "submission":
        return bool(sub)
    if prop_key == "round_1_finish":
        return round_num == 1 and bool(ko or sub)
    if prop_key == "over_1_5_rounds":
        # Complements under_1_5: fight continues past round 1
        return round_num > 1
    if prop_key == "under_1_5_rounds":
        # Fight ends in round 1 (does not go past 1.5)
        return round_num == 1
    m = _OVER_UNDER_POINT_RE.match(str(prop_key))
    if m:
        side, whole, frac = m.group(1), m.group(2), m.group(3)
        point = float(f"{whole}.{frac}")
        # Over X.5 wins when the fight reaches round (floor(X)+1), i.e. round_num > int(X)
        went_over = round_num > int(point)
        return bool(went_over if side == "over" else not went_over)
    if prop_key == "fighter_ko":
        pick_won = (pick_side == "f1" and int(actual_f1_win or 0) == 1) or (
            pick_side == "f2" and int(actual_f1_win or 0) == 0
        )
        return bool(pick_won and ko)
    if prop_key == "fighter_sub":
        pick_won = (pick_side == "f1" and int(actual_f1_win or 0) == 1) or (
            pick_side == "f2" and int(actual_f1_win or 0) == 0
        )
        return bool(pick_won and sub)
    if prop_key == "fighter_decision":
        pick_won = (pick_side == "f1" and int(actual_f1_win or 0) == 1) or (
            pick_side == "f2" and int(actual_f1_win or 0) == 0
        )
        return bool(pick_won and dec)
    return None


def extract_prop_candidate(
    row: pd.Series,
    prop_key: str,
    *,
    config: StrategyConfig | None = None,
    min_edge: float | None = None,
    probs: dict[str, float] | None = None,
    book: str = "",
    prop_odds: pd.DataFrame | None = None,
    for_display: bool = False,
) -> BetCandidate | None:
    """Build a prop BetCandidate when model edge clears threshold (or synthetic display)."""
    from src.high_accuracy_strategy import (
        log_strategy_block,
        profile_rules,
        prop_allowed,
    )

    if prop_key not in PROP_MARKET_LABELS:
        return None

    f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
    f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
    fight_lbl = f"{f1} vs {f2}"

    try:
        from src.fighter_flags import should_skip_fight

        skip_flag, flag_detail = should_skip_fight(f1, f2)
        if skip_flag:
            # Block betting and ranked prop lists (no stakes on flagged fights)
            if not for_display:
                log_strategy_block(
                    "fighter_integrity_flag",
                    context="prop",
                    fight=fight_lbl,
                    prop_key=prop_key,
                    detail=flag_detail,
                )
            else:
                log_strategy_block(
                    "fighter_integrity_flag",
                    context="prop_display",
                    fight=fight_lbl,
                    prop_key=prop_key,
                    detail=flag_detail,
                )
            return None
    except Exception:
        pass

    # Hard-code: Over 1.5 Rounds is the only bettable prop
    if not prop_allowed(prop_key):
        if not for_display:
            log_strategy_block(
                "prop_market_disabled",
                context="prop",
                fight=fight_lbl,
                prop_key=prop_key,
            )
        return None

    import config as _cfg

    rules = profile_rules()
    edge_floor = float(
        min_edge if min_edge is not None else rules.get("prop_min_live_edge", _cfg.PROP_MIN_EDGE)
    )
    model_floor = float(rules.get("prop_min_model_prob", _cfg.PROP_MIN_MODEL_PROB))
    strat = config or StrategyConfig(min_edge=edge_floor)
    p = probs or method_probs_from_row(row)
    quote = resolve_prop_quote(
        row,
        prop_key,
        book=book or "Consensus",
        prop_odds=prop_odds,
        probs=p,
    )
    model_p = prop_model_prob(prop_key, row, p)
    edge = float(quote["edge"])
    odds = float(quote["decimal_odds"])
    odds_source = str(quote.get("odds_source", "synthetic"))

    from src.strategy import edge_is_actionable

    # Betting path: require live / Odds API odds + model floor + live edge + not a bogus scrape
    qualifies_live = (
        is_live_prop_odds_source(odds_source)
        and model_p >= model_floor
        and edge >= edge_floor
        and edge_is_actionable(edge, decimal_odds=odds, model_prob=model_p)
    )
    # Display-only synthetic (research browse) — Over 1.5 only via ALLOWED_PROP_KEYS
    qualifies_synth = (
        for_display
        and odds_source == "synthetic"
        and model_p >= model_floor
    )
    if not qualifies_live and not qualifies_synth:
        if not is_live_prop_odds_source(odds_source) and not for_display:
            log_strategy_block(
                "prop_requires_live_odds",
                context="prop",
                fight=fight_lbl,
                prop_key=prop_key,
                detail=f"source={odds_source}",
            )
        elif model_p < model_floor:
            log_strategy_block(
                "prop_low_model_prob",
                context="prop",
                fight=fight_lbl,
                prop_key=prop_key,
                detail=f"prob={model_p:.3f}<{model_floor:.3f}",
            )
        elif is_live_prop_odds_source(odds_source) and not edge_is_actionable(
            edge, decimal_odds=odds, model_prob=model_p
        ):
            log_strategy_block(
                "prop_suspect_edge",
                context="prop",
                fight=fight_lbl,
                prop_key=prop_key,
                detail=f"edge={edge:.3f}",
            )
        elif is_live_prop_odds_source(odds_source) and edge < edge_floor:
            log_strategy_block(
                "prop_low_edge",
                context="prop",
                fight=fight_lbl,
                prop_key=prop_key,
                detail=f"edge={edge:.3f}<{edge_floor:.3f}",
            )
        return None

    label = prop_display_label(prop_key, row, p)

    cand = BetCandidate(
        fight_id=str(row.get("fight_id", "")),
        event_key=str(row.get("event_name", row.get("event", ""))),
        bet_side=p.get("pick_side", "f1"),
        prob=model_p,
        decimal_odds=odds,
        edge=edge,
        kelly_full=0.0,
        expected_value=bet_expected_value(model_p, odds),
        fighter1_name=f1,
        fighter2_name=f2,
        pick_name=label,
        winner_name=label,
        market_type="prop",
        prop_key=prop_key,
        display_label=label,
    )
    cand.odds_source = odds_source
    return cand


def extract_prop_candidates_for_row(
    row: pd.Series,
    *,
    strategy: StrategyConfig | None = None,
    markets: list[str] | None = None,
    book: str = "",
    prop_odds: pd.DataFrame | None = None,
    for_display: bool = False,
) -> list[BetCandidate]:
    """All qualifying prop singles for one fight row (Over 1.5 only)."""
    if not config.ENABLE_PROPS:
        return []
    from src.high_accuracy_strategy import ALLOWED_PROP_KEYS

    keys = markets or list(config.PROP_MARKETS)
    keys = [k for k in keys if k in ALLOWED_PROP_KEYS]
    if not keys:
        keys = list(ALLOWED_PROP_KEYS)
    probs = method_probs_from_row(row)
    out: list[BetCandidate] = []
    for key in keys:
        cand = extract_prop_candidate(
            row,
            key,
            config=strategy,
            probs=probs,
            book=book,
            prop_odds=prop_odds,
            for_display=for_display,
        )
        if cand is not None:
            out.append(cand)
    return out


def enrich_predictions_with_props(
    df: pd.DataFrame,
    *,
    book: str = "",
    prop_odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach prop probability and live/synthetic quote columns for dashboard tables."""
    if not config.ENABLE_PROPS or df.empty:
        return df
    from src.odds_providers.prop_odds_common import attach_prop_odds_to_predictions

    out = df
    # Props-only pathway/decision rates when missing (does not enable ML flags).
    if "f1_r1_finish_rate_l5" not in out.columns or out["f1_r1_finish_rate_l5"].isna().all():
        try:
            from src.pathway_features import attach_pathway_rates_for_props

            out = attach_pathway_rates_for_props(out)
        except Exception:
            pass
    if "f1_decision_finish_share_l5" not in out.columns:
        try:
            from src.decision_profile import attach_decision_profile_to_wide

            out = attach_decision_profile_to_wide(out)
        except Exception:
            pass

    odds_df = prop_odds if prop_odds is not None else pd.DataFrame()
    out = attach_prop_odds_to_predictions(out, odds_df)
    for key in config.PROP_MARKETS:
        col = f"prop_prob_{key}"
        out[col] = out.apply(lambda r: prop_model_prob(key, r), axis=1)
        quote_col = f"prop_odds_source_{key}"
        if quote_col not in out.columns:
            out[quote_col] = ""
        for idx, row in out.iterrows():
            if str(out.at[idx, quote_col]).strip():
                continue
            quote = resolve_prop_quote(row, key, book=book, prop_odds=prop_odds)
            out.at[idx, f"prop_odds_{key}"] = quote["decimal_odds"]
            out.at[idx, f"prop_edge_{key}"] = quote["edge"]
            out.at[idx, quote_col] = quote["odds_source"]
    return out


def event_from_record(obj: Any) -> str:
    """Event title from a prediction row, prop dict, or ticket."""
    if obj is None:
        return ""
    if isinstance(obj, pd.Series):
        return str(obj.get("event_name") or obj.get("event") or obj.get("event_key") or "").strip()
    if isinstance(obj, dict):
        return str(
            obj.get("event")
            or obj.get("event_name")
            or obj.get("event_key")
            or ""
        ).strip()
    return str(obj).strip()


def _prop_fighter_name(
    prop_key: str,
    row: pd.Series,
    probs: dict[str, float] | None = None,
    *,
    fighter_name: str | None = None,
) -> str:
    """Fighter column for prop table — pick name for fighter-specific markets."""
    if prop_key in _FIGHTER_METHOD_KEYS:
        named = str(fighter_name or "").strip()
        if named:
            return named
        p = probs or method_probs_from_row(row)
        return str(p.get("pick_name", "-"))
    return "-"


def _prop_row_dict(
    cand: BetCandidate,
    *,
    rank: int,
    book: str,
    row: pd.Series,
    probs: dict[str, float],
    strict_qualified: bool,
) -> dict[str, Any]:
    odds_source = getattr(cand, "odds_source", "synthetic")
    edge_pct = cand.edge * 100.0 if is_live_prop_odds_source(odds_source) else None
    prop_key = cand.prop_key
    fighter_override = str(getattr(cand, "prop_fighter", "") or "").strip()
    short = prop_short_label(prop_key, row, probs, fighter_name=fighter_override or None)
    event = event_from_record(row) or str(getattr(cand, "event_key", "") or "").strip()
    return {
        "rank": rank,
        "book": book,
        "fight_id": cand.fight_id,
        "fight": f"{cand.fighter1_name} vs {cand.fighter2_name}",
        "event": event,
        "event_name": event,
        "prop_key": prop_key,
        "prop_type": short,
        "prop_short": short,
        "fighter": _prop_fighter_name(
            prop_key, row, probs, fighter_name=fighter_override or None
        ),
        "label": cand.display_label or cand.pick_name,
        "prob": cand.prob,
        "odds": cand.decimal_odds,
        "edge": cand.edge,
        "edge_pct": edge_pct,
        "ev": cand.expected_value,
        "market_type": "prop",
        "odds_source": odds_source,
        "source_label": (
            "live"
            if is_live_prop_odds_source(odds_source)
            else "synthetic"
        ),
        "strict_qualified": strict_qualified,
        "parlay_allowed": config.BOOK_PROP_RULES.get(book, {}).get("allow_prop_parlays", False),
        "suggested_stake": 0.0 if not strict_qualified else None,
    }


def _collect_live_method_display(
    rows: pd.DataFrame,
    *,
    book: str,
    prop_odds: pd.DataFrame | None,
    seen: set[str],
) -> list[tuple[BetCandidate, pd.Series, dict[str, float], bool]]:
    """Research-only: surface live fighter KO/sub/decision lines with model edge."""
    from src.predictor import _names_match

    out: list[tuple[BetCandidate, pd.Series, dict[str, float], bool]] = []
    if prop_odds is None or prop_odds.empty or rows.empty:
        return out

    method_odds = prop_odds[prop_odds["prop_key"].astype(str).isin(_FIGHTER_METHOD_KEYS)]
    if method_odds.empty:
        return out

    for _, live in method_odds.iterrows():
        prop_key = str(live.get("prop_key", "")).strip()
        if prop_key not in _FIGHTER_METHOD_KEYS:
            continue
        lf1 = str(live.get("fighter_1", "")).strip()
        lf2 = str(live.get("fighter_2", "")).strip()
        fighter = selection_fighter_name(str(live.get("selection", "")))
        if not fighter:
            continue
        decimal = float(live.get("decimal_odds", 0) or 0)
        if decimal <= 1:
            continue
        implied = float(live.get("implied_prob") or (1.0 / decimal))
        src = str(live.get("odds_source") or "live")
        if not is_live_prop_odds_source(src):
            src = "live"

        pred_row: pd.Series | None = None
        for _, row in rows.iterrows():
            f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
            f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
            aligned = _names_match(f1, lf1) and _names_match(f2, lf2)
            swapped = _names_match(f1, lf2) and _names_match(f2, lf1)
            if aligned or swapped:
                pred_row = row
                break
        if pred_row is None:
            # Orphan live line (fight not on scored card) — still show for research
            pred_row = pd.Series(
                {
                    "fight_id": f"live|{lf1}|{lf2}",
                    "event_name": "",
                    "event": "",
                    "fighter_1": lf1,
                    "fighter_2": lf2,
                    "prob_f1_win": 0.5,
                    "prob_f2_win": 0.5,
                }
            )

        probs = method_probs_from_row(pred_row)
        model_p = fighter_method_model_prob(prop_key, pred_row, fighter, probs=probs)
        # No model signal for orphan 50/50 stubs beyond method priors already in probs
        edge = model_p - implied
        f1 = str(pred_row.get("fighter_1", lf1)).strip()
        f2 = str(pred_row.get("fighter_2", lf2)).strip()
        short = _method_short_label(prop_key, fighter)
        label = f"{short} ({f1} vs {f2})"
        dedupe = f"{pred_row.get('fight_id', '')}|{prop_key}|{fighter.lower()}"
        if dedupe in seen:
            continue
        seen.add(dedupe)

        cand = BetCandidate(
            fight_id=str(pred_row.get("fight_id", "")),
            event_key=str(pred_row.get("event_name", pred_row.get("event", ""))),
            bet_side=probs.get("pick_side", "f1"),
            prob=model_p,
            decimal_odds=decimal,
            edge=edge,
            kelly_full=0.0,
            expected_value=bet_expected_value(model_p, decimal),
            fighter1_name=f1,
            fighter2_name=f2,
            pick_name=short,
            winner_name=fighter,
            market_type="prop",
            prop_key=prop_key,
            display_label=label,
        )
        cand.odds_source = src
        cand.prop_fighter = fighter  # type: ignore[attr-defined]
        # strict_qualified=False → research only, never HA-sized / Blue
        out.append((cand, pred_row, probs, False))
    return out


def _collect_prop_candidates(
    rows: pd.DataFrame,
    *,
    book: str,
    strategy: StrategyConfig | None,
    prop_odds: pd.DataFrame | None,
    include_relaxed: bool,
) -> tuple[list[tuple[BetCandidate, pd.Series, dict[str, float], bool]], set[str]]:
    """Return (candidate, row, probs, strict) tuples and seen keys for dedupe."""
    collected: list[tuple[BetCandidate, pd.Series, dict[str, float], bool]] = []
    seen: set[str] = set()

    for _, row in rows.iterrows():
        probs = method_probs_from_row(row)
        from src.high_accuracy_strategy import ALLOWED_PROP_KEYS

        markets = [k for k in config.PROP_MARKETS if k in ALLOWED_PROP_KEYS] or list(ALLOWED_PROP_KEYS)
        for key in markets:
            cand = extract_prop_candidate(
                row,
                key,
                config=strategy,
                probs=probs,
                book=book,
                prop_odds=prop_odds,
                for_display=True,
            )
            if cand is not None:
                from src.strategy import edge_is_actionable

                src = str(getattr(cand, "odds_source", "") or "")
                strict = is_live_prop_odds_source(src) and edge_is_actionable(
                    float(cand.edge),
                    decimal_odds=float(cand.decimal_odds or 0) or None,
                    model_prob=float(cand.prob),
                )
                dedupe = f"{cand.fight_id}|{key}"
                if dedupe not in seen:
                    seen.add(dedupe)
                    collected.append((cand, row, probs, strict))
                continue
            if not include_relaxed:
                continue
            quote = resolve_prop_quote(row, key, book=book, prop_odds=prop_odds, probs=probs)
            model_p = prop_model_prob(key, row, probs)
            src = str(quote.get("odds_source", "synthetic"))
            # Live line that missed HA (suspect 26% scrape, low prob, etc.) — show $0
            if is_live_prop_odds_source(src):
                if model_p < config.PROP_SHOW_ALL_MIN_PROB:
                    continue
                dedupe = f"{row.get('fight_id', '')}|{key}"
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
                f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
                label = prop_display_label(key, row, probs)
                relaxed = BetCandidate(
                    fight_id=str(row.get("fight_id", "")),
                    event_key=str(row.get("event_name", row.get("event", ""))),
                    bet_side=probs.get("pick_side", "f1"),
                    prob=model_p,
                    decimal_odds=float(quote["decimal_odds"]),
                    edge=float(quote["edge"]),
                    kelly_full=0.0,
                    expected_value=bet_expected_value(model_p, float(quote["decimal_odds"])),
                    fighter1_name=f1,
                    fighter2_name=f2,
                    pick_name=label,
                    winner_name=label,
                    market_type="prop",
                    prop_key=key,
                    display_label=label,
                )
                relaxed.odds_source = src
                collected.append((relaxed, row, probs, False))
                continue
            if src != "synthetic":
                continue
            if model_p < config.PROP_SHOW_ALL_MIN_PROB:
                continue
            dedupe = f"{row.get('fight_id', '')}|{key}"
            if dedupe in seen:
                continue
            seen.add(dedupe)
            f1 = str(row.get("fighter_1", row.get("fighter1", ""))).strip()
            f2 = str(row.get("fighter_2", row.get("fighter2", ""))).strip()
            label = prop_display_label(key, row, probs)
            relaxed = BetCandidate(
                fight_id=str(row.get("fight_id", "")),
                event_key=str(row.get("event_name", row.get("event", ""))),
                bet_side=probs.get("pick_side", "f1"),
                prob=model_p,
                decimal_odds=float(quote["decimal_odds"]),
                edge=float(quote["edge"]),
                kelly_full=0.0,
                expected_value=bet_expected_value(model_p, float(quote["decimal_odds"])),
                fighter1_name=f1,
                fighter2_name=f2,
                pick_name=label,
                winner_name=label,
                market_type="prop",
                prop_key=key,
                display_label=label,
            )
            relaxed.odds_source = "synthetic"
            collected.append((relaxed, row, probs, False))

    # Live method markets (KO / sub / decision) — display + edge only, never HA-sized
    collected.extend(
        _collect_live_method_display(rows, book=book, prop_odds=prop_odds, seen=seen)
    )

    return collected, seen


def rank_prop_singles(
    rows: pd.DataFrame,
    *,
    book: str,
    strategy: StrategyConfig | None = None,
    max_results: int | None = None,
    prop_odds: pd.DataFrame | None = None,
    include_relaxed: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Top prop singles by edge for a book.

    Returns (ranked rows, meta) where meta has total_found and strict_count.
    """
    empty_meta = {"total_found": 0, "strict_count": 0, "relaxed_count": 0}
    if not config.ENABLE_PROPS or rows.empty:
        return [], empty_meta

    base_cap = max_results if max_results is not None else config.PROP_MAX_RESULTS
    collected, _ = _collect_prop_candidates(
        rows,
        book=book,
        strategy=strategy,
        prop_odds=prop_odds,
        include_relaxed=include_relaxed,
    )

    def _rank_key(item: tuple[BetCandidate, pd.Series, dict, bool]) -> tuple[int, float]:
        cand, _, _, strict = item
        source = getattr(cand, "odds_source", "synthetic")
        live = is_live_prop_odds_source(source)
        score = cand.edge if live else cand.prob
        return (2 if strict and live else 1 if strict else 0, score)

    collected.sort(key=_rank_key, reverse=True)
    strict_count = sum(1 for *_, strict in collected if strict)
    total_found = len(collected)
    # Raise display cap when live method research lines are present so the board isn't truncated
    has_method = any(c.prop_key in _FIGHTER_METHOD_KEYS for c, *_ in collected)
    cap = max(base_cap, min(120, total_found)) if has_method else base_cap

    ranked: list[dict[str, Any]] = []
    for i, (cand, row, probs, strict) in enumerate(collected[:cap], 1):
        ranked.append(
            _prop_row_dict(cand, rank=i, book=book, row=row, probs=probs, strict_qualified=strict)
        )

    live_count = sum(
        1
        for cand, *_ in collected
        if is_live_prop_odds_source(getattr(cand, "odds_source", ""))
    )
    synthetic_count = total_found - live_count
    meta = {
        "total_found": total_found,
        "strict_count": strict_count,
        "relaxed_count": max(0, total_found - strict_count),
        "live_count": live_count,
        "synthetic_count": synthetic_count,
        "shown": len(ranked),
        "cap": cap,
    }
    return ranked, meta


@dataclass
class PropParlayCandidate:
    legs: list[BetCandidate]
    combined_prob: float
    combined_odds: float
    expected_value: float
    min_leg_edge: float
    correlation_adjusted: bool = False


def _same_fight_groups(legs: list[BetCandidate]) -> dict[str, list[BetCandidate]]:
    groups: dict[str, list[BetCandidate]] = {}
    for leg in legs:
        groups.setdefault(leg.fight_id, []).append(leg)
    return groups


def correlated_combined_prob(legs: list[BetCandidate], row_lookup: dict[str, pd.Series]) -> float:
    """
    Adjust combined probability when legs share a fight or mix ML + prop.

    Policy (unchanged):
    - Different fights: independence (product of leg probs).
    - Same-fight ML + fighter_ko / fighter_sub: use joint method_probs
      (fighter_ko / fighter_sub already = P(pick wins AND method)).
    - Same-fight ML + goes_to_decision / finish: scale by pick_prob.
    - Same-fight ML + round/total markets: discount prop.prob * 0.85.
    - Same-fight multi-prop only: product * (1 - PROP_CORRELATION_DISCOUNT).
    - Missing row lookup: naive product * (1 - PROP_CORRELATION_DISCOUNT).
    """
    groups = _same_fight_groups(legs)
    joint_parts: list[float] = []

    for fight_id, group in groups.items():
        if len(group) == 1:
            joint_parts.append(group[0].prob)
            continue

        row = row_lookup.get(fight_id)
        if row is None:
            naive = float(np.prod([g.prob for g in group]))
            joint_parts.append(naive * (1.0 - config.PROP_CORRELATION_DISCOUNT))
            continue

        probs = method_probs_from_row(row)
        ml_legs = [g for g in group if g.market_type == "moneyline"]
        prop_legs = [g for g in group if g.market_type == "prop"]

        if ml_legs and prop_legs:
            ml = ml_legs[0]
            prop = prop_legs[0]
            p_win = ml.prob
            if prop.prop_key == "fighter_ko":
                joint = probs["fighter_ko"]
            elif prop.prop_key == "fighter_sub":
                joint = probs["fighter_sub"]
            elif prop.prop_key == "goes_to_decision":
                joint = p_win * probs["dec"] / max(probs["pick_prob"], 0.01)
            elif prop.prop_key == "finish":
                joint = p_win * (probs["ko"] + probs["sub"]) / max(probs["pick_prob"], 0.01)
            elif prop.prop_key in ("ko_tko", "submission", "round_1_finish", "over_1_5_rounds"):
                joint = prop.prob * 0.85
            else:
                joint = min(p_win, prop.prob)
            joint_parts.append(_clip_prob(joint))
        elif len(prop_legs) > 1:
            joint = float(np.prod([g.prob for g in prop_legs]))
            joint_parts.append(joint * (1.0 - config.PROP_CORRELATION_DISCOUNT))
        else:
            joint_parts.append(float(np.prod([g.prob for g in group])))

    return float(np.prod(joint_parts))


def build_prop_parlay_candidates(
    card_rows: pd.DataFrame,
    *,
    book: str = "DraftKings",
    strategy: StrategyConfig | None = None,
    include_moneyline: bool = True,
    prop_odds: pd.DataFrame | None = None,
    allow_despite_ha: bool = False,
) -> list[PropParlayCandidate]:
    """Same-card prop / mixed parlays per BOOK_PROP_RULES.

    Live HA keeps ``PROP_PARLAYS_ENABLED=False`` (no parlays on tickets).
    Research backtests may pass ``allow_despite_ha=True`` when the book allows
    prop/mixed parlays (DraftKings / MyBookie). BetNow stays singles-only.
    """
    from src.high_accuracy_strategy import PROP_PARLAYS_ENABLED, log_strategy_block

    if not PROP_PARLAYS_ENABLED and not allow_despite_ha:
        log_strategy_block("prop_parlays_disabled", context="prop_parlay", detail=f"book={book}")
        return []
    if not config.ENABLE_PROPS:
        return []

    rules = config.BOOK_PROP_RULES.get(book, {})
    if not rules.get("allow_prop_parlays"):
        return []

    from src.strategy import extract_bet_candidates

    strat = strategy or StrategyConfig(min_edge=config.PROP_MIN_EDGE)
    legs: list[BetCandidate] = []
    row_lookup: dict[str, pd.Series] = {}

    for _, row in card_rows.iterrows():
        fid = str(row.get("fight_id", ""))
        if fid:
            row_lookup[fid] = row
        legs.extend(
            extract_prop_candidates_for_row(
                row,
                strategy=strat,
                book=book,
                prop_odds=prop_odds,
                for_display=True,
            )
        )
        if include_moneyline and rules.get("allow_mixed_parlays"):
            ml = extract_bet_candidates(row, config=strat)
            if ml is not None and ml.edge >= strat.parlay_min_edge:
                legs.append(ml)

    if len(legs) < 2:
        return []

    max_legs = min(int(rules.get("max_prop_parlay_legs", 3)), len(legs))
    parlays: list[PropParlayCandidate] = []
    for n in range(2, max_legs + 1):
        for combo in combinations(legs, n):
            fight_ids = [c.fight_id for c in combo]
            if len(fight_ids) != len(set(fight_ids)):
                if not rules.get("allow_mixed_parlays"):
                    continue
            combined_prob = correlated_combined_prob(list(combo), row_lookup)
            if combined_prob < config.PROP_PARLAY_MIN_COMBINED_PROB:
                continue
            combined_odds = float(np.prod([c.decimal_odds for c in combo]))
            ev = combined_prob * (combined_odds - 1.0) - (1.0 - combined_prob)
            if ev < config.PROP_PARLAY_MIN_EV:
                continue
            parlays.append(
                PropParlayCandidate(
                    legs=list(combo),
                    combined_prob=combined_prob,
                    combined_odds=combined_odds,
                    expected_value=ev,
                    min_leg_edge=min(c.edge for c in combo),
                    correlation_adjusted=len(set(fight_ids)) < len(fight_ids),
                )
            )

    parlays.sort(key=lambda p: p.expected_value, reverse=True)
    return parlays


def prop_parlay_to_display_dict(parlay: PropParlayCandidate, *, rank: int, book: str) -> dict[str, Any]:
    """Serialize prop parlay for dashboard."""
    from src.parlay_builder import decimal_to_american, leg_betnow_label

    leg_rows = []
    for i, leg in enumerate(parlay.legs, 1):
        if leg.market_type == "prop":
            label = leg.display_label or leg.pick_name
            am_odds = decimal_to_american(leg.decimal_odds)
            leg_rows.append(f"{i}. {label} ({am_odds}) (UFC)")
        else:
            leg_rows.append(f"{i}. {leg_betnow_label(leg)}")

    return {
        "rank": rank,
        "book": book,
        "n_legs": len(parlay.legs),
        "combined_prob": parlay.combined_prob,
        "combined_odds": parlay.combined_odds,
        "expected_value": parlay.expected_value,
        "min_leg_edge": parlay.min_leg_edge,
        "correlation_adjusted": parlay.correlation_adjusted,
        "legs": [
            {
                "fight_id": c.fight_id,
                "market_type": c.market_type,
                "prop_key": c.prop_key,
                "label": c.display_label or c.pick_name,
                "prob": c.prob,
                "odds": c.decimal_odds,
                "edge": c.edge,
            }
            for c in parlay.legs
        ],
        "_leg_rows": leg_rows,
    }


def rank_prop_parlays_for_card(
    card_rows: pd.DataFrame,
    *,
    book: str = "DraftKings",
    strategy: StrategyConfig | None = None,
    max_results: int = 5,
    prop_odds: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    parlays = build_prop_parlay_candidates(
        card_rows,
        book=book,
        strategy=strategy,
        prop_odds=prop_odds,
    )
    return [
        prop_parlay_to_display_dict(p, rank=i + 1, book=book)
        for i, p in enumerate(parlays[:max_results])
    ]


def evaluate_prop_accuracy(predictions: pd.DataFrame) -> dict[str, float]:
    """Historical prop model accuracy on labeled fights."""
    if predictions.empty:
        return {"n_fights": 0.0}

    rows: list[dict[str, float]] = []
    for key in config.PROP_MARKETS:
        hits = 0
        total = 0
        for _, row in predictions.iterrows():
            actual = settle_prop(key, row)
            if actual is None:
                continue
            pred_p = prop_model_prob(key, row)
            pred_yes = pred_p >= 0.5
            if pred_yes == actual:
                hits += 1
            total += 1
        if total:
            rows.append({"prop_key": key, "accuracy": hits / total, "n": total})

    if not rows:
        return {"n_fights": 0.0}

    df = pd.DataFrame(rows)
    return {
        "n_fights": float(len(predictions)),
        "mean_prop_accuracy": float(df["accuracy"].mean()),
        "props_scored": float(len(df)),
    }


def simulate_prop_bets(
    predictions: pd.DataFrame,
    *,
    min_edge: float | None = None,
    initial_bankroll: float = 1000.0,
    flat_stake: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Flat-stake prop value betting with synthetic market lines."""
    if not config.ENABLE_PROPS:
        empty = pd.DataFrame()
        return empty, {"trades": 0.0, "roi_pct": 0.0, "hit_rate": 0.0}

    edge_floor = min_edge if min_edge is not None else config.PROP_MIN_EDGE
    bankroll = initial_bankroll
    rows: list[dict] = []

    for _, row in predictions.iterrows():
        for cand in extract_prop_candidates_for_row(row, strategy=StrategyConfig(min_edge=edge_floor)):
            if cand.edge < edge_floor:
                continue
            won = settle_prop(cand.prop_key, row)
            if won is None:
                continue
            pnl = flat_stake * (cand.decimal_odds - 1) if won else -flat_stake
            bankroll += pnl
            rows.append(
                {
                    config.FIGHT_ID_COLUMN: row.get(config.FIGHT_ID_COLUMN),
                    config.DATE_COLUMN: row.get(config.DATE_COLUMN),
                    "fighter_1": row.get("fighter_1", row.get("fighter1")),
                    "fighter_2": row.get("fighter_2", row.get("fighter2")),
                    "market_type": "prop",
                    "prop_key": cand.prop_key,
                    "label": cand.display_label,
                    "edge": cand.edge,
                    "edge_pct": cand.edge * 100.0,
                    "stake": flat_stake,
                    "odds": cand.decimal_odds,
                    "odds_source": getattr(cand, "odds_source", "synthetic"),
                    "won": int(won),
                    "pnl": pnl,
                    "equity": bankroll,
                }
            )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {
            "trades": 0.0,
            "hit_rate": 0.0,
            "total_pnl": 0.0,
            "final_equity": initial_bankroll,
            "roi_pct": 0.0,
        }

    summary = {
        "trades": float(len(trades)),
        "hit_rate": float(trades["won"].mean()),
        "total_pnl": float(trades["pnl"].sum()),
        "final_equity": float(trades["equity"].iloc[-1]),
        "roi_pct": float((trades["equity"].iloc[-1] - initial_bankroll) / initial_bankroll * 100),
    }
    return trades, summary


def simulate_mixed_parlays(
    predictions: pd.DataFrame,
    *,
    book: str = "DraftKings",
    initial_bankroll: float = 1000.0,
    flat_stake: float = 10.0,
    max_parlays_per_card: int = 3,
    allow_despite_ha: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate top prop/mixed parlays per event card (DK / MyBookie book rules).

    Defaults ``allow_despite_ha=True`` so research/backtest can score parlays
    even while live HA keeps PROP_PARLAYS_ENABLED=False. Live tickets still use
    ``build_prop_parlay_candidates`` without the override.
    """
    if not config.ENABLE_PROPS:
        return pd.DataFrame(), {"trades": 0.0, "roi_pct": 0.0}

    rules = config.BOOK_PROP_RULES.get(book, {})
    if not rules.get("allow_prop_parlays"):
        return pd.DataFrame(), {"trades": 0.0, "roi_pct": 0.0, "note": 0.0}

    bankroll = initial_bankroll
    rows: list[dict] = []
    event_col = "event_name" if "event_name" in predictions.columns else "event"
    if event_col not in predictions.columns:
        groups = [("card", predictions)]
    else:
        groups = list(predictions.groupby(event_col, dropna=False))

    for event_key, card_df in groups:
        parlays = build_prop_parlay_candidates(
            card_df, book=book, allow_despite_ha=allow_despite_ha
        )[:max_parlays_per_card]
        for parlay in parlays:
            leg_results: list[bool] = []
            for leg in parlay.legs:
                if leg.market_type == "moneyline":
                    actual = row_for_leg = None
                    for _, r in card_df.iterrows():
                        if str(r.get("fight_id", "")) == leg.fight_id:
                            row_for_leg = r
                            break
                    if row_for_leg is None:
                        leg_results.append(False)
                        continue
                    actual_f1 = row_for_leg.get(config.TARGET_COLUMN)
                    if pd.isna(actual_f1):
                        leg_results.append(False)
                        continue
                    leg_results.append(
                        (leg.bet_side == "f1" and int(actual_f1) == 1)
                        or (leg.bet_side == "f2" and int(actual_f1) == 0)
                    )
                else:
                    for _, r in card_df.iterrows():
                        if str(r.get("fight_id", "")) == leg.fight_id:
                            settled = settle_prop(leg.prop_key, r)
                            leg_results.append(bool(settled))
                            break
                    else:
                        leg_results.append(False)

            won = all(leg_results) and len(leg_results) == len(parlay.legs)
            pnl = flat_stake * (parlay.combined_odds - 1) if won else -flat_stake
            bankroll += pnl
            rows.append(
                {
                    "event": event_key,
                    "book": book,
                    "market_type": "parlay",
                    "n_legs": len(parlay.legs),
                    "combined_odds": parlay.combined_odds,
                    "combined_prob": parlay.combined_prob,
                    "correlation_adjusted": int(parlay.correlation_adjusted),
                    "won": int(won),
                    "stake": flat_stake,
                    "pnl": pnl,
                    "equity": bankroll,
                }
            )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {"trades": 0.0, "roi_pct": 0.0, "hit_rate": 0.0}

    return trades, {
        "trades": float(len(trades)),
        "hit_rate": float(trades["won"].mean()),
        "total_pnl": float(trades["pnl"].sum()),
        "final_equity": float(trades["equity"].iloc[-1]),
        "roi_pct": float((trades["equity"].iloc[-1] - initial_bankroll) / initial_bankroll * 100),
    }
