"""Signed, capped matchup rules applied on top of the learned ensemble.

These are explicit log-odds nudges (toward fighter 1 when positive). They use
columns already on the feature row — prior fights only — and never enter
``FEATURE_COLUMNS`` or retrain the model. The total is clipped to
``config.STYLE_BONUS_MAX`` so several agreeing facts cannot stack into a
large probability move.

Card order is not a feature: every rule is signed. A southpaw edge favors the
southpaw, not whoever is listed as fighter 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

import config

# Rules whose inputs are matchup flags, not f1−f2 diffs. Mirroring a row
# leaves these unchanged.
UNSIGNED_FLAGS = frozenset(
    {
        "stance_matchup",
        "striker_vs_grappler",
        "style_clash",
    }
)


@dataclass(frozen=True)
class ProgrammedRuleHit:
    """One fired rule. ``bonus`` is log-odds toward fighter 1 after the cap."""

    name: str
    label: str
    bonus: float


def _num(row: pd.Series | dict[str, Any], *keys: str) -> float | None:
    getter = row.get if hasattr(row, "get") else None
    if getter is None:
        return None
    for key in keys:
        try:
            raw = getter(key)
        except Exception:
            continue
        if raw is None:
            continue
        try:
            if pd.isna(raw):
                continue
        except (TypeError, ValueError):
            pass
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _sign(value: float) -> float:
    if value > 1e-12:
        return 1.0
    if value < -1e-12:
        return -1.0
    return 0.0


def _same_direction(a: float, b: float) -> bool:
    return _sign(a) != 0.0 and _sign(a) == _sign(b)


def _collect(row: pd.Series | dict[str, Any]) -> list[ProgrammedRuleHit]:
    hits: list[ProgrammedRuleHit] = []

    def add(name: str, label: str, bonus: float) -> None:
        if not math.isfinite(bonus) or abs(bonus) < 1e-9:
            return
        hits.append(ProgrammedRuleHit(name, label, float(bonus)))

    # Southpaw vs orthodox. ``southpaw_advantage`` is already ±0.08 toward the
    # southpaw. Do not add ``stance_matchup`` on its own — that flag is 1 for
    # either order and used to give fighter 1 a free +0.01.
    southpaw = _num(row, "southpaw_advantage")
    if southpaw is not None and abs(southpaw) > 1e-9:
        add("southpaw", "Southpaw vs orthodox", southpaw * 0.5)

    striker_vs_grappler = _num(row, "striker_vs_grappler") or 0.0
    striker_diff = _num(row, "striker_score_diff")
    if striker_vs_grappler >= 0.5 and striker_diff is not None and abs(striker_diff) > 1e-9:
        add("striker_path", "Striker vs grappler", 0.04 * _sign(striker_diff))

    style_clash = _num(row, "style_clash") or 0.0
    grappler_diff = _num(row, "grappler_score_diff")
    if style_clash >= 0.5 and grappler_diff is not None and abs(grappler_diff) > 1e-9:
        add("grappler_path", "Style-clash grappler", 0.02 * _sign(grappler_diff))

    # Length only helps when the longer fighter also lands. Volume is extra.
    reach = _num(row, "reach_diff")
    accuracy = _num(row, "striking_acc_diff")
    if (
        reach is not None
        and accuracy is not None
        and abs(reach) >= 3.0
        and abs(accuracy) >= 0.02
        and _same_direction(reach, accuracy)
    ):
        bonus = 0.012 * _sign(reach)
        volume = _num(row, "sig_strikes_per_min_diff")
        if volume is not None and abs(volume) >= 0.4 and _same_direction(volume, reach):
            bonus += 0.006 * _sign(reach)
        add("reach_lands", "Reach that lands", bonus)

    height = _num(row, "height_diff")
    if (
        height is not None
        and reach is not None
        and abs(height) >= 2.0
        and abs(reach) >= 2.0
        and _same_direction(height, reach)
    ):
        add("length_stack", "Height and reach", 0.008 * _sign(reach))

    # One camp-length penalty. Prefer the 365-day flag over the 180-day flag
    # so the two do not stack. A strong record in that spot shrinks the tax.
    short_notice = _num(row, "hv_short_notice_flag_diff")
    if short_notice is None or abs(short_notice) < 0.5:
        short_notice = _num(row, "short_notice_flag_diff")
    if short_notice is not None and abs(short_notice) >= 0.5:
        penalty = 0.016 * _sign(short_notice)
        perf = _num(row, "short_notice_perf_diff")
        if perf is not None and _same_direction(perf, short_notice) and abs(perf) >= 0.05:
            penalty *= 0.4
        add("short_notice", "Short-notice camp", -penalty)

    year_layoff = _num(row, "hv_long_layoff_flag_diff")
    half_year = _num(row, "long_layoff_flag_diff")
    layoff_flag: float | None = None
    layoff_mag = 0.0
    if year_layoff is not None and abs(year_layoff) >= 0.5:
        layoff_flag = year_layoff
        layoff_mag = 0.014
    elif half_year is not None and abs(half_year) >= 0.5:
        layoff_flag = half_year
        layoff_mag = 0.010
    if layoff_flag is not None:
        penalty = layoff_mag * _sign(layoff_flag)
        perf = _num(row, "long_layoff_perf_diff")
        if perf is not None and _same_direction(perf, layoff_flag) and abs(perf) >= 0.05:
            penalty *= 0.4
        add("layoff_rust", "Layoff rust", -penalty)

    new_wc = _num(row, "first_fight_new_wc_flag_diff")
    if new_wc is not None and abs(new_wc) >= 0.5:
        add("new_division", "First fight in a new division", -0.012 * _sign(new_wc))

    chin = _num(row, "ko_losses_career_flag_diff")
    if chin is not None and abs(chin) >= 0.5:
        bonus = -0.010 * _sign(chin)
        power = _num(row, "power_proxy_diff")
        ko_rate = _num(row, "ko_rate_diff")
        opponent_power = False
        if power is not None and abs(power) >= 0.05 and not _same_direction(power, chin):
            if _sign(power) != 0.0:
                opponent_power = True
        if ko_rate is not None and abs(ko_rate) >= 0.05 and not _same_direction(ko_rate, chin):
            if _sign(ko_rate) != 0.0:
                opponent_power = True
        if opponent_power:
            bonus += -0.006 * _sign(chin)
        add("chin", "Prior KO loss", bonus)

    age_adj = _num(row, "division_age_adj_diff")
    if age_adj is not None and abs(age_adj) >= 3.0:
        add("past_peak", "Past division peak", -0.010 * _sign(age_adj))

    # Grappler path is open only when takedown accuracy agrees and the other
    # fighter does not own a clear takedown-defense wall.
    if style_clash >= 0.5 and grappler_diff is not None and abs(grappler_diff) > 1e-9:
        td_acc = _num(row, "takedown_acc_diff")
        td_def = _num(row, "td_defense_diff")
        if td_acc is not None and abs(td_acc) >= 0.08 and _same_direction(td_acc, grappler_diff):
            wall = (
                td_def is not None
                and abs(td_def) >= 0.10
                and _sign(td_def) != 0.0
                and not _same_direction(td_def, grappler_diff)
            )
            if not wall:
                add("wrestling_path", "Open wrestling path", 0.010 * _sign(grappler_diff))

    last5 = _num(row, "last5_winrate_diff")
    momentum = _num(row, "momentum_diff")
    if (
        last5 is not None
        and momentum is not None
        and abs(last5) >= 0.40
        and abs(momentum) >= 0.15
        and _same_direction(last5, momentum)
    ):
        add("form_streak", "Recent form streak", 0.008 * _sign(last5))

    experience = _num(row, "experience_diff")
    if (
        experience is not None
        and last5 is not None
        and abs(experience) >= 8.0
        and abs(last5) >= 0.20
        and _same_direction(experience, last5)
    ):
        add("experience_form", "Experience plus recent form", 0.008 * _sign(experience))

    return hits


def _bonus_cap() -> float:
    try:
        cap = float(getattr(config, "STYLE_BONUS_MAX", 0.05) or 0.0)
    except (TypeError, ValueError):
        cap = 0.05
    if not math.isfinite(cap) or cap < 0.0:
        return 0.0
    return cap


def evaluate_programmed_rules(
    row: pd.Series | dict[str, Any] | None,
) -> tuple[float, list[ProgrammedRuleHit]]:
    """Return ``(clipped_log_odds_toward_f1, hits)``.

    When the raw sum exceeds the cap, each hit is scaled so the parts still
    add up to the nudge that is actually applied.
    """
    if row is None:
        return 0.0, []
    hits = _collect(row)
    raw = float(sum(hit.bonus for hit in hits))
    cap = _bonus_cap()
    clipped = float(max(-cap, min(cap, raw)))
    if not hits:
        return 0.0, []
    if abs(raw) > 1e-12 and abs(clipped - raw) > 1e-9:
        scale = clipped / raw
        hits = [
            ProgrammedRuleHit(hit.name, hit.label, hit.bonus * scale)
            for hit in hits
            if abs(hit.bonus * scale) >= 1e-9
        ]
    return clipped, hits


def compute_programmed_bonus(row: pd.Series | dict[str, Any] | None) -> float:
    """Clipped log-odds nudge toward fighter 1."""
    total, _hits = evaluate_programmed_rules(row)
    return total


def format_programmed_rules(row: pd.Series | dict[str, Any] | None) -> str:
    """Compact audit string: ``name:+0.0120;other:-0.0100``."""
    _total, hits = evaluate_programmed_rules(row)
    return ";".join(f"{hit.name}:{hit.bonus:+.4f}" for hit in hits)


def programmed_rules_series(frame: pd.DataFrame) -> pd.Series:
    """One audit string per feature row, aligned to ``frame.index``."""
    if frame is None or frame.empty:
        return pd.Series(dtype=object, name="programmed_rules")
    values = [format_programmed_rules(row) for _, row in frame.iterrows()]
    return pd.Series(values, index=frame.index, name="programmed_rules")


def _side_name(row: pd.Series | dict[str, Any], *, fighter_1: bool) -> str:
    keys = ("fighter_1", "fighter1") if fighter_1 else ("fighter_2", "fighter2")
    for key in keys:
        raw = row.get(key) if hasattr(row, "get") else None
        text = str(raw or "").strip()
        if text and text.lower() != "nan":
            return text
    return "f1" if fighter_1 else "f2"


def format_programmed_line(row: pd.Series | dict[str, Any] | None) -> str | None:
    """Display line naming which fighter each fired rule favors."""
    if row is None:
        return None
    _total, hits = evaluate_programmed_rules(row)
    if not hits:
        return None
    f1 = _side_name(row, fighter_1=True)
    f2 = _side_name(row, fighter_1=False)
    bits: list[str] = []
    for hit in hits[:6]:
        side = f1 if hit.bonus > 0 else f2
        bits.append(f"{hit.label} → {side}")
    extra = len(hits) - len(bits)
    line = "Programmed: " + "; ".join(bits)
    if extra > 0:
        line += f" (+{extra} more)"
    return line


def rules_supporting_pick(row: pd.Series | dict[str, Any] | None, *, limit: int = 3) -> list[str]:
    """Labels of rules whose sign agrees with the predicted winner."""
    if row is None or not hasattr(row, "get"):
        return []
    pick = str(row.get("predicted_winner") or row.get("pick") or "").strip()
    f1 = _side_name(row, fighter_1=True)
    f2 = _side_name(row, fighter_1=False)
    if pick and pick == f2:
        want_positive = False
    elif pick and pick == f1:
        want_positive = True
    else:
        return []
    _total, hits = evaluate_programmed_rules(row)
    labels: list[str] = []
    for hit in hits:
        favors_pick = hit.bonus > 0 if want_positive else hit.bonus < 0
        if favors_pick:
            labels.append(hit.label)
        if len(labels) >= limit:
            break
    return labels
