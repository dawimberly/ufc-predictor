"""Rule-based fight brief — SHAP + edge + MC (no LLM)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.explainability import parse_explanation_json


def _top_shap_label(row: pd.Series) -> str:
    if pd.notna(row.get("shap_explanation")):
        exp = parse_explanation_json(row.get("shap_explanation"))
        toward = exp.get("toward_pick") or exp.get("top_features") or []
        if toward:
            return str(toward[0].get("label", ""))
    if pd.notna(row.get("reasoning")):
        text = str(row["reasoning"])
        if "due to" in text:
            return text.split("due to", 1)[-1].split(",")[0].strip()[:60]
    return ""


def build_fight_brief(
    row: pd.Series | dict[str, Any],
    *,
    risk_metrics: dict[str, Any] | None = None,
    edge_pct: float | None = None,
    max_len: int = 200,
) -> str:
    """
    Heuristic pre-fight narrative for alerts (fast, no Ollama).

    Composes: pick + edge + top SHAP driver + optional MC card note.
    """
    if isinstance(row, dict):
        row = pd.Series(row)

    pick = str(row.get("predicted_winner", row.get("pick", "")))
    f1 = str(row.get("fighter_1", row.get("fighter1", "")))
    f2 = str(row.get("fighter_2", row.get("fighter2", "")))
    fight = f"{f1} vs {f2}" if f1 and f2 else str(row.get("fight", ""))

    prob = row.get("predicted_prob", row.get("prob"))
    if pd.isna(prob) and pd.notna(row.get("prob_f1_win")):
        prob = float(row["prob_f1_win"]) if pick == f1 else float(row.get("prob_f2_win", 0))

    if edge_pct is None:
        if pd.notna(row.get("edge_pct")):
            edge_pct = float(row["edge_pct"])
        elif pd.notna(row.get("edge")):
            edge_pct = float(row["edge"]) * 100.0
        elif pd.notna(row.get("best_edge")):
            edge_pct = float(row["best_edge"]) * 100.0
        else:
            edge_pct = None

    parts: list[str] = []
    if pick:
        prob_txt = f" ({float(prob):.0%})" if pd.notna(prob) else ""
        parts.append(f"Take {pick}{prob_txt}")
    if edge_pct is not None:
        parts.append(f"edge {edge_pct:+.1f}%")
    driver = _top_shap_label(row)
    if driver:
        parts.append(f"key driver: {driver}")
    conf = str(row.get("confidence_label", "")).strip()
    if conf and conf not in ("", "nan"):
        parts.append(f"{conf} confidence")

    # Uncertainty gate status (display + actionable)
    unc_action = str(row.get("uncertainty_action") or "").strip().lower()
    unc_reason = str(row.get("uncertainty_reason") or row.get("skip_reason") or "").strip()
    if unc_action == "skip" and unc_reason:
        parts.append(f"SKIP:{unc_reason}")
    else:
        try:
            from src.uncertainty_gates import evaluate_uncertainty_gate

            gate = evaluate_uncertainty_gate(row)
            if gate.action == "skip":
                parts.append(f"SKIP:{gate.reason_label()}")
            elif gate.action == "tighten":
                parts.append(f"tighten:{gate.reason_label()}")
            elif gate.disagreement is not None or gate.interval_width is not None:
                d = f"d={gate.disagreement:.2f}" if gate.disagreement is not None else ""
                w = f"w={gate.interval_width:.2f}" if gate.interval_width is not None else ""
                bits = " ".join(x for x in (d, w) if x)
                if bits:
                    parts.append(bits)
        except Exception:
            if str(row.get("uncertainty_label") or "").lower() == "high":
                parts.append("high uncertainty")

    # Gym / camp note for the predicted side (or both when short)
    gym_bits: list[str] = []
    try:
        from src.gym_data import gym_narrative_line, lookup_gym

        pick_name = pick if pick in (f1, f2) else f1
        other = f2 if pick_name == f1 else f1
        prefix = "f1" if pick_name == f1 else "f2"
        gym = str(row.get(f"{prefix}_gym") or "").strip()
        strengths = str(row.get(f"{prefix}_gym_strengths") or "").strip()
        notes = str(row.get(f"{prefix}_gym_notes") or "").strip()
        local = bool(row.get(f"{prefix}_local_advantage"))
        if not gym:
            profile = lookup_gym(pick_name)
            gym, strengths, notes = profile["gym"], profile["strengths"], profile["notes"]
        else:
            profile = {"gym": gym, "strengths": strengths, "notes": notes, "location": ""}
        line = gym_narrative_line(pick_name, profile, local=local)
        if line:
            gym_bits.append(line)
        # Mention opponent gym only if very short
        other_prefix = "f2" if prefix == "f1" else "f1"
        og = str(row.get(f"{other_prefix}_gym") or "").strip()
        if og and len(" · ".join(parts + gym_bits)) < max_len - 40:
            os_ = str(row.get(f"{other_prefix}_gym_strengths") or "").split(",")[0].strip()
            gym_bits.append(f"{other}: {og}" + (f" ({os_})" if os_ else ""))
    except Exception:
        pass
    parts.extend(gym_bits)

    sos = str(row.get("sos_competition_note") or "").strip()
    if sos and len(" · ".join(parts) + sos) < max_len - 5:
        parts.append(sos)

    if risk_metrics and risk_metrics.get("available"):
        cap = risk_metrics.get("suggested_max_risk_pct")
        if cap is not None:
            parts.append(f"card cap {float(cap):.1f}%")

    if not parts:
        brief = f"Value signal on {fight or 'card'}."
    else:
        brief = " · ".join(parts)
        if fight:
            brief = f"{fight}: {brief}"

    if len(brief) > max_len:
        return brief[: max_len - 3].rstrip() + "..."
    return brief


def build_card_brief(
    alert_data: dict[str, Any],
    *,
    max_len: int = 280,
) -> str:
    """One-line card summary for Discord/Telegram header."""
    event = alert_data.get("event_name", "Upcoming card")
    n_s = alert_data.get("singles_count", 0)
    n_p = alert_data.get("parlays_count", 0)
    risk = alert_data.get("risk_summary", "")
    top = alert_data.get("singles", [{}])[0] if alert_data.get("singles") else {}
    top_pick = top.get("pick", "")
    top_edge = top.get("edge_pct")
    line = f"{event}: {n_s} singles, {n_p} parlays"
    if top_pick and top_edge is not None:
        line += f" | top {top_pick} {top_edge:+.1f}%"
    if risk:
        line += f" | {risk}"
    if len(line) > max_len:
        return line[: max_len - 3] + "..."
    return line
