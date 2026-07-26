"""Post-fight learning loop — thinking model → lessons for future prompts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests

from sports_bot.bank.ledger import accuracy_stats, load_rows
from sports_bot.core import config


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_lessons() -> dict[str, Any]:
    path = config.PREDICTION_LESSONS_JSON
    if not path.is_file():
        return {"updated_at": "", "lessons": [], "calibration_notes": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"updated_at": "", "lessons": [], "calibration_notes": ""}


def save_lessons(payload: dict[str, Any]) -> None:
    config.ensure_dirs()
    payload = dict(payload)
    payload["updated_at"] = _utc_now()
    config.PREDICTION_LESSONS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def lessons_prompt_block(*, max_items: int = 8) -> str:
    data = load_lessons()
    lessons = data.get("lessons") or []
    notes = str(data.get("calibration_notes") or "").strip()
    if not lessons and not notes:
        return ""
    lines = ["Prior bank lessons (apply when the spot matches):"]
    for item in lessons[:max_items]:
        if isinstance(item, dict):
            tag = str(item.get("tag") or "").strip()
            text = str(item.get("lesson") or "").strip()
            if text:
                lines.append(f"- [{tag}] {text}" if tag else f"- {text}")
        else:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")
    if notes:
        lines.append(f"Calibration: {notes}")
    return "\n".join(lines)


def _ollama_generate(prompt: str, *, model: str | None = None) -> str:
    if not config.OLLAMA_ENABLED:
        raise RuntimeError("Ollama disabled")
    model = model or config.OLLAMA_THINK_MODEL or config.OLLAMA_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 900},
    }
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=config.OLLAMA_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return str(resp.json().get("response") or "")


def run_thinking_review(*, max_cases: int = 12) -> dict[str, Any]:
    """
    Review settled bank rows with a thinking model; write lessons JSON.

    Skeleton: calls Ollama when available; otherwise returns a dry-run structure.
    """
    stats = accuracy_stats()
    settled = [r for r in load_rows() if r.get("status") == "settled"][-max_cases:]
    if not settled:
        return {"ok": False, "error": "No settled predictions yet.", "stats": stats}

    case_lines = [
        f"- {r.get('selection')} @ {r.get('prob')} odds={r.get('odds')} "
        f"actual={r.get('actual')} correct={r.get('correct')} reasons={r.get('reasons')}"
        for r in settled
    ]
    prompt = f"""You are a sports betting auditor. Accuracy={stats.get('accuracy')} on {stats.get('settled')} settled.
Cases:
{chr(10).join(case_lines)}

Return ONLY JSON:
{{"calibration_notes":"...","lessons":[{{"tag":"style|odds|cardio|short-notice","lesson":"..."}}]}}
"""
    if not config.OLLAMA_ENABLED:
        return {"ok": False, "error": "OLLAMA_ENABLED=false", "stats": stats, "dry_run_prompt": prompt}

    try:
        raw = _ollama_generate(prompt)
        parsed = json.loads(raw)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "stats": stats}

    existing = load_lessons()
    merged = list(parsed.get("lessons") or []) + list(existing.get("lessons") or [])
    save_lessons(
        {
            "lessons": merged[:40],
            "calibration_notes": parsed.get("calibration_notes")
            or existing.get("calibration_notes", ""),
        }
    )
    return {"ok": True, "stats": stats, "lessons": merged[:12], "model": config.OLLAMA_THINK_MODEL}
