"""
Prediction bank — log card picks with reasons, settle vs results, learn via Ollama.

Stores an append-friendly CSV ledger and a compact lessons JSON that the analysis
prompt injects so the thinking model can improve future narratives/sizing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config

logger = logging.getLogger(__name__)

BANK_FIELDS = [
    "prediction_id",
    "logged_at",
    "event",
    "fight_id",
    "fighter_1",
    "fighter_2",
    "pick",
    "pick_prob",
    "prob_f1",
    "prob_f2",
    "confidence",
    "edge_pct",
    "odds",
    "book",
    "weight_class",
    "odds_bucket",
    "prop_type",
    "market_type",
    "uncertainty_level",
    "stake",
    "pnl",
    "closing_odds",
    "clv",
    "settlement_complete",
    "profile",
    "reason_brief",
    "reason_shap",
    "reason_gym",
    "reason_ollama",
    "f1_gym",
    "f2_gym",
    "f1_gym_strengths",
    "f2_gym_strengths",
    "status",
    "actual_winner",
    "correct",
    "settled_at",
    "method",
    "lesson",
    "lesson_at",
]


def bank_csv_path() -> Path:
    return Path(getattr(config, "PREDICTION_BANK_CSV", config.DATA_DIR / "prediction_bank.csv"))


def predictions_log_path() -> Path:
    log_dir = Path(getattr(config, "LOG_DIR", config.DATA_DIR / "logs"))
    return log_dir / "predictions.log"


def lessons_path() -> Path:
    return Path(
        getattr(config, "PREDICTION_LESSONS_JSON", config.DATA_DIR / "prediction_lessons.json")
    )


def append_predictions_log(line: str) -> None:
    """Append one human-readable line to data/logs/predictions.log."""
    try:
        path = predictions_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = line.rstrip("\n") + "\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text)
    except Exception as exc:
        logger.debug("predictions.log write failed: %s", exc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _clean(name: Any) -> str:
    try:
        from src.data_loader import clean_fighter_name

        return clean_fighter_name(name)
    except Exception:
        return re.sub(r"\s+", " ", str(name or "")).strip()


def _prediction_id(event: str, fight_id: str, f1: str, f2: str, pick: str) -> str:
    raw = "|".join(
        [
            str(event or "").strip().lower(),
            str(fight_id or "").strip().lower(),
            _clean(f1).lower(),
            _clean(f2).lower(),
            _clean(pick).lower(),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _ensure_bank(path: Path | None = None) -> Path:
    path = path or bank_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=BANK_FIELDS).writeheader()
    return path


def load_bank(path: Path | str | None = None) -> pd.DataFrame:
    """Load the prediction ledger (empty frame with schema if missing)."""
    p = Path(path) if path else bank_csv_path()
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame(columns=BANK_FIELDS)
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception as exc:
        logger.warning("Failed to read prediction bank: %s", exc)
        return pd.DataFrame(columns=BANK_FIELDS)
    for col in BANK_FIELDS:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(str).fillna("")
    return df


def _save_bank(df: pd.DataFrame, path: Path | None = None) -> None:
    path = _ensure_bank(path)
    out = df.copy()
    for col in BANK_FIELDS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str).fillna("")
    out[BANK_FIELDS].to_csv(path, index=False, encoding="utf-8")


def _top_shap(row: pd.Series | dict[str, Any]) -> str:
    try:
        from src.explainability import parse_explanation_json

        raw = row.get("shap_explanation") if hasattr(row, "get") else None
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return ""
        exp = parse_explanation_json(raw)
        toward = exp.get("toward_pick") or exp.get("top_features") or []
        parts = []
        for item in toward[:4]:
            label = str(item.get("label") or item.get("feature") or "").strip()
            if label:
                parts.append(label)
        return "; ".join(parts)
    except Exception:
        return ""


def _row_reasons(row: pd.Series) -> dict[str, str]:
    brief = ""
    try:
        from src.fight_brief import build_fight_brief

        brief = build_fight_brief(row, max_len=220)
    except Exception:
        brief = str(row.get("reasoning") or "")[:220]

    gym = str(row.get("gym_matchup_note") or "").strip()
    if not gym:
        try:
            from src.gym_data import format_gym_cell

            gym = format_gym_cell(row)
        except Exception:
            gym = ""

    sos = str(row.get("sos_competition_note") or "").strip()
    if sos:
        gym = f"{gym} | {sos}".strip(" |") if gym else sos

    return {
        "reason_brief": brief,
        "reason_shap": _top_shap(row),
        "reason_gym": gym[:400],
        "reason_ollama": str(row.get("grok_narrative") or row.get("ollama_narrative") or "")[:500],
        "f1_gym": str(row.get("f1_gym") or ""),
        "f2_gym": str(row.get("f2_gym") or ""),
        "f1_gym_strengths": str(row.get("f1_gym_strengths") or ""),
        "f2_gym_strengths": str(row.get("f2_gym_strengths") or ""),
    }


def log_prediction_row(
    row: pd.Series | dict[str, Any],
    *,
    event: str = "",
    book: str = "",
    path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Upsert one fight prediction into the bank. Returns the logged row or None."""
    if isinstance(row, dict):
        row = pd.Series(row)

    f1 = str(row.get("fighter_1") or row.get("fighter1") or "").strip()
    f2 = str(row.get("fighter_2") or row.get("fighter2") or "").strip()
    pick = str(row.get("predicted_winner") or row.get("pick") or "").strip()
    if not f1 or not f2 or not pick:
        return None

    event_name = str(event or row.get("event_name") or row.get("event") or "").strip()
    fight_id = str(row.get("fight_id") or "").strip()
    pid = _prediction_id(event_name, fight_id, f1, f2, pick)

    pick_prob = row.get("predicted_prob", row.get("prob"))
    if pd.isna(pick_prob) if not isinstance(pick_prob, str) else False:
        p1 = pd.to_numeric(row.get("prob_f1_win"), errors="coerce")
        p2 = pd.to_numeric(row.get("prob_f2_win"), errors="coerce")
        pick_prob = float(p1) if _clean(pick) == _clean(f1) else float(p2) if pd.notna(p2) else ""

    edge = row.get("best_edge", row.get("edge"))
    edge_pct = ""
    if edge is not None and not (isinstance(edge, float) and pd.isna(edge)):
        try:
            edge_pct = f"{float(edge) * 100:+.1f}"
        except (TypeError, ValueError):
            edge_pct = str(row.get("edge_pct") or "")

    reasons = _row_reasons(row)
    odds_str = _fmt_odds(row, pick, f1)
    try:
        from src.strategy_performance import classify_bet_segments

        segs = classify_bet_segments(
            weight_class=row.get("weight_class"),
            decimal_odds=float(odds_str) if odds_str else row.get("decimal_odds"),
            confidence_label=row.get("confidence_label") or row.get("confidence"),
            prop_type=row.get("prop_type") or row.get("prop_key") or "moneyline",
            market_type=str(row.get("market_type") or "moneyline"),
        )
    except Exception:
        segs = {
            "weight_class": str(row.get("weight_class") or "unknown"),
            "odds_bucket": "unknown",
            "confidence_label": str(row.get("confidence_label") or row.get("confidence") or "unknown"),
            "prop_type": "moneyline",
        }

    stake_val = row.get("stake") or row.get("suggested_stake") or ""
    try:
        stake_str = f"{float(stake_val):.2f}" if stake_val not in ("", None) and pd.notna(stake_val) else ""
    except (TypeError, ValueError):
        stake_str = ""

    unc_level = ""
    try:
        from src.sleeve_stats import uncertainty_level_from_row

        unc_level = uncertainty_level_from_row(row)
        if unc_level == "unknown":
            unc_level = ""
    except Exception:
        unc_level = str(row.get("uncertainty_level") or row.get("uncertainty_label") or "").strip()

    record = {
        "prediction_id": pid,
        "logged_at": _utc_now(),
        "event": event_name,
        "fight_id": fight_id,
        "fighter_1": f1,
        "fighter_2": f2,
        "pick": pick,
        "pick_prob": f"{float(pick_prob):.4f}" if pick_prob not in ("", None) and pd.notna(pick_prob) else "",
        "prob_f1": _fmt_prob(row.get("prob_f1_win")),
        "prob_f2": _fmt_prob(row.get("prob_f2_win")),
        "confidence": str(row.get("confidence_label") or row.get("confidence") or ""),
        "edge_pct": edge_pct,
        "odds": odds_str,
        "book": book,
        "weight_class": segs.get("weight_class", "unknown"),
        "odds_bucket": segs.get("odds_bucket", "unknown"),
        "prop_type": segs.get("prop_type", "moneyline"),
        "market_type": str(row.get("market_type") or "moneyline"),
        "uncertainty_level": unc_level,
        "stake": stake_str,
        "pnl": "",
        "closing_odds": "",
        "clv": "",
        "settlement_complete": "",
        "profile": str(getattr(config, "UFC_PROFILE", "paper")),
        **reasons,
        "status": "open",
        "actual_winner": "",
        "correct": "",
        "settled_at": "",
        "method": "",
        "lesson": "",
        "lesson_at": "",
    }

    bank_path = Path(path) if path else bank_csv_path()
    df = load_bank(bank_path)
    if not df.empty and (df["prediction_id"] == pid).any():
        # Refresh reasons/probs on open rows; never overwrite settled outcomes.
        idx = df.index[df["prediction_id"] == pid][0]
        if str(df.at[idx, "status"]).lower() == "settled":
            return df.loc[idx].to_dict()
        for k, v in record.items():
            if k in ("status", "actual_winner", "correct", "settled_at", "method", "lesson", "lesson_at"):
                continue
            df.at[idx, k] = v
        _save_bank(df, bank_path)
        return df.loc[idx].to_dict()

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    _save_bank(df, bank_path)
    append_predictions_log(
        f"{record['logged_at']} | {event_name or '-'} | {f1} vs {f2} | "
        f"pick={pick} | p={record.get('pick_prob') or '-'} | "
        f"edge={record.get('edge_pct') or '-'}% | conf={record.get('confidence') or '-'} | "
        f"status=open | id={pid}"
    )
    return record


def _fmt_prob(val: Any) -> str:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return ""


def _fmt_odds(row: pd.Series, pick: str, f1: str) -> str:
    try:
        if _clean(pick) == _clean(f1):
            o = row.get("f1_odds", row.get("decimal_odds"))
        else:
            o = row.get("f2_odds", row.get("decimal_odds"))
        if o is None or (isinstance(o, float) and pd.isna(o)):
            return ""
        return f"{float(o):.2f}"
    except (TypeError, ValueError):
        return ""


def bank_predictions(
    preds: pd.DataFrame,
    *,
    event: str = "",
    book: str = "",
    path: Path | str | None = None,
) -> int:
    """Log all rows from a prediction frame. Returns count written/updated."""
    if preds is None or not isinstance(preds, pd.DataFrame) or preds.empty:
        return 0
    # Attach gym if missing so reasons are complete.
    work = preds
    if "f1_gym" not in work.columns:
        try:
            from src.gym_data import attach_gym_features

            work = attach_gym_features(work)
        except Exception:
            pass
    n = 0
    for _, row in work.iterrows():
        if log_prediction_row(row, event=event, book=book, path=path):
            n += 1
    logger.info("Prediction bank: logged/updated %s rows (event=%r)", n, event)
    return n


def bank_from_dashboard_payload(payload: dict[str, Any] | None) -> int:
    """Log predictions from a dashboard refresh payload (all cards)."""
    if not payload:
        return 0
    total = 0
    cards = payload.get("cards") or []
    if cards:
        for card in cards:
            preds = card.get("predictions")
            ev = str(card.get("event_name") or "")
            if isinstance(preds, pd.DataFrame) and not preds.empty:
                total += bank_predictions(preds, event=ev)
    elif isinstance(payload.get("combined"), pd.DataFrame):
        total += bank_predictions(payload["combined"], event=str(payload.get("event_label") or ""))
    elif isinstance(payload.get("predictions"), pd.DataFrame):
        total += bank_predictions(payload["predictions"])
    return total


def _fighters_match(a: str, b: str) -> bool:
    ca, cb = _clean(a).lower(), _clean(b).lower()
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    try:
        from src.data_loader import _fighters_same_person

        return _fighters_same_person(ca, cb)
    except Exception:
        return ca.split()[-1] == cb.split()[-1] and len(ca.split()[-1]) > 3


def settle_open_predictions(
    *,
    historical: pd.DataFrame | None = None,
    path: Path | str | None = None,
) -> dict[str, int]:
    """
    Match open bank rows to completed fights and mark correct/incorrect.

    Uses load_fights() winners when historical is not provided.
    """
    bank_path = Path(path) if path else bank_csv_path()
    df = load_bank(bank_path)
    if df.empty:
        return {"settled": 0, "open": 0, "matched": 0}

    if historical is None:
        try:
            from src.data_loader import load_fights

            historical = load_fights()
        except Exception as exc:
            logger.warning("Settlement skipped — cannot load fights: %s", exc)
            return {"settled": 0, "open": int((df["status"] != "settled").sum()), "matched": 0}

    if historical is None or historical.empty or "winner" not in historical.columns:
        return {"settled": 0, "open": int((df["status"].astype(str) != "settled").sum()), "matched": 0}

    f1c = "fighter_1" if "fighter_1" in historical.columns else "fighter1"
    f2c = "fighter_2" if "fighter_2" in historical.columns else "fighter2"
    hist = historical.dropna(subset=["winner"]).copy()
    hist["_f1"] = hist[f1c].map(_clean)
    hist["_f2"] = hist[f2c].map(_clean)
    hist["_w"] = hist["winner"].map(_clean)

    settled = 0
    for idx, row in df.iterrows():
        if str(row.get("status") or "").lower() == "settled":
            continue
        f1, f2 = str(row.get("fighter_1") or ""), str(row.get("fighter_2") or "")
        pick = str(row.get("pick") or "")
        hit = None
        for _, h in hist.iterrows():
            if (_fighters_match(f1, h["_f1"]) and _fighters_match(f2, h["_f2"])) or (
                _fighters_match(f1, h["_f2"]) and _fighters_match(f2, h["_f1"])
            ):
                hit = h
                break
        if hit is None:
            continue
        actual = str(hit.get("_w") or hit.get("winner") or "")
        if not actual:
            continue
        correct = 1 if _fighters_match(pick, actual) else 0
        settled_at = _utc_now()
        df.at[idx, "status"] = "settled"
        df.at[idx, "actual_winner"] = str(actual)
        df.at[idx, "correct"] = str(int(correct))
        df.at[idx, "settled_at"] = settled_at
        df.at[idx, "method"] = str(hit.get("method") or hit.get("finish") or "")

        from src.settlement import (
            closing_odds_from_fight_row,
            compute_clv,
            compute_pnl,
            settlement_complete,
        )

        try:
            odds_f = float(row.get("odds") or 0) if str(row.get("odds") or "").strip() else None
        except (TypeError, ValueError):
            odds_f = None
        if odds_f is not None and odds_f <= 1.0:
            odds_f = None
        try:
            stake_f = float(row.get("stake") or 0) if str(row.get("stake") or "").strip() else None
        except (TypeError, ValueError):
            stake_f = None
        if stake_f is not None and stake_f <= 0:
            stake_f = None

        close_odds = closing_odds_from_fight_row(
            hit, pick=pick, fighter_1=f1, fighter_2=f2
        )
        pnl = compute_pnl(correct=bool(correct), stake=stake_f, opening_odds=odds_f)
        clv = compute_clv(opening_odds=odds_f, closing_odds=close_odds)
        complete = settlement_complete(stake=stake_f, opening_odds=odds_f, pnl=pnl)

        df.at[idx, "pnl"] = f"{pnl:.4f}" if pnl is not None else ""
        df.at[idx, "closing_odds"] = f"{close_odds:.4f}" if close_odds is not None else ""
        df.at[idx, "clv"] = f"{clv:.6f}" if clv is not None else ""
        df.at[idx, "settlement_complete"] = "1" if complete else "0"
        if stake_f is not None and not str(row.get("stake") or "").strip():
            df.at[idx, "stake"] = f"{stake_f:.2f}"

        if not str(row.get("weight_class") or "").strip() and "weight_class" in hit.index:
            try:
                from src.strategy_performance import normalize_weight_class

                df.at[idx, "weight_class"] = normalize_weight_class(hit.get("weight_class"))
            except Exception:
                pass
        if not str(row.get("odds_bucket") or "").strip():
            try:
                from src.strategy_performance import odds_bucket_from_decimal

                df.at[idx, "odds_bucket"] = odds_bucket_from_decimal(odds_f)
            except Exception:
                df.at[idx, "odds_bucket"] = "unknown"

        # Keep segment tags on the settled row for journal / performance sync
        weight_class = str(df.at[idx, "weight_class"] or row.get("weight_class") or "unknown")
        odds_bucket = str(df.at[idx, "odds_bucket"] or row.get("odds_bucket") or "unknown")
        prop_type = str(row.get("prop_type") or "moneyline")
        confidence = str(row.get("confidence") or "")

        try:
            from src.bet_journal import log_settlement

            log_settlement(
                prediction_id=str(row.get("prediction_id") or ""),
                event=str(row.get("event") or ""),
                fight=f"{f1} vs {f2}",
                pick=pick,
                correct=bool(correct),
                stake=stake_f,
                opening_odds=odds_f,
                closing_odds=close_odds,
                pnl=pnl,
                clv=clv,
                weight_class=weight_class,
                odds_bucket=odds_bucket,
                prop_type=prop_type,
                confidence=confidence,
                profile=str(row.get("profile") or config.UFC_PROFILE),
                settlement_complete=complete,
            )
        except Exception as exc:
            logger.debug("bet_journal settlement log skipped: %s", exc)

        try:
            from src.strategy_performance import record_closed_bet

            record_closed_bet(
                prediction_id=str(row.get("prediction_id") or ""),
                settled_at=settled_at,
                profile=str(row.get("profile") or config.UFC_PROFILE),
                market_type=str(row.get("market_type") or "moneyline"),
                weight_class=weight_class,
                odds_bucket=odds_bucket,
                confidence_label=confidence,
                prop_type=prop_type,
                correct=bool(correct),
                stake=stake_f if stake_f is not None else 0.0,
                pnl=pnl,
                decimal_odds=odds_f,
                closing_odds=close_odds,
                clv=clv,
                settlement_complete=complete,
                edge=(
                    float(row.get("edge_pct")) / 100.0
                    if str(row.get("edge_pct") or "").strip()
                    else None
                ),
                source="prediction_bank",
            )
        except Exception as exc:
            logger.debug("strategy_performance settle ingest skipped: %s", exc)

        settled += 1
        append_predictions_log(
            f"{settled_at} | {row.get('event') or '-'} | {f1} vs {f2} | "
            f"pick={pick} | actual={actual} | correct={correct} | "
            f"pnl={df.at[idx, 'pnl'] or 'n/a'} | clv={df.at[idx, 'clv'] or 'n/a'} | "
            f"complete={int(complete)} | status=settled | "
            f"id={row.get('prediction_id') or '-'}"
        )

    if settled:
        _save_bank(df, bank_path)
        try:
            from src.strategy_rating import clear_rating_cache

            clear_rating_cache()
        except Exception as exc:
            logger.debug("strategy_rating cache clear after settle skipped: %s", exc)
    open_n = int((df["status"].astype(str).str.lower() != "settled").sum())
    return {"settled": settled, "open": open_n, "matched": settled}


def accuracy_stats(path: Path | str | None = None) -> dict[str, Any]:
    """Aggregate accuracy / calibration snapshot for UI and thinking prompts."""
    df = load_bank(path)
    if df.empty:
        return {
            "total": 0,
            "open": 0,
            "settled": 0,
            "correct": 0,
            "accuracy": None,
            "by_confidence": {},
            "recent": [],
        }
    settled = df[df["status"].astype(str).str.lower() == "settled"].copy()
    open_n = int((df["status"].astype(str).str.lower() != "settled").sum())
    correct_n = 0
    if not settled.empty:
        correct_n = int(pd.to_numeric(settled["correct"], errors="coerce").fillna(0).sum())
    acc = (correct_n / len(settled)) if len(settled) else None

    by_conf: dict[str, dict[str, Any]] = {}
    if not settled.empty:
        for conf, grp in settled.groupby(settled["confidence"].astype(str).fillna("")):
            c = int(pd.to_numeric(grp["correct"], errors="coerce").fillna(0).sum())
            by_conf[str(conf) or "unknown"] = {
                "n": len(grp),
                "correct": c,
                "accuracy": c / len(grp) if len(grp) else None,
            }

    recent = []
    show = settled.sort_values("settled_at", ascending=False).head(12) if not settled.empty else df.head(0)
    for _, r in show.iterrows():
        recent.append(
            {
                "event": r.get("event"),
                "fight": f"{r.get('fighter_1')} vs {r.get('fighter_2')}",
                "pick": r.get("pick"),
                "actual": r.get("actual_winner"),
                "correct": bool(int(pd.to_numeric(r.get("correct"), errors="coerce") or 0)),
                "prob": r.get("pick_prob"),
                "lesson": r.get("lesson"),
            }
        )

    return {
        "total": len(df),
        "open": open_n,
        "settled": len(settled),
        "correct": correct_n,
        "accuracy": acc,
        "by_confidence": by_conf,
        "recent": recent,
    }


def load_lessons() -> dict[str, Any]:
    path = lessons_path()
    if not path.is_file():
        return {"updated_at": "", "lessons": [], "calibration_notes": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"updated_at": "", "lessons": [], "calibration_notes": ""}
        data.setdefault("lessons", [])
        data.setdefault("calibration_notes", "")
        return data
    except (OSError, json.JSONDecodeError):
        return {"updated_at": "", "lessons": [], "calibration_notes": ""}


def save_lessons(payload: dict[str, Any]) -> None:
    path = lessons_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = _utc_now()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def lessons_prompt_block(*, max_items: int = 8) -> str:
    """Short block injected into Ollama analysis prompts."""
    data = load_lessons()
    lessons = data.get("lessons") or []
    notes = str(data.get("calibration_notes") or "").strip()
    if not lessons and not notes:
        return ""
    lines = ["Prior bank lessons (apply carefully to similar matchups):"]
    for item in lessons[:max_items]:
        if isinstance(item, dict):
            text = str(item.get("lesson") or item.get("text") or "").strip()
            tag = str(item.get("tag") or item.get("theme") or "").strip()
            if text:
                lines.append(f"- [{tag}] {text}" if tag else f"- {text}")
        else:
            text = str(item).strip()
            if text:
                lines.append(f"- {text}")
    if notes:
        lines.append(f"Calibration: {notes}")
    return "\n".join(lines)


def _thinking_model_name() -> str:
    """Prefer a reasoning model when installed; else configured OLLAMA_MODEL."""
    preferred = str(
        getattr(config, "PREDICTION_BANK_THINK_MODEL", "")
        or getattr(config, "OLLAMA_THINK_MODEL", "")
        or "deepseek-r1:8b"
    ).strip()
    try:
        from src.ollama_client import ollama_installed_models, resolve_model

        installed = ollama_installed_models()
        pick = resolve_model(preferred, installed)
        if pick:
            return pick
        # Fall back to primary accuracy model
        primary = str(getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:14b"))
        return resolve_model(primary, installed) or primary
    except Exception:
        return preferred


def run_thinking_review(
    *,
    max_rows: int = 12,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Ask a thinking model to review recent settled misses/hits and write lessons.

    Updates prediction_lessons.json and fills empty `lesson` cells on reviewed rows.
    """
    from src.grok_analysis import _extract_json_blob
    from src.ollama_client import ollama_complete

    settle_open_predictions(path=path)
    df = load_bank(path)
    settled = df[df["status"].astype(str).str.lower() == "settled"].copy()
    if settled.empty:
        return {"ok": False, "error": "No settled predictions yet — wait for fight results.", "lessons": []}

    # Prefer unsettled lessons first, then recent
    need = settled[settled["lesson"].astype(str).str.strip() == ""].copy()
    if need.empty:
        need = settled.copy()
    need = need.sort_values("settled_at", ascending=False).head(max_rows)

    stats = accuracy_stats(path)
    case_lines = []
    for _, r in need.iterrows():
        case_lines.append(
            f"- id={r.get('prediction_id')} | {r.get('fighter_1')} vs {r.get('fighter_2')} | "
            f"pick={r.get('pick')} ({r.get('pick_prob')}) | actual={r.get('actual_winner')} | "
            f"correct={r.get('correct')} | shap={r.get('reason_shap')} | "
            f"brief={r.get('reason_brief')} | gym={r.get('reason_gym')} | "
            f"prior_ollama={r.get('reason_ollama')}"
        )

    prompt = f"""You are a UFC prediction auditor. Improve future betting analysis using settled results.

Bank accuracy: {stats.get('accuracy')} on {stats.get('settled')} settled fights ({stats.get('correct')} correct).
Confidence breakdown: {json.dumps(stats.get('by_confidence') or {})}

Cases:
{chr(10).join(case_lines)}

Return ONLY valid JSON:
{{
  "calibration_notes": "how the model is over/under confident overall",
  "lessons": [
    {{"tag": "wrestling|striking|cardio|odds|gym|short-notice|...", "lesson": "actionable rule for future similar spots"}}
  ],
  "row_lessons": [
    {{"id": "prediction_id", "lesson": "1-2 sentence postmortem for this fight"}}
  ]
}}
Focus on transferable mistakes and edges — not generic advice. Prefer accuracy over brevity."""

    model = _thinking_model_name()
    timeout = int(getattr(config, "PREDICTION_BANK_THINK_TIMEOUT_SEC", 0) or 0) or int(
        getattr(config, "OLLAMA_TIMEOUT_SEC", 600)
    )
    try:
        model_used, raw = ollama_complete(
            prompt,
            system="You are a careful MMA analyst. JSON only.",
            model=model,
            timeout_sec=timeout,
            json_mode=True,
            temperature=0.2,
        )
        parsed = _extract_json_blob(raw)
    except Exception as exc:
        logger.warning("Thinking review failed: %s", exc)
        return {"ok": False, "error": str(exc), "lessons": [], "model": model}

    lessons = parsed.get("lessons") or []
    calibration = str(parsed.get("calibration_notes") or "").strip()
    # Merge with existing lessons (newest first, dedupe by text)
    existing = load_lessons()
    merged: list[Any] = []
    seen: set[str] = set()
    for item in list(lessons) + list(existing.get("lessons") or []):
        text = str(item.get("lesson") if isinstance(item, dict) else item).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(item)
    save_lessons({"lessons": merged[:40], "calibration_notes": calibration or existing.get("calibration_notes", "")})

    # Write per-row lessons
    row_lessons = {str(x.get("id")): str(x.get("lesson") or "") for x in (parsed.get("row_lessons") or []) if x.get("id")}
    bank = load_bank(path)
    updated = 0
    for idx, row in bank.iterrows():
        pid = str(row.get("prediction_id") or "")
        if pid in row_lessons and row_lessons[pid]:
            bank.at[idx, "lesson"] = row_lessons[pid][:600]
            bank.at[idx, "lesson_at"] = _utc_now()
            updated += 1
    if updated:
        _save_bank(bank, Path(path) if path else bank_csv_path())

    return {
        "ok": True,
        "model": model_used,
        "lessons": merged[:12],
        "calibration_notes": calibration,
        "rows_updated": updated,
        "stats": accuracy_stats(path),
    }
