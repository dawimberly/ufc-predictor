"""Rolling segment performance from settled UFC bets (trading-bot style).

Tracks closed bets by weight class, odds bucket, confidence label, and prop type.
Scores feed ``strategy_rating`` Kelly clamps once a segment has enough samples.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

import config

logger = logging.getLogger(__name__)

SEGMENT_DIMENSIONS: tuple[str, ...] = (
    "weight_class",
    "odds_bucket",
    "confidence_label",
    "prop_type",
)

SEGMENT_LABELS: dict[str, str] = {
    "weight_class": "Weight class",
    "odds_bucket": "Odds bucket",
    "confidence_label": "Confidence",
    "prop_type": "Prop type",
}


def _db_path() -> Path:
    path = Path(getattr(config, "STRATEGY_METRICS_DB", config.DATA_DIR / "strategy_metrics.db"))
    if not path.is_absolute():
        path = config.ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def performance_json_path() -> Path:
    path = Path(
        getattr(config, "STRATEGY_PERFORMANCE_JSON", config.DATA_DIR / "strategy_performance.json")
    )
    if not path.is_absolute():
        path = config.ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS closed_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT NOT NULL UNIQUE,
            settled_at TEXT NOT NULL,
            profile TEXT DEFAULT 'paper',
            market_type TEXT DEFAULT 'moneyline',
            weight_class TEXT DEFAULT 'unknown',
            odds_bucket TEXT DEFAULT 'unknown',
            confidence_label TEXT DEFAULT 'unknown',
            prop_type TEXT DEFAULT 'moneyline',
            correct INTEGER NOT NULL,
            stake REAL NOT NULL DEFAULT 0.0,
            pnl REAL,
            pnl_pct REAL,
            decimal_odds REAL,
            closing_odds REAL,
            clv REAL,
            settlement_complete INTEGER NOT NULL DEFAULT 0,
            edge REAL,
            source TEXT DEFAULT 'prediction_bank'
        );
        CREATE INDEX IF NOT EXISTS idx_closed_bets_settled ON closed_bets(settled_at);
        CREATE INDEX IF NOT EXISTS idx_closed_bets_wc ON closed_bets(weight_class);
        CREATE INDEX IF NOT EXISTS idx_closed_bets_odds ON closed_bets(odds_bucket);
        CREATE INDEX IF NOT EXISTS idx_closed_bets_conf ON closed_bets(confidence_label);
        CREATE INDEX IF NOT EXISTS idx_closed_bets_prop ON closed_bets(prop_type);
        CREATE TABLE IF NOT EXISTS segment_snapshots (
            snap_date TEXT NOT NULL,
            segment_key TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            PRIMARY KEY (snap_date, segment_key)
        );
        """
    )
    _migrate_closed_bets(conn)
    conn.commit()


def _migrate_closed_bets(conn: sqlite3.Connection) -> None:
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(closed_bets)").fetchall()}
    alters = []
    if "closing_odds" not in cols:
        alters.append("ALTER TABLE closed_bets ADD COLUMN closing_odds REAL")
    if "clv" not in cols:
        alters.append("ALTER TABLE closed_bets ADD COLUMN clv REAL")
    if "settlement_complete" not in cols:
        alters.append(
            "ALTER TABLE closed_bets ADD COLUMN settlement_complete INTEGER NOT NULL DEFAULT 0"
        )
    for sql in alters:
        conn.execute(sql)
    # Backfill: rows with stake>0, odds>1, pnl present → complete
    if alters:
        conn.execute(
            """
            UPDATE closed_bets
            SET settlement_complete = 1
            WHERE settlement_complete = 0
              AND stake IS NOT NULL AND stake > 0
              AND decimal_odds IS NOT NULL AND decimal_odds > 1.0
              AND pnl IS NOT NULL
            """
        )


def rating_from_score(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    if score > 0:
        return "Weak"
    return "No data"


def slugify_segment(value: Any, *, default: str = "unknown") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text or text in ("nan", "none", "-"):
        return default
    text = re.sub(r"(?i)\s*(interim\s+)?title\s*$", "", text).strip()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or default


def normalize_weight_class(value: Any) -> str:
    try:
        from src.data_loader import clean_weight_class

        cleaned = clean_weight_class(value)
    except Exception:
        cleaned = str(value or "Unknown")
    return slugify_segment(cleaned, default="unknown")


def odds_bucket_from_decimal(odds: Any) -> str:
    try:
        dec = float(odds)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(dec) or dec <= 1.0:
        return "unknown"
    if dec < 1.80:
        return "favorite"
    if dec <= 2.20:
        return "pickem"
    if dec <= 3.50:
        return "mild_dog"
    return "longshot"


def normalize_confidence_label(value: Any) -> str:
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


def normalize_prop_type(value: Any, *, market_type: str = "moneyline") -> str:
    mt = str(market_type or "moneyline").strip().lower()
    if mt in ("moneyline", "ml", ""):
        raw = str(value or "").strip().lower()
        if not raw or raw in ("moneyline", "ml", "winner"):
            return "moneyline"
        return slugify_segment(raw, default="moneyline")
    return slugify_segment(value or mt, default="prop")


def segment_key(dimension: str, value: str) -> str:
    return f"{dimension}:{slugify_segment(value)}"


def classify_bet_segments(
    *,
    weight_class: Any = None,
    decimal_odds: Any = None,
    confidence_label: Any = None,
    prop_type: Any = None,
    market_type: str = "moneyline",
    row: pd.Series | dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build segment tags for a bet / prediction row."""
    if row is not None:
        if isinstance(row, dict):
            row = pd.Series(row)
        if weight_class is None:
            weight_class = row.get("weight_class")
        if decimal_odds is None:
            decimal_odds = row.get("decimal_odds", row.get("odds"))
        if confidence_label is None:
            confidence_label = row.get("confidence_label", row.get("confidence"))
        if prop_type is None:
            prop_type = row.get("prop_type", row.get("prop_key", row.get("market_type")))
        if market_type == "moneyline":
            market_type = str(row.get("market_type") or row.get("bet_type") or "moneyline")

    return {
        "weight_class": normalize_weight_class(weight_class),
        "odds_bucket": odds_bucket_from_decimal(decimal_odds),
        "confidence_label": normalize_confidence_label(confidence_label),
        "prop_type": normalize_prop_type(prop_type, market_type=market_type),
    }


def _unit_pnl(*, correct: bool, decimal_odds: float | None, stake: float) -> tuple[float | None, float | None]:
    """PnL only when stake and opening odds are present (fail-closed)."""
    try:
        stake_f = float(stake) if stake is not None else None
    except (TypeError, ValueError):
        stake_f = None
    try:
        dec = float(decimal_odds) if decimal_odds is not None else None
    except (TypeError, ValueError):
        dec = None
    if stake_f is None or stake_f <= 0 or dec is None or dec <= 1.0:
        return None, None
    pnl = stake_f * (dec - 1.0) if correct else -stake_f
    return pnl, 100.0 * pnl / stake_f


def record_closed_bet(
    *,
    prediction_id: str,
    settled_at: str | None = None,
    profile: str | None = None,
    market_type: str = "moneyline",
    weight_class: str = "unknown",
    odds_bucket: str = "unknown",
    confidence_label: str = "unknown",
    prop_type: str = "moneyline",
    correct: bool | int,
    stake: float | None = None,
    pnl: float | None = None,
    pnl_pct: float | None = None,
    decimal_odds: float | None = None,
    closing_odds: float | None = None,
    clv: float | None = None,
    settlement_complete: bool | None = None,
    edge: float | None = None,
    source: str = "prediction_bank",
) -> bool:
    """Upsert one settled bet. Incomplete settlements store outcome but exclude from health."""
    pid = str(prediction_id or "").strip()
    if not pid:
        return False
    segs = classify_bet_segments(
        weight_class=weight_class,
        decimal_odds=decimal_odds,
        confidence_label=confidence_label,
        prop_type=prop_type,
        market_type=market_type,
    )
    if odds_bucket and str(odds_bucket).strip() and str(odds_bucket).lower() != "unknown":
        segs["odds_bucket"] = slugify_segment(odds_bucket)

    correct_i = 1 if int(correct) else 0
    try:
        stake_f = float(stake) if stake is not None and str(stake).strip() != "" else None
    except (TypeError, ValueError):
        stake_f = None
    if stake_f is not None and stake_f <= 0:
        stake_f = None

    if pnl is None:
        pnl, pnl_pct = _unit_pnl(
            correct=bool(correct_i),
            decimal_odds=decimal_odds,
            stake=stake_f if stake_f is not None else 0.0,
        )
    elif pnl_pct is None and stake_f and pnl is not None:
        pnl_pct = 100.0 * float(pnl) / stake_f

    if clv is None and decimal_odds is not None and closing_odds is not None:
        try:
            from src.settlement import compute_clv

            clv = compute_clv(opening_odds=decimal_odds, closing_odds=closing_odds)
        except Exception:
            clv = None

    if settlement_complete is None:
        from src.settlement import settlement_complete as _complete

        settlement_complete = _complete(stake=stake_f, opening_odds=decimal_odds, pnl=pnl)

    settled = settled_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    profile_s = config.normalize_profile(profile or getattr(config, "UFC_PROFILE", "paper"))

    with _connect() as conn:
        _init_db(conn)
        conn.execute(
            """
            INSERT INTO closed_bets (
                prediction_id, settled_at, profile, market_type,
                weight_class, odds_bucket, confidence_label, prop_type,
                correct, stake, pnl, pnl_pct, decimal_odds, closing_odds, clv,
                settlement_complete, edge, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                settled_at=excluded.settled_at,
                profile=excluded.profile,
                market_type=excluded.market_type,
                weight_class=excluded.weight_class,
                odds_bucket=excluded.odds_bucket,
                confidence_label=excluded.confidence_label,
                prop_type=excluded.prop_type,
                correct=excluded.correct,
                stake=excluded.stake,
                pnl=excluded.pnl,
                pnl_pct=excluded.pnl_pct,
                decimal_odds=excluded.decimal_odds,
                closing_odds=excluded.closing_odds,
                clv=excluded.clv,
                settlement_complete=excluded.settlement_complete,
                edge=excluded.edge,
                source=excluded.source
            """,
            (
                pid,
                settled,
                profile_s,
                str(market_type or "moneyline"),
                segs["weight_class"],
                segs["odds_bucket"],
                segs["confidence_label"],
                segs["prop_type"],
                correct_i,
                float(stake_f) if stake_f is not None else 0.0,
                float(pnl) if pnl is not None else None,
                float(pnl_pct) if pnl_pct is not None else None,
                float(decimal_odds) if decimal_odds is not None else None,
                float(closing_odds) if closing_odds is not None else None,
                float(clv) if clv is not None else None,
                1 if settlement_complete else 0,
                float(edge) if edge is not None else None,
                source,
            ),
        )
        conn.commit()
    return True


def sync_from_prediction_bank(path: Path | str | None = None) -> int:
    """Ingest settled prediction_bank rows into closed_bets. Returns rows upserted."""
    try:
        from src.prediction_bank import load_bank
    except Exception as exc:
        logger.debug("prediction bank import failed: %s", exc)
        return 0

    df = load_bank(path)
    if df.empty:
        return 0
    settled = df[df["status"].astype(str).str.lower() == "settled"].copy()
    if settled.empty:
        return 0

    added = 0
    for _, row in settled.iterrows():
        pid = str(row.get("prediction_id") or "").strip()
        if not pid:
            continue
        correct_raw = pd.to_numeric(row.get("correct"), errors="coerce")
        if pd.isna(correct_raw):
            continue
        try:
            odds = float(row.get("odds") or 0) if str(row.get("odds") or "").strip() else None
        except (TypeError, ValueError):
            odds = None
        if odds is not None and odds <= 1.0:
            odds = None
        try:
            stake = float(row.get("stake") or 0) if str(row.get("stake") or "").strip() else None
        except (TypeError, ValueError):
            stake = None
        if stake is not None and stake <= 0:
            stake = None
        try:
            pnl = float(row.get("pnl")) if str(row.get("pnl") or "").strip() else None
        except (TypeError, ValueError):
            pnl = None
        try:
            close_odds = (
                float(row.get("closing_odds"))
                if str(row.get("closing_odds") or "").strip()
                else None
            )
        except (TypeError, ValueError):
            close_odds = None
        try:
            clv = float(row.get("clv")) if str(row.get("clv") or "").strip() else None
        except (TypeError, ValueError):
            clv = None
        complete_raw = str(row.get("settlement_complete") or "").strip()
        if complete_raw in ("1", "true", "yes"):
            complete = True
        elif complete_raw in ("0", "false", "no"):
            complete = False
        else:
            from src.settlement import settlement_complete as _complete

            complete = _complete(stake=stake, opening_odds=odds, pnl=pnl)
        try:
            edge_pct = float(row.get("edge_pct") or 0) if str(row.get("edge_pct") or "").strip() else None
            edge = (edge_pct / 100.0) if edge_pct is not None else None
        except (TypeError, ValueError):
            edge = None

        segs = classify_bet_segments(
            weight_class=row.get("weight_class"),
            decimal_odds=odds,
            confidence_label=row.get("confidence") or row.get("confidence_label"),
            prop_type=row.get("prop_type") or "moneyline",
            market_type=str(row.get("market_type") or "moneyline"),
        )
        if record_closed_bet(
            prediction_id=pid,
            settled_at=str(row.get("settled_at") or "") or None,
            profile=str(row.get("profile") or config.UFC_PROFILE),
            market_type=str(row.get("market_type") or "moneyline"),
            weight_class=segs["weight_class"],
            odds_bucket=segs["odds_bucket"],
            confidence_label=segs["confidence_label"],
            prop_type=segs["prop_type"],
            correct=bool(int(correct_raw)),
            stake=stake,
            pnl=pnl,
            decimal_odds=odds,
            closing_odds=close_odds,
            clv=clv,
            settlement_complete=complete,
            edge=edge,
            source="prediction_bank",
        ):
            added += 1
    return added


def _fetch_bets(days: int | None, *, complete_only: bool = False) -> list[sqlite3.Row]:
    with _connect() as conn:
        _init_db(conn)
        if days is None or days <= 0:
            sql = "SELECT * FROM closed_bets"
            params: tuple[Any, ...] = ()
        else:
            cutoff = (
                dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=int(days))
            ).isoformat()
            sql = "SELECT * FROM closed_bets WHERE settled_at >= ?"
            params = (cutoff,)
        if complete_only:
            sql += (" AND" if "WHERE" in sql else " WHERE") + " settlement_complete = 1"
        sql += " ORDER BY settled_at"
        rows = conn.execute(sql, params).fetchall()
    return list(rows)


def _score_from_trades(trades: list[sqlite3.Row]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "return_pct": 0.0,
            "sharpe": 0.0,
            "win_rate_pct": 0.0,
            "pnl_contribution": 0.0,
            "avg_clv": None,
            "risk_adjusted_score": 0.0,
            "rating": "No data",
        }

    pnls = [float(r["pnl"]) for r in trades if r["pnl"] is not None]
    if not pnls:
        return {
            "trade_count": len(trades),
            "return_pct": 0.0,
            "sharpe": 0.0,
            "win_rate_pct": 0.0,
            "pnl_contribution": 0.0,
            "avg_clv": None,
            "risk_adjusted_score": 0.0,
            "rating": "No data",
        }
    pnl_pcts = [float(r["pnl_pct"]) for r in trades if r["pnl_pct"] is not None]
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    notional_sum = sum(float(r["stake"] or 0) for r in trades) or 1.0
    return_pct = 100.0 * total_pnl / notional_sum

    clvs = [float(r["clv"]) for r in trades if "clv" in r.keys() and r["clv"] is not None]
    avg_clv = (sum(clvs) / len(clvs)) if clvs else None

    if len(pnl_pcts) >= 2:
        mean = sum(pnl_pcts) / len(pnl_pcts)
        var = sum((x - mean) ** 2 for x in pnl_pcts) / (len(pnl_pcts) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        sharpe = (mean / std) * math.sqrt(min(len(pnl_pcts), 52)) if std > 1e-9 else 0.0
    else:
        sharpe = 0.0

    win_rate = 100.0 * wins / len(pnls)
    score = 50.0
    score += min(25.0, max(-15.0, return_pct * 2.0))
    score += min(20.0, max(-10.0, sharpe * 8.0))
    score += min(15.0, (win_rate - 50.0) * 0.3)
    if avg_clv is not None:
        score += min(8.0, max(-8.0, avg_clv * 400.0))
    score = max(0.0, min(100.0, score))

    return {
        "trade_count": len(trades),
        "return_pct": round(return_pct, 2),
        "sharpe": round(sharpe, 2),
        "win_rate_pct": round(win_rate, 1),
        "pnl_contribution": round(total_pnl, 2),
        "avg_clv": round(avg_clv, 5) if avg_clv is not None else None,
        "risk_adjusted_score": round(score, 1),
        "rating": rating_from_score(score),
    }


def compute_segment_metrics(rows: list[sqlite3.Row] | None = None, *, days: int | None = None) -> dict[str, dict[str, Any]]:
    """Score every observed segment key across the four dimensions."""
    rows = list(rows) if rows is not None else _fetch_bets(days, complete_only=True)
    out: dict[str, dict[str, Any]] = {}
    for dim in SEGMENT_DIMENSIONS:
        by_value: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            val = slugify_segment(row[dim] if dim in row.keys() else "unknown")
            by_value.setdefault(val, []).append(row)
        for val, trades in by_value.items():
            key = segment_key(dim, val)
            metrics = _score_from_trades(trades)
            out[key] = {
                "segment_key": key,
                "dimension": dim,
                "value": val,
                "label": f"{SEGMENT_LABELS.get(dim, dim)}: {val}",
                **metrics,
            }
    return out


def segment_health(
    *,
    days: int | None = None,
    profile: str | None = None,
    weight_class: str | None = None,
    odds_bucket: str | None = None,
    confidence_label: str | None = None,
    prop_type: str | None = None,
) -> dict[str, Any]:
    """
    Recent segment health for threshold feedback (ROI, hit rate, CLV).

    Fail-closed: only settlement_complete=1 rows count; thin samples → complete=False.
    """
    from src.settlement import health_lookback_days

    lookback = int(days) if days is not None else health_lookback_days(profile)
    min_n = max(1, int(getattr(config, "HEALTH_MIN_SETTLED_BETS", 8) or 8))
    rows = _fetch_bets(lookback, complete_only=True)
    prof = config.normalize_profile(profile or getattr(config, "UFC_PROFILE", "paper"))
    filtered: list[sqlite3.Row] = []
    for r in rows:
        if config.normalize_profile(str(r["profile"] or "")) != prof:
            continue
        if weight_class and slugify_segment(r["weight_class"]) != slugify_segment(weight_class):
            continue
        if odds_bucket and slugify_segment(r["odds_bucket"]) != slugify_segment(odds_bucket):
            continue
        if confidence_label and slugify_segment(r["confidence_label"]) != slugify_segment(
            confidence_label
        ):
            continue
        if prop_type and slugify_segment(r["prop_type"]) != slugify_segment(prop_type):
            continue
        filtered.append(r)

    metrics = _score_from_trades(filtered)
    n = int(metrics.get("trade_count") or 0)
    hit_rate = (metrics["win_rate_pct"] / 100.0) if n else None
    roi = (metrics["return_pct"] / 100.0) if n else None
    complete = n >= min_n
    return {
        "complete": complete,
        "lookback_days": lookback,
        "min_settled": min_n,
        "trade_count": n,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_clv": metrics.get("avg_clv"),
        "return_pct": metrics.get("return_pct"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "pnl_contribution": metrics.get("pnl_contribution"),
        "profile": prof,
        "fail_closed": not complete,
    }


def get_segment_ratings(days: int | None = None) -> dict[str, Any]:
    """Rolling segment metrics snapshot (syncs prediction bank first)."""
    lookback = (
        int(days)
        if days is not None
        else max(7, int(getattr(config, "STRATEGY_RATING_LOOKBACK_DAYS", 365) or 365))
    )
    try:
        sync_from_prediction_bank()
    except Exception as exc:
        logger.debug("strategy_performance bank sync skipped: %s", exc)

    metrics = compute_segment_metrics(days=lookback)
    ranked = sorted(
        [v for v in metrics.values() if int(v.get("trade_count") or 0) > 0],
        key=lambda x: float(x.get("risk_adjusted_score") or 0),
        reverse=True,
    )
    payload = {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "window_days": lookback,
        "segments": metrics,
        "ranked": ranked,
        "top_5": ranked[:5],
        "bottom_5": list(reversed(ranked[-5:])) if len(ranked) >= 5 else list(reversed(ranked)),
    }
    try:
        path = performance_json_path()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("strategy_performance.json write failed: %s", exc)
    return payload


def save_daily_snapshot() -> None:
    today = dt.date.today().isoformat()
    metrics = compute_segment_metrics(days=30)
    with _connect() as conn:
        _init_db(conn)
        for key, payload in metrics.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO segment_snapshots (snap_date, segment_key, metrics_json)
                VALUES (?, ?, ?)
                """,
                (today, key, json.dumps(payload)),
            )
        conn.commit()
