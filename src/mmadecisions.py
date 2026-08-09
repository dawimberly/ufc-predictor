"""mmadecisions.com client — per-judge per-round scorecards (research / display).

Terminology: stores judge scoring data only. UI may label extreme-tail judges
as \"controversial\" — never use corrupt/controversial in field names.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

UA = "UFC-Predictor/research (judge scorecards; local use)"
CACHE_DIR = config.CACHE_DIR / "mmadecisions"
DECISIONS_JSONL = CACHE_DIR / "decisions.jsonl"


def _get(url: str, timeout: float = 30.0) -> str:
    req = Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_decision_html(html: str, *, decision_id: int | None = None) -> dict[str, Any] | None:
    """Parse a decision page into structured per-judge round scores."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "") or ""
    event = ""
    # Titles look like: "Alex Pereira def. Jan Blachowicz :: UFC 291 :: MMA Decisions"
    tm = re.search(r"::\s*(UFC[^:]+?)\s*::", title, re.I)
    if tm:
        event = tm.group(1).strip()
    if not event:
        event_el = soup.select_one("td.event2, .event2")
        if event_el:
            txt = event_el.get_text(" ", strip=True)
            if re.search(r"\bUFC\b", txt, re.I):
                event = txt
    if not event:
        m = re.search(r"(UFC(?:\s+Fight Night)?[^<\n|]{0,60})", html, re.I)
        if m:
            event = re.sub(r"\s+", " ", m.group(1)).strip()

    is_ufc = bool(re.search(r"\bUFC\b", title + " " + event, re.I))
    judges: list[dict[str, Any]] = []
    for judge_td in soup.select("td.judge"):
        link = judge_td.find("a", href=re.compile(r"judge/\d+"))
        if not link:
            continue
        judge_name = link.get_text(" ", strip=True).replace("\xa0", " ")
        href = str(link.get("href") or "")
        jm = re.search(r"judge/(\d+)/", href)
        judge_id = int(jm.group(1)) if jm else None
        # Walk up to enclosing table
        table = judge_td.find_parent("table")
        if table is None:
            continue
        rounds: list[dict[str, Any]] = []
        f1_name = ""
        f2_name = ""
        header = table.select_one("tr.top-row td.top-cell")
        headers = table.select("tr.top-row td.top-cell")
        if len(headers) >= 2:
            f1_name = headers[0].get_text(" ", strip=True)
            f2_name = headers[1].get_text(" ", strip=True)
        for tr in table.select("tr.decision"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            try:
                rnd = int(cells[0])
                s1 = int(cells[1])
                s2 = int(cells[2])
            except ValueError:
                continue
            rounds.append({"round": rnd, "score_f1": s1, "score_f2": s2})
        total_f1 = total_f2 = None
        bottom = table.select("tr.bottom-row td, tr.decision-bottom td")
        # Prefer summing rounds
        if rounds:
            total_f1 = sum(r["score_f1"] for r in rounds)
            total_f2 = sum(r["score_f2"] for r in rounds)
        if not rounds:
            continue
        judges.append(
            {
                "judge_id": judge_id,
                "judge_name": judge_name,
                "fighter_1": f1_name,
                "fighter_2": f2_name,
                "rounds": rounds,
                "total_f1": total_f1,
                "total_f2": total_f2,
            }
        )

    if len(judges) < 2:
        return None

    # Winner line
    decision_type = ""
    if re.search(r"split\s+decision", html, re.I):
        decision_type = "split"
    elif re.search(r"majority\s+decision", html, re.I):
        decision_type = "majority"
    elif re.search(r"unanimous\s+decision", html, re.I):
        decision_type = "unanimous"

    f1 = judges[0].get("fighter_1") or ""
    f2 = judges[0].get("fighter_2") or ""
    return {
        "decision_id": decision_id,
        "title": title,
        "event": event,
        "is_ufc": is_ufc,
        "decision_type": decision_type,
        "fighter_1": f1,
        "fighter_2": f2,
        "n_judges": len(judges),
        "n_rounds": len(judges[0]["rounds"]),
        "judges": judges,
        "has_per_round": True,
    }


def fetch_decision(decision_id: int, *, sleep_s: float = 0.35) -> dict[str, Any] | None:
    url = f"https://mmadecisions.com/decision/{decision_id}/fight"
    try:
        html = _get(url)
    except Exception as exc:
        logger.debug("fetch %s failed: %s", decision_id, exc)
        return None
    time.sleep(sleep_s)
    return parse_decision_html(html, decision_id=decision_id)


def append_decision_cache(row: dict[str, Any], path: Path | None = None) -> None:
    path = path or DECISIONS_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_decision_cache(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DECISIONS_JSONL
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def crawl_decision_id_range(
    start_id: int,
    end_id: int,
    *,
    ufc_only: bool = True,
    sleep_s: float = 0.35,
    max_keep: int | None = None,
) -> dict[str, Any]:
    """Crawl inclusive ID range; append UFC (or all) decisions with RBR to cache."""
    kept = 0
    fetched = 0
    ufc = 0
    for did in range(start_id, end_id + 1):
        row = fetch_decision(did, sleep_s=sleep_s)
        fetched += 1
        if not row:
            continue
        if ufc_only and not row.get("is_ufc"):
            continue
        ufc += 1
        append_decision_cache(row)
        kept += 1
        if max_keep is not None and kept >= max_keep:
            break
        if kept % 25 == 0:
            logger.info("mmadecisions crawl kept=%s fetched=%s last_id=%s", kept, fetched, did)
    return {"fetched": fetched, "ufc_kept": kept, "ufc_seen": ufc, "cache": str(DECISIONS_JSONL)}
