"""Fighter external-data coverage report (Sherdog / Wikipedia / CompuBox-style)."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.compubox_stats import compubox_coverage, load_detailed_bout_striking
from src.data_loader import clean_fighter_name, ensure_data_dirs, load_fights
from src.prior_sport import (
    build_prior_sport_cache_from_available,
    prior_sport_coverage,
)
from src.sherdog import (
    load_sherdog_fighters,
    load_sherdog_fights,
    refresh_sherdog_for_names,
    sherdog_coverage,
)
from src.wikipedia_fighters import (
    load_wikipedia_fighters,
    refresh_wikipedia_for_names,
    wikipedia_coverage,
)

logger = logging.getLogger(__name__)


def _unique_fighters(fights: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for col in ("fighter1", "fighter2", "fighter_1", "fighter_2"):
        if col in fights.columns:
            names.extend(fights[col].dropna().astype(str).tolist())
    cleaned = [clean_fighter_name(n) for n in names]
    return sorted({n for n in cleaned if n})


def select_notable_fighters(
    fights: pd.DataFrame | None = None,
    *,
    target: int = 100,
) -> list[str]:
    """
    Prefer gym roster + recent/high-frequency fighters for enrichment samples.
    """
    ensure_data_dirs()
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        clean = clean_fighter_name(name)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            ordered.append(clean)

    # 1) Curated gym / notes roster (notable / active)
    gyms_path = Path(getattr(config, "GYMS_CSV", config.DATA_DIR / "gyms.csv"))
    if gyms_path.is_file():
        try:
            gyms = pd.read_csv(gyms_path)
            col = "fighter_name" if "fighter_name" in gyms.columns else gyms.columns[0]
            for n in gyms[col].dropna().astype(str):
                _add(n)
        except Exception as exc:
            logger.debug("Gym roster load skipped: %s", exc)

    # 2) Upcoming card caches
    for path in sorted(config.CACHE_DIR.glob("upcoming_card*.csv")):
        try:
            card = pd.read_csv(path)
            for col in ("fighter1", "fighter2", "fighter_1", "fighter_2"):
                if col in card.columns:
                    for n in card[col].dropna().astype(str):
                        _add(n)
        except Exception:
            continue

    # 3) Most frequent fighters in recent fights window
    fights = fights if fights is not None else load_fights()
    if not fights.empty:
        work = fights.copy()
        date_col = "date" if "date" in work.columns else config.DATE_COLUMN
        if date_col in work.columns:
            work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
            cutoff = work[date_col].max()
            if pd.notna(cutoff):
                recent = work[work[date_col] >= (cutoff - pd.Timedelta(days=365 * 2))]
            else:
                recent = work
        else:
            recent = work
        counts: Counter[str] = Counter()
        for col in ("fighter1", "fighter2", "fighter_1", "fighter_2"):
            if col in recent.columns:
                for n in recent[col].dropna().astype(str):
                    c = clean_fighter_name(n)
                    if c:
                        counts[c] += 1
        for name, _ in counts.most_common(max(target * 2, 200)):
            _add(name)

    return ordered[: max(50, min(target, len(ordered)))]


def enrich_fighter_sources(
    *,
    max_fetch: int = 100,
    fights: pd.DataFrame | None = None,
    sample: list[str] | None = None,
    wiki_batch_size: int = 5,
    wiki_batch_delay_sec: float = 8.0,
    wiki_per_fighter_delay_sec: float = 5.0,
) -> dict[str, Any]:
    """Enrich Sherdog + Wikipedia for a notable sample. Fail-soft; Wiki never blocks."""
    fights = fights if fights is not None else load_fights()
    if sample is None:
        sample = select_notable_fighters(fights, target=max_fetch)
    else:
        sample = [clean_fighter_name(n) for n in sample if clean_fighter_name(str(n))]
    logger.info("Enriching %s notable fighters (Sherdog + Wikipedia)…", len(sample))
    out: dict[str, Any] = {
        "sample_size": len(sample),
        "sample": sample,
        "sherdog": {},
        "wikipedia": {},
    }
    try:
        out["sherdog"] = refresh_sherdog_for_names(sample, max_fetch=max_fetch)
    except Exception as exc:
        out["sherdog"] = {"error": str(exc), "failure_reasons": {"exception": 1}}
        logger.warning("Sherdog enrichment failed soft: %s", exc)
    try:
        # Slow batched Wiki enrichment — sparse Wiki must not stall the pipeline.
        out["wikipedia"] = refresh_wikipedia_for_names(
            sample,
            max_fetch=max_fetch,
            batch_size=wiki_batch_size,
            batch_delay_sec=wiki_batch_delay_sec,
            per_fighter_delay_sec=wiki_per_fighter_delay_sec,
        )
    except Exception as exc:
        out["wikipedia"] = {"error": str(exc), "failure_reasons": {"exception": 1}}
        logger.warning("Wikipedia enrichment failed soft: %s", exc)
    return out


def build_fighter_data_coverage(
    fights: pd.DataFrame | None = None,
    *,
    refresh: bool = False,
    max_fetch: int = 100,
    sample: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize % of fighters with Sherdog / Wiki / CompuBox-style striking data."""
    ensure_data_dirs()
    fights = fights if fights is not None else load_fights()
    fighters = _unique_fighters(fights)

    refresh_stats: dict[str, Any] = {}
    if refresh:
        try:
            refresh_stats = enrich_fighter_sources(
                max_fetch=max_fetch, fights=fights, sample=sample
            )
            if refresh_stats.get("sample"):
                sample = list(refresh_stats["sample"])
        except Exception as exc:
            refresh_stats = {"error": str(exc)}
            logger.warning("Enrichment failed soft: %s", exc)

    sh = sherdog_coverage(fighters)
    wiki = wikipedia_coverage(fighters)
    # Also report coverage on the enrichment sample (more actionable)
    if sample is None:
        sample = select_notable_fighters(fights, target=max(100, max_fetch))
    else:
        sample = [clean_fighter_name(n) for n in sample if clean_fighter_name(str(n))]
    sh_sample = sherdog_coverage(sample)
    wiki_sample = wikipedia_coverage(sample)
    cb = compubox_coverage(fighters)
    try:
        build_prior_sport_cache_from_available(persist=True)
    except Exception as exc:
        logger.debug("Prior-sport cache rebuild skipped: %s", exc)
    prior = prior_sport_coverage(fighters)
    bout = load_detailed_bout_striking()
    cb_source = (
        float((bout["source"] == "compubox").mean())
        if not bout.empty and "source" in bout.columns
        else 0.0
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_fighters": len(fighters),
        "n_fights": int(len(fights)),
        "sample_size": len(sample),
        "sherdog": {
            **sh,
            "cache_fighters": int(len(load_sherdog_fighters())),
            "cache_fights": int(len(load_sherdog_fights())),
            "sample_pct_fighters": sh_sample.get("pct_fighters"),
            "sample_pct_with_history": sh_sample.get("pct_with_history"),
            "sample_n_matched": sh_sample.get("n_matched"),
            "sample_n_with_history": sh_sample.get("n_with_history"),
        },
        "wikipedia": {
            **wiki,
            "cache_fighters": int(len(load_wikipedia_fighters())),
            "sample_pct_fighters": wiki_sample.get("pct_fighters"),
            "sample_pct_with_bio_fields": wiki_sample.get("pct_with_bio_fields"),
            "sample_pct_height": wiki_sample.get("pct_height"),
            "sample_pct_reach": wiki_sample.get("pct_reach"),
            "sample_pct_stance": wiki_sample.get("pct_stance"),
            "sample_pct_team": wiki_sample.get("pct_team"),
            "sample_n_matched": wiki_sample.get("n_matched"),
            "sample_n_with_bio_fields": wiki_sample.get("n_with_bio_fields"),
        },
        "compubox_style": {
            **cb,
            "bout_rows": int(len(bout)),
            "real_compubox_share": cb_source,
            "note": "Prefers data/cache/compubox_striking.csv; else Greco/UFCStats detail",
        },
        "prior_sport": {
            **prior,
            "note": "Tiers from Wiki/Sherdog/gym notes; unknown→0",
        },
        "refresh": refresh_stats,
    }


def _pct(x: Any) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "n/a"


def save_fighter_data_coverage_report(
    report: dict[str, Any] | None = None,
    *,
    refresh: bool = False,
    max_fetch: int = 100,
) -> Path:
    """Write HTML + JSON coverage report under reports/."""
    report = report or build_fighter_data_coverage(refresh=refresh, max_fetch=max_fetch)
    reports_dir = config.ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    html_path = reports_dir / f"fighter_data_coverage_{stamp}.html"
    json_path = reports_dir / f"fighter_data_coverage_{stamp}.json"

    sh = report.get("sherdog") or {}
    wiki = report.get("wikipedia") or {}
    cb = report.get("compubox_style") or {}
    prior = report.get("prior_sport") or {}
    refresh = report.get("refresh") or {}
    sh_ref = refresh.get("sherdog") or {}
    wiki_ref = refresh.get("wikipedia") or {}

    def _reasons_html(block: dict[str, Any]) -> str:
        reasons = block.get("failure_reasons") or {}
        if not reasons:
            return "<em>none</em>"
        items = "".join(
            f"<li><code>{k}</code>: {v}</li>"
            for k, v in sorted(reasons.items(), key=lambda kv: -int(kv[1]))
        )
        return f"<ul>{items}</ul>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Fighter data coverage</title>
<style>
body {{ font-family: Segoe UI, sans-serif; margin: 2rem; background: #0f1419; color: #e7ecf1; }}
h1,h2 {{ font-weight: 600; }}
.card {{ display:inline-block; min-width: 160px; margin: 0.5rem 1rem 0.5rem 0;
  padding: 1rem 1.2rem; background: #1a2332; border-radius: 10px; }}
.l {{ opacity: 0.7; font-size: 0.85rem; }} .v {{ font-size: 1.35rem; margin-top: 0.25rem; }}
table {{ border-collapse: collapse; margin-top: 1.5rem; width: 100%; max-width: 900px; }}
td, th {{ border-bottom: 1px solid #2a3545; padding: 0.55rem 0.7rem; text-align: left; }}
.note {{ opacity: 0.75; margin-top: 1.5rem; max-width: 900px; line-height: 1.45; }}
code {{ background:#243044; padding:0.1rem 0.35rem; border-radius:4px; }}
</style></head><body>
<h1>Fighter data coverage</h1>
<p class="l">Generated {report.get('generated_at')} · {report.get('n_fighters')} unique fighters ·
{report.get('n_fights')} fights · enrichment sample {report.get('sample_size')}</p>
<div class="card"><div class="l">Sherdog (all)</div><div class="v">{_pct(sh.get('pct_fighters'))}</div>
<div class="l">history {_pct(sh.get('pct_with_history'))}</div></div>
<div class="card"><div class="l">Wikipedia (all)</div><div class="v">{_pct(wiki.get('pct_fighters'))}</div>
<div class="l">bio fields {_pct(wiki.get('pct_with_bio_fields'))}</div></div>
<div class="card"><div class="l">CompuBox-style</div><div class="v">{_pct(cb.get('pct_fighters'))}</div></div>
<div class="card"><div class="l">Prior-sport tier</div><div class="v">{_pct(prior.get('pct_known'))}</div></div>

<h2>Notable sample coverage</h2>
<table>
<tr><th>Source</th><th>Matched</th><th>Useful fields</th><th>Notes</th></tr>
<tr><td>Sherdog</td>
<td>{int(sh.get('sample_n_matched') or 0)} / {int(report.get('sample_size') or 0)} ({_pct(sh.get('sample_pct_fighters'))})</td>
<td>history {int(sh.get('sample_n_with_history') or 0)} ({_pct(sh.get('sample_pct_with_history'))})</td>
<td>cache {sh.get('cache_fighters', 0)} profiles · {sh.get('cache_fights', 0)} fights</td></tr>
<tr><td>Wikipedia</td>
<td>{int(wiki.get('sample_n_matched') or 0)} / {int(report.get('sample_size') or 0)} ({_pct(wiki.get('sample_pct_fighters'))})</td>
<td>bio {int(wiki.get('sample_n_with_bio_fields') or 0)} ({_pct(wiki.get('sample_pct_with_bio_fields'))})</td>
<td>height {_pct(wiki.get('sample_pct_height'))} · reach {_pct(wiki.get('sample_pct_reach'))} ·
stance {_pct(wiki.get('sample_pct_stance'))} · team {_pct(wiki.get('sample_pct_team'))}</td></tr>
<tr><td>CompuBox-style</td>
<td>{int(cb.get('n_matched') or 0)} / {int(cb.get('n') or 0)} ({_pct(cb.get('pct_fighters'))})</td>
<td>{cb.get('bout_rows', 0)} bout rows</td>
<td>{cb.get('note', '')}; real CompuBox share {_pct(cb.get('real_compubox_share'))}</td></tr>
</table>

<h2>Enrichment run</h2>
<table>
<tr><th>Source</th><th>Fetched</th><th>Cached</th><th>Failed</th><th>Failure reasons</th></tr>
<tr><td>Sherdog</td>
<td>{sh_ref.get('fetched', '—')}</td><td>{sh_ref.get('cached', '—')}</td><td>{sh_ref.get('failed', '—')}</td>
<td>{_reasons_html(sh_ref)}</td></tr>
<tr><td>Wikipedia</td>
<td>{wiki_ref.get('fetched', '—')}</td><td>{wiki_ref.get('cached', '—')}</td><td>{wiki_ref.get('failed', '—')}</td>
<td>{_reasons_html(wiki_ref)}</td></tr>
</table>

<p class="note">Missing or blocked sources do not stop the pipeline — UFCStats/Greco remain the baseline.
Curate prior-sport at <code>data/cache/prior_sport_profiles.csv</code>.
Optional real CompuBox dump: <code>data/cache/compubox_striking.csv</code>.
Feature schema v{getattr(config, 'FEATURE_SCHEMA_VERSION', '?')}.</p>
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    import json

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Fighter data coverage → %s", html_path)
    return html_path
