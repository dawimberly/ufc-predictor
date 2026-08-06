"""Dashboard analysis: full refresh, quick odds, watch helpers."""

from __future__ import annotations

import importlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

import config
from src.alerts import generate_alerts
from src.card_cache import card_fingerprint, load_event_cache, predict_card_cached
from src.parlay_builder import threshold_context_for_alerts
from src.predictor import merge_predictions_with_odds

logger = logging.getLogger(__name__)


def _reload_config_flags() -> None:
    """Re-read .env and refresh ENABLE_PROPS / Odds API key before analysis."""
    try:
        from src.project_paths import reload_runtime_env

        reload_runtime_env()
    except Exception as exc:
        logger.warning("Config reload failed: %s", exc)
    try:
        from src.odds_providers.odds_api_client import refresh_odds_api_runtime

        meta = refresh_odds_api_runtime()
        logger.info(
            "Odds API key: loaded=%s len=%s last4=%s source=%s sport=%s regions=%s",
            meta.get("key_loaded"),
            meta.get("key_length"),
            meta.get("key_last4") or "-",
            meta.get("key_source"),
            meta.get("sport"),
            meta.get("regions"),
        )
    except Exception as exc:
        logger.warning("Odds API key reload failed: %s", exc)
    logger.info("ENABLE_PROPS loaded as: %s", config.ENABLE_PROPS)


ProgressFn = Callable[[str, float | None], None]

_ODDS_OVERVIEW_COLS = (
    "f1_odds",
    "f2_odds",
    "edge_f1",
    "edge_f2",
    "edge_pct",
    "implied_prob_f1",
    "implied_prob_f2",
    "odds_matched",
)


def _pick_best_odds_overview(
    base: pd.DataFrame,
    merged_by_book: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Vectorized best-edge overview across book prediction frames."""
    if base.empty or not merged_by_book:
        return base
    frames: list[pd.DataFrame] = []
    for mdf in merged_by_book.values():
        if mdf is None or mdf.empty:
            continue
        if "fighter_1" not in mdf.columns or "fighter_2" not in mdf.columns:
            continue
        keep = [c for c in ("fighter_1", "fighter_2", *_ODDS_OVERVIEW_COLS) if c in mdf.columns]
        frames.append(mdf[keep].copy())
    if not frames:
        return base
    stacked = pd.concat(frames, ignore_index=True)
    if "edge_pct" not in stacked.columns:
        return base
    stacked["_edge_sort"] = pd.to_numeric(stacked["edge_pct"], errors="coerce").fillna(-1e9)
    best = (
        stacked.sort_values("_edge_sort", ascending=False)
        .drop_duplicates(subset=["fighter_1", "fighter_2"], keep="first")
        .drop(columns=["_edge_sort"])
    )
    cols_drop = [c for c in _ODDS_OVERVIEW_COLS if c in base.columns]
    overview = base.drop(columns=cols_drop, errors="ignore")
    return overview.merge(best, on=["fighter_1", "fighter_2"], how="left", suffixes=("", "_book"))

BOOK_LOADERS = {
    "Odds API": ("src.odds_providers.the_odds_api", "fetch_the_odds_api_odds"),
    "BetNow.eu": ("src.odds_providers.betnow_scraper", "fetch_betnow_odds"),
    "DraftKings": ("src.odds_providers.draftkings", "fetch_draftkings_odds"),
    "MyBookie": ("src.odds_providers.mybookie_scraper", "fetch_mybookie_odds"),
    "Consensus": ("src.predictor", "fetch_ufc_odds"),
}


def active_book_loaders() -> dict[str, tuple[str, str]]:
    """Book loaders: Odds API always on; scrapers only when enabled."""
    loaders: dict[str, tuple[str, str]] = {
        "Odds API": BOOK_LOADERS["Odds API"],
    }
    if getattr(config, "BETNOW_ENABLED", False):
        loaders["BetNow.eu"] = BOOK_LOADERS["BetNow.eu"]
    if getattr(config, "DRAFTKINGS_ENABLED", False):
        loaders["DraftKings"] = BOOK_LOADERS["DraftKings"]
    if getattr(config, "MYBOOKIE_ENABLED", False):
        loaders["MyBookie"] = BOOK_LOADERS["MyBookie"]
    return loaders


def _log(progress: ProgressFn | None, msg: str, pct: float | None = None) -> None:
    if progress:
        progress(msg, pct)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _build_arb_scan(
    books: dict[str, Any],
    combined: pd.DataFrame,
    *,
    force_refresh_odds: bool = False,
    budget_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.arb_scanner import scan_cross_book_arbs

    return scan_cross_book_arbs(
        books=books,
        combined=combined,
        force_refresh=force_refresh_odds,
        budget_state=budget_state,
    )


def _build_props_payload(
    predictions: pd.DataFrame,
    book_name: str,
    *,
    force_refresh_odds: bool,
    budget_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank prop singles/parlays from predictions; live odds when available, else synthetic."""
    from src.odds_providers.prop_odds_common import prop_odds_summary
    from src.props import enrich_predictions_with_props, fetch_live_prop_odds, rank_prop_parlays_for_card, rank_prop_singles
    from src.strategy import bankroll_from_budget, strategy_from_profile

    if not config.ENABLE_PROPS:
        logger.debug("Props payload skipped — ENABLE_PROPS is False")
        return {}

    if predictions.empty:
        return {
            "singles": [],
            "singles_meta": {
                "total_found": 0,
                "strict_count": 0,
                "relaxed_count": 0,
                "live_count": 0,
                "synthetic_count": 0,
            },
            "parlays": [],
            "rules": config.BOOK_PROP_RULES.get(book_name, {}),
            "live_prop_lines": {"live": 0, "synthetic": 0},
            "prop_odds_rows": 0,
        }

    prop_odds = fetch_live_prop_odds(book_name, force_refresh=force_refresh_odds)
    merged = enrich_predictions_with_props(predictions, book=book_name, prop_odds=prop_odds)
    bankroll = bankroll_from_budget(budget_state)
    strategy = strategy_from_profile(bankroll=bankroll)
    singles_raw = rank_prop_singles(
        merged,
        book=book_name,
        strategy=strategy,
        prop_odds=prop_odds,
        max_results=config.PROP_MAX_RESULTS,
        include_relaxed=True,
    )
    if isinstance(singles_raw, dict):
        singles = singles_raw.get("items") or []
        singles_meta = singles_raw.get("meta") or {}
    else:
        singles, singles_meta = singles_raw
    prop_warning = ""
    if book_name == "DraftKings":
        try:
            from src.odds_providers import draftkings_props as _dk_props

            prop_warning = str(getattr(_dk_props, "LAST_WARNING", "") or "").strip()
        except Exception:
            prop_warning = ""
    elif book_name == "Odds API":
        try:
            from src.odds_providers import the_odds_api as _toa

            prop_warning = str(getattr(_toa, "LAST_WARNING", "") or "").strip()
        except Exception:
            prop_warning = ""
    return {
        "singles": singles,
        "singles_meta": singles_meta,
        "parlays": (
            rank_prop_parlays_for_card(
                merged,
                book=book_name,
                strategy=strategy,
                prop_odds=prop_odds,
            )
            if config.BOOK_PROP_RULES.get(book_name, {}).get("allow_prop_parlays")
            else []
        ),
        "rules": config.BOOK_PROP_RULES.get(book_name, {}),
        "live_prop_lines": prop_odds_summary(prop_odds),
        "prop_odds_rows": len(prop_odds),
        "warning": prop_warning,
    }


def _load_book_odds(
    book_name: str,
    mod_path: str,
    fn_name: str,
    combined: pd.DataFrame,
    *,
    force_refresh_odds: bool,
    event_label: str,
    bankroll: float | None = None,
    budget_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one book; on failure chain Odds API → other books (BetNow/MyBookie)."""
    warning = ""
    source = book_name
    # Props: never force-refresh on the moneyline path (per-event Odds API calls
    # burn free-tier credits). Soft Update uses cache; Refresh Props forces live.
    props_force = False if book_name == "Odds API" else force_refresh_odds
    props_payload = (
        _build_props_payload(
            combined,
            book_name,
            force_refresh_odds=props_force,
            budget_state=budget_state,
        )
        if config.ENABLE_PROPS
        else {}
    )

    def _alerts_for(merged: pd.DataFrame) -> dict[str, Any]:
        from src.strategy import bankroll_from_budget, budget_aware_alerts

        br = bankroll if bankroll is not None else bankroll_from_budget(budget_state)
        alerts = generate_alerts(
            merged,
            event_name=event_label,
            use_dynamic_thresholds=config.DYNAMIC_THRESHOLDS_ENABLED,
            bankroll=br,
        )
        return budget_aware_alerts(alerts, budget_state, book_name)

    def _fallback_chain(primary_exc: BaseException | None = None) -> dict[str, Any]:
        """Automatic multi-book fallback; fail-closed only if all sources empty."""
        from src.odds_providers.odds_fallback import fetch_best_available_odds

        skip = {book_name, book_name.replace(".eu", "")}
        if book_name == "DraftKings":
            skip.add("draftkings")
        elif book_name == "BetNow.eu":
            skip.add("betnow")
        elif book_name == "MyBookie":
            skip.add("mybookie")
        elif book_name == "Odds API":
            skip.update({"odds_api", "consensus", "the_odds_api"})
        elif book_name == "Consensus":
            skip.update({"odds_api", "consensus"})

        odds_df, meta = fetch_best_available_odds(
            force_refresh=force_refresh_odds,
            skip_sources=skip,
        )
        if odds_df is None or odds_df.empty or meta.get("fail_closed"):
            err = str(primary_exc) if primary_exc else "no lines"
            warn = meta.get("warning") or (
                f"{book_name} unavailable and all odds fallbacks failed ({err})."
            )
            return {
                "predictions": combined.copy(),
                "alerts": {},
                "odds_matched": 0,
                "odds_total": len(combined),
                "error": warn,
                "warning": warn,
                "source": "",
                "props": props_payload,
            }
        merged = merge_predictions_with_odds(
            combined.copy(), odds_df, fetch_if_missing=False
        )
        matched = int(merged.get("odds_matched", pd.Series(False)).sum())
        fb_src = str(meta.get("source") or "fallback")
        warn = str(meta.get("warning") or "")
        if primary_exc:
            warn = (
                f"{book_name} failed ({primary_exc}). "
                f"Using {fb_src} ({matched}/{len(combined)} matched). "
                + (warn or "")
            ).strip()
        return {
            "predictions": merged,
            "alerts": _alerts_for(merged),
            "odds_matched": matched,
            "odds_total": len(combined),
            "source": fb_src,
            "warning": warn,
            "props": {},
        }

    try:
        mod = importlib.import_module(mod_path)
        odds_df = getattr(mod, fn_name)(force_refresh=force_refresh_odds)
        merged = merge_predictions_with_odds(combined.copy(), odds_df, fetch_if_missing=False)
        matched = int(merged.get("odds_matched", pd.Series(False)).sum())
        # ODDS_FETCH_ONCE can lock a prior-card cache (e.g. Belgrade vs next week's slate).
        # If match rate is poor, clear once-caches and pull live odds one time.
        if book_name == "Odds API" and len(combined) > 0:
            min_ok = max(3, (len(combined) + 1) // 2)
            if matched < min_ok:
                from src.odds_providers.odds_api_client import clear_odds_api_fetch_once_caches

                cleared = clear_odds_api_fetch_once_caches()
                if cleared or matched == 0:
                    logger.warning(
                        "Odds API matched only %s/%s (need >=%s) — cleared fetch-once cache "
                        "(%s files) and retrying live pull once",
                        matched,
                        len(combined),
                        min_ok,
                        len(cleared),
                    )
                    odds_df = getattr(mod, fn_name)(force_refresh=True)
                    merged = merge_predictions_with_odds(
                        combined.copy(), odds_df, fetch_if_missing=False
                    )
                    matched = int(merged.get("odds_matched", pd.Series(False)).sum())
        if book_name == "BetNow.eu" and matched == 0:
            raise ValueError("BetNow scraper returned no matched fights")
        if book_name == "DraftKings":
            dk_warn = str(getattr(mod, "LAST_WARNING", "") or "").strip()
            if dk_warn:
                warning = dk_warn
                if "consensus" in dk_warn.lower():
                    source = "Consensus (DraftKings fallback)"
                elif "betnow" in dk_warn.lower() or "mybookie" in dk_warn.lower():
                    source = dk_warn.split("Using ")[-1].split(" odds")[0] if "Using " in dk_warn else "Book fallback"
                elif "cached" in dk_warn.lower():
                    source = "DraftKings (cached)"
        if book_name == "Odds API":
            from src.predictor import LAST_ODDS_MATCH_META
            import src.odds_providers.odds_api_client as _odds_client

            api_warn = str(getattr(mod, "LAST_WARNING", "") or getattr(mod, "LAST_ERROR", "") or "").strip()
            match_meta = dict(LAST_ODDS_MATCH_META)
            api_events = int(match_meta.get("api_events") or (0 if odds_df is None else len(odds_df)))
            rem = _odds_client.LAST_REQUEST_META.get("requests_remaining")
            last4 = _odds_client.LAST_REQUEST_META.get("key_last4") or "-"
            src_path = _odds_client.LAST_REQUEST_META.get("key_source") or ""
            logger.info(
                "Odds API merge: key_source=%s len=%s last4=%s remaining=%s "
                "api_events=%s card=%s matched=%s reason=%s",
                src_path,
                _odds_client.LAST_REQUEST_META.get("key_length"),
                last4,
                rem,
                api_events,
                match_meta.get("card_fights"),
                matched,
                match_meta.get("reason"),
            )
            if api_warn:
                warning = api_warn
                if "quota" in api_warn.lower() or "out_of_usage" in api_warn.lower():
                    source = "the_odds_api (cached)"
            if matched == 0:
                reason = str(match_meta.get("reason") or "")
                unmatched = match_meta.get("unmatched") or []
                sample = match_meta.get("api_sample") or []
                if api_events <= 0 and api_warn:
                    fail_msg = api_warn
                elif api_events <= 0:
                    fail_msg = (
                        "NO BET — no usable odds (fail-closed): Odds API returned no events "
                        f"[last4={last4}, remaining={rem}]"
                    )
                else:
                    # API has lines but they don't map onto the loaded card roster
                    u_txt = "; ".join(f"{a} vs {b}" for a, b in unmatched[:3]) or "n/a"
                    s_txt = "; ".join(f"{a} vs {b}" for a, b in sample[:2]) or "n/a"
                    fail_msg = (
                        f"Odds API: {api_events} events, 0 lines matched to loaded card "
                        f"(reason={reason or 'name_mismatch'}). "
                        f"Unmatched sample: {u_txt}. "
                        f"API has e.g. {s_txt}. "
                        "Refresh Next Two if the card is stale."
                    )
                    logger.warning(fail_msg)
                # Keep merged predictions (odds columns False) so the tab still lists fights
                return {
                    "predictions": merged,
                    "alerts": {},
                    "odds_matched": 0,
                    "odds_total": len(combined),
                    "error": fail_msg if api_events <= 0 else "",
                    "warning": fail_msg,
                    "source": "the_odds_api",
                    "props": props_payload,
                    "odds_match_meta": {
                        **match_meta,
                        "key_last4": last4,
                        "requests_remaining": rem,
                        "key_source": src_path,
                        "status_line": (
                            f"Odds API: {api_events} events, 0 lines matched"
                            if api_events
                            else "Odds API: 0 events"
                        ),
                    },
                }
            # Success path: attach match meta for status banner
            warning = warning or ""
            status_line = f"Odds API: {api_events} events, {matched} lines matched"
            return {
                "predictions": merged,
                "alerts": _alerts_for(merged),
                "odds_matched": matched,
                "odds_total": len(combined),
                "source": source if source != book_name else "the_odds_api",
                "warning": warning,
                "props": {},
                "odds_match_meta": {
                    **match_meta,
                    "key_last4": last4,
                    "requests_remaining": rem,
                    "key_source": src_path,
                    "status_line": status_line,
                },
            }
        if matched == 0 and book_name in ("DraftKings", "Consensus", "MyBookie"):
            # Primary returned empty/unmatched — try full chain
            return _fallback_chain(ValueError(f"{book_name} matched 0 fights"))
    except Exception as exc:
        logger.warning("%s odds failed: %s", book_name, exc)
        if book_name == "Odds API":
            fail_msg = str(exc) or "NO BET — no usable odds (fail-closed)"
            if "NO BET" not in fail_msg:
                fail_msg = f"NO BET — no usable odds (fail-closed): {fail_msg}"
            return {
                "predictions": combined.copy(),
                "alerts": {},
                "odds_matched": 0,
                "odds_total": len(combined),
                "error": fail_msg,
                "warning": fail_msg,
                "source": "the_odds_api",
                "props": props_payload,
            }
        return _fallback_chain(exc)

    return {
        "predictions": merged,
        "alerts": _alerts_for(merged),
        "odds_matched": matched,
        "odds_total": len(combined),
        "source": source,
        "warning": warning,
        "props": {},
    }


def refresh_books_props(
    books: dict[str, dict[str, Any]],
    *,
    force_refresh_odds: bool = True,
    budget_state: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> None:
    """Rebuild prop rankings for each book from existing predictions (no ML re-run)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _reload_config_flags()
    if not config.ENABLE_PROPS:
        return

    # Soft Update / Refresh reuse first Odds API prop download when ODDS_FETCH_ONCE.
    if force_refresh_odds and bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        try:
            from src.odds_providers.the_odds_api import PROP_CACHE_PATH, PROP_FETCH_ONCE_MARKER, _cache_fresh

            if _cache_fresh(PROP_CACHE_PATH) or PROP_FETCH_ONCE_MARKER.is_file():
                force_refresh_odds = False
                logger.info("ODDS_FETCH_ONCE: props refresh will reuse cache (no live Odds API props)")
        except Exception:
            pass

    enabled = config.enabled_books_from_budget(budget_state) if budget_state else None
    targets: list[tuple[str, pd.DataFrame]] = []
    for book_name in ("Odds API", "BetNow.eu", "DraftKings", "MyBookie"):
        if book_name not in books:
            continue
        # Odds API is never gated by Budget Manager book toggles
        if book_name != "Odds API" and enabled is not None and book_name not in enabled:
            continue
        if book_name == "MyBookie" and not config.MYBOOKIE_ENABLED:
            continue
        if book_name == "BetNow.eu" and not getattr(config, "BETNOW_ENABLED", False):
            continue
        if book_name == "DraftKings" and not getattr(config, "DRAFTKINGS_ENABLED", False):
            continue
        # Skip books that already built props in _load_book_odds unless forcing a live pull.
        existing = books[book_name].get("props")
        if (
            not force_refresh_odds
            and isinstance(existing, dict)
            and ("prop_odds_rows" in existing or "singles_meta" in existing)
        ):
            logger.info("Props refresh: %s already loaded — skip duplicate Odds API pull", book_name)
            continue
        preds = books[book_name].get("predictions")
        if not isinstance(preds, pd.DataFrame) or preds.empty:
            continue
        targets.append((book_name, preds))

    if not targets:
        logger.info("Props refresh skipped — nothing to rebuild")
        return

    def _build_one(name: str, preds: pd.DataFrame) -> tuple[str, dict[str, Any]]:
        logger.info("Props refresh: %s (%d fights)", name, len(preds))
        payload = _build_props_payload(
            preds,
            name,
            force_refresh_odds=force_refresh_odds,
            budget_state=budget_state,
        )
        n = len(payload.get("singles") or [])
        logger.info("Props refresh: %s -> %d ranked lines", name, n)
        return name, payload

    _log(progress, f"Props: fetching lines for {len(targets)} book(s)…", 0.72)
    with ThreadPoolExecutor(max_workers=min(3, len(targets))) as pool:
        futs = {
            pool.submit(_build_one, name, preds): name
            for name, preds in targets
        }
        done = 0
        for fut in as_completed(futs):
            name, props = fut.result()
            books[name]["props"] = props
            done += 1
            pct = 0.72 + (0.25 * done / max(len(targets), 1))
            _log(progress, f"Props: {name} ranked ({done}/{len(targets)})", pct)


def run_quick_props_refresh(
    books: dict[str, dict[str, Any]],
    *,
    progress: ProgressFn | None = None,
    budget_state: dict[str, Any] | None = None,
    force_refresh_odds: bool | None = None,
) -> dict[str, Any]:
    """Fast props-only path (~30–90s): uses cached predictions, parallel per book.

    Default is cache-first when ``ODDS_FETCH_ONCE`` (delete prop cache / marker to re-pull).
    """
    _reload_config_flags()
    if not config.ENABLE_PROPS:
        return {"books": books, "props_updated_at": _utc_now(), "skipped": "ENABLE_PROPS=false"}

    if force_refresh_odds is None:
        force_refresh_odds = not bool(getattr(config, "ODDS_FETCH_ONCE", True))

    logger.info("DEBUG: run_quick_props_refresh() → refresh_books_props(force=%s)", force_refresh_odds)
    books = {k: dict(v) for k, v in books.items()}
    refresh_books_props(
        books,
        force_refresh_odds=force_refresh_odds,
        budget_state=budget_state,
        progress=progress,
    )
    _log(progress, "Props refresh complete.", 1.0)
    return {"books": books, "props_updated_at": _utc_now()}


def apply_books_to_predictions(
    combined: pd.DataFrame,
    *,
    force_refresh_odds: bool = False,
    event_label: str = "",
    progress: ProgressFn | None = None,
    books_filter: set[str] | None = None,
    budget_state: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch odds per book and build alert payloads."""
    from src.strategy import bankroll_from_budget, budget_aware_alerts

    if combined is not None and isinstance(combined, pd.DataFrame) and not combined.empty:
        try:
            from src.gym_data import attach_gym_features

            combined = attach_gym_features(combined)
        except Exception:
            pass

    if budget_state is not None:
        # Odds API is free-tier primary and is never gated by budget book toggles.
        enabled = config.enabled_books_from_budget(budget_state) | {"Odds API"}
        if books_filter is None:
            books_filter = enabled
        else:
            books_filter = (set(books_filter) & enabled) | {"Odds API"}

    bankroll = bankroll_from_budget(budget_state)
    books: dict[str, dict[str, Any]] = {}
    merged_by_book: dict[str, pd.DataFrame] = {}
    loaders = [
        (n, v) for n, v in active_book_loaders().items() if books_filter is None or n in books_filter
    ]

    for i, (book_name, (mod_path, fn_name)) in enumerate(loaders):
        pct = 0.55 + (i / max(len(loaders), 1)) * 0.35
        _log(progress, f"Odds: {book_name}…", pct)
        book_data = _load_book_odds(
            book_name,
            mod_path,
            fn_name,
            combined,
            force_refresh_odds=force_refresh_odds,
            event_label=event_label,
            bankroll=bankroll,
            budget_state=budget_state,
        )
        books[book_name] = book_data
        if book_data.get("odds_matched", 0) > 0:
            merged_by_book[book_name] = book_data["predictions"]

    overview = _pick_best_odds_overview(combined, merged_by_book)

    overview_alerts = generate_alerts(
        overview,
        event_name=event_label,
        use_dynamic_thresholds=config.DYNAMIC_THRESHOLDS_ENABLED,
        bankroll=bankroll,
    )
    books["Overview"] = {
        "predictions": overview,
        "alerts": budget_aware_alerts(overview_alerts, budget_state, "Overview"),
        "odds_matched": int(overview.get("odds_matched", pd.Series(False)).sum()),
        "odds_total": len(overview),
    }
    refresh_books_props(
        books,
        force_refresh_odds=force_refresh_odds,
        budget_state=budget_state,
        progress=progress,
    )
    return books


def _ensure_book_props(
    books: dict[str, dict[str, Any]],
    combined: pd.DataFrame,
    *,
    force_refresh_odds: bool,
    budget_state: dict[str, Any] | None = None,
) -> None:
    """Build props payloads when ENABLE_PROPS is on but books lack prop data."""
    if not config.ENABLE_PROPS:
        return
    # Never force a second Odds API props burn when ODDS_FETCH_ONCE and cache exists.
    props_force = force_refresh_odds
    if props_force and bool(getattr(config, "ODDS_FETCH_ONCE", True)):
        try:
            from src.odds_providers.the_odds_api import PROP_CACHE_PATH, PROP_FETCH_ONCE_MARKER, _cache_fresh

            if _cache_fresh(PROP_CACHE_PATH) or PROP_FETCH_ONCE_MARKER.is_file():
                props_force = False
        except Exception:
            pass
    for book_name, entry in books.items():
        if book_name == "Overview":
            continue
        props = entry.get("props") or {}
        # Already attempted (even if 0 singles) — do not re-hit Odds API.
        if isinstance(props, dict) and ("prop_odds_rows" in props or "singles_meta" in props):
            continue
        if props.get("singles") or int((props.get("singles_meta") or {}).get("total_found", 0)) > 0:
            continue
        preds = entry.get("predictions", combined)
        if not isinstance(preds, pd.DataFrame) or preds.empty:
            continue
        entry["props"] = _build_props_payload(
            preds,
            book_name,
            force_refresh_odds=props_force,
            budget_state=budget_state,
        )
        logger.info("Built props for %s (ENABLE_PROPS=True)", book_name)


def run_quick_odds_refresh(
    base_preds: pd.DataFrame,
    *,
    event_label: str = "",
    progress: ProgressFn | None = None,
    budget_state: dict[str, Any] | None = None,
    force_refresh_odds: bool | None = None,
) -> dict[str, Any]:
    """Fast path: Odds API (+ optional scrapers when enabled).

    Cache-first by default to protect free-tier Odds API credits. With
    ``ODDS_FETCH_ONCE`` (default), Soft Update / Quick Odds reuse the first
    moneyline download and do not re-request while the cache file exists.
    """
    _reload_config_flags()
    from src.predictor import _odds_cache_fresh
    from src.strategy import bankroll_from_budget, budget_aware_alerts

    if force_refresh_odds is None:
        # Only hit the network when moneyline cache is missing (or stale if once=off)
        force_refresh_odds = not _odds_cache_fresh()
    elif force_refresh_odds and bool(getattr(config, "ODDS_FETCH_ONCE", True)) and _odds_cache_fresh():
        # Soft Update / Refresh must not burn credits when a download already exists
        force_refresh_odds = False
        logger.info("ODDS_FETCH_ONCE: ignoring force_refresh — reusing cached moneylines")

    books_label = "Odds API"
    quick_filter: set[str] = {"Odds API"}
    if getattr(config, "BETNOW_ENABLED", False):
        books_label += " + BetNow"
        quick_filter.add("BetNow.eu")
    if getattr(config, "DRAFTKINGS_ENABLED", False):
        books_label += " + DraftKings"
        quick_filter.add("DraftKings")
    if config.MYBOOKIE_ENABLED:
        books_label += " + MyBookie"
        quick_filter.add("MyBookie")
    if budget_state is not None:
        # Keep Odds API even when budget toggles exclude scrapers
        enabled = config.enabled_books_from_budget(budget_state)
        quick_filter = {b for b in quick_filter if b == "Odds API" or b in enabled}
    mode = "live" if force_refresh_odds else "cache"
    _log(progress, f"Quick odds refresh ({books_label}, {mode})…", 0.1)
    logger.info(
        "run_quick_odds_refresh force_refresh=%s mode=%s",
        force_refresh_odds,
        mode,
    )
    books = apply_books_to_predictions(
        base_preds,
        force_refresh_odds=force_refresh_odds,
        event_label=event_label,
        progress=progress,
        books_filter=quick_filter,
        budget_state=budget_state,
    )
    merged_by_book = {k: v["predictions"] for k, v in books.items() if k != "Overview" and "predictions" in v}
    overview = _pick_best_odds_overview(base_preds, merged_by_book)
    bankroll = bankroll_from_budget(budget_state)
    overview_alerts = generate_alerts(overview, event_name=event_label, bankroll=bankroll)
    books["Overview"] = {
        "predictions": overview,
        "alerts": budget_aware_alerts(overview_alerts, budget_state, "Overview"),
        "odds_matched": int(overview.get("odds_matched", pd.Series(False)).sum()),
        "odds_total": len(overview),
    }
    threshold_ctx = threshold_context_for_alerts(overview, bankroll=bankroll)
    # Never burn another Odds API quota pull for arb scan
    arb_scan = _build_arb_scan(books, base_preds, force_refresh_odds=False, budget_state=budget_state)
    _log(progress, "Quick odds + props complete.", 1.0)
    return {
        "books": books,
        "threshold_ctx": threshold_ctx,
        "odds_updated_at": _utc_now(),
        "props_updated_at": _utc_now(),
        "arb_scan": arb_scan,
        "odds_fetch_mode": mode,
    }


def _fetch_card_for_analysis(
    event_index: int,
    event_name: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch raw card; fall back to on-disk cache when live scrape fails."""
    from main import fetch_event_card
    from src.data_loader import _card_matches_event_index

    try:
        card = fetch_event_card(event_index, refresh=force_refresh)
        if not card.empty and _card_matches_event_index(card, event_index):
            card = card.copy()
            card["event_name"] = event_name
            logger.info(
                "Loaded %d fights for card %r (index %d, live/cache)",
                len(card),
                event_name,
                event_index,
            )
            return card
        if not card.empty:
            logger.warning(
                "Card index %d mismatch for %r — forcing refresh",
                event_index,
                event_name,
            )
            card = fetch_event_card(event_index, refresh=True)
            if not card.empty:
                card = card.copy()
                card["event_name"] = event_name
                return card
    except Exception as exc:
        logger.warning("Live card fetch failed for %r: %s", event_name, exc)

    cache_path = config.CACHE_DIR / f"upcoming_card_{event_index}.csv"
    if cache_path.is_file():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            if not cached.empty and _card_matches_event_index(cached, event_index):
                cached = cached.copy()
                cached["event_name"] = event_name
                logger.info(
                    "Loaded %d fights for card %r (upcoming_card_%d.csv fallback)",
                    len(cached),
                    event_name,
                    event_index,
                )
                return cached
            if not cached.empty:
                logger.warning(
                    "Ignoring mismatched upcoming_card_%d.csv for %r — not using fallback",
                    event_index,
                    event_name,
                )
        except Exception as exc:
            logger.warning("Upcoming card csv fallback failed for %r: %s", event_name, exc)

    raise SystemExit(f"Could not load card {event_name!r} (index {event_index})")


def _predictions_from_background_card(event_index: int, event_name: str) -> pd.DataFrame | None:
    """Last-resort per-card predictions from background snapshot parquet."""
    from src.background_runner import _paths

    bg_dir, _ = _paths()
    card_path = bg_dir / "cards" / f"{event_index}.parquet"
    if not card_path.is_file():
        return None
    try:
        preds = pd.read_parquet(card_path)
    except Exception as exc:
        logger.warning("Background card parquet read failed for %r: %s", event_name, exc)
        return None
    if preds.empty:
        return None
    logger.info(
        "Loaded %d fights for card %r (background cards/%d.parquet fallback)",
        len(preds),
        event_name,
        event_index,
    )
    return preds


def _snapshot_has_past_event_dates(snap: dict[str, Any]) -> bool:
    """True if any card/combined row has an event_date strictly before today (UTC)."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date()

    def _frame_past(df: Any) -> bool:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        for col in ("event_date", "date"):
            if col not in df.columns:
                continue
            series = pd.to_datetime(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            if any(ts.date() < today for ts in series):
                return True
        return False

    if _frame_past(snap.get("combined")):
        return True
    for card in snap.get("cards") or []:
        if isinstance(card, dict) and _frame_past(card.get("predictions")):
            return True
    return False


def _snapshot_matches_live_events(snap: dict[str, Any]) -> bool:
    """True when every background card still matches a live upcoming event."""
    if _snapshot_has_past_event_dates(snap):
        logger.warning(
            "Background snapshot has past event_date(s) — skip fallback"
        )
        return False

    snap_events = [
        " ".join(str(x or "").strip().lower().split())
        for x in (snap.get("card_events") or [])
        if str(x or "").strip()
    ]
    if not snap_events:
        label = str(snap.get("event_label") or "")
        snap_events = [" ".join(p.strip().lower().split()) for p in label.split("+") if p.strip()]
    if not snap_events:
        # Prefer card event_name fields when manifest omitted card_events.
        for card in snap.get("cards") or []:
            if isinstance(card, dict):
                name = " ".join(str(card.get("event_name") or "").strip().lower().split())
                if name:
                    snap_events.append(name)
    if not snap_events:
        return False
    try:
        from src.data_loader import list_upcoming_events
        from main import _event_label

        live = [
            " ".join(_event_label(e).strip().lower().split())
            for e in list_upcoming_events()[:6]
        ]
    except Exception as exc:
        logger.debug("Live event check for snapshot skipped: %s", exc)
        return True
    if not live:
        return True

    def _name_matches(se: str, ln: str) -> bool:
        return bool(se and ln and (se in ln or ln in se))

    # Require every cached card to still be on the live slate (partial overlap
    # previously kept yesterday's Belgrade card because Aug 8 still matched).
    for se in snap_events:
        if not any(_name_matches(se, ln) for ln in live):
            logger.warning(
                "Background snapshot event %r not in live upcoming %s — skip fallback",
                se,
                live,
            )
            return False
    return True


def _background_analysis_fallback(*, profile: str = "paper") -> dict[str, Any] | None:
    """Use background snapshot when live refresh fails or returns no fights."""
    from src.background_runner import load_background_snapshot

    # Never use unbounded "any age" fallback — that reloads Freedom 250 forever.
    for label, hours in (("fresh", 24), ("stale", 72)):
        snap = load_background_snapshot(max_age_hours=hours)
        combined = snap.get("combined") if snap else None
        if not snap or not isinstance(combined, pd.DataFrame) or combined.empty:
            continue
        if not _snapshot_matches_live_events(snap):
            continue
        logger.warning(
            "Using %s background cache fallback (%d fights)",
            label,
            len(combined),
        )
        out = dict(snap)
        out["profile"] = profile
        out["errors"] = list(snap.get("errors") or []) + [
            f"Live load unavailable — showing {label} cached fights."
        ]
        return out
    return None


def _fighter_pair_key(row: pd.Series) -> tuple[str, str] | None:
    f1 = row.get("fighter_1") or row.get("fighter1")
    f2 = row.get("fighter_2") or row.get("fighter2")
    if pd.isna(f1) or pd.isna(f2):
        return None
    a, b = str(f1).strip().lower(), str(f2).strip().lower()
    if not a or not b:
        return None
    return tuple(sorted((a, b)))


def _combine_card_predictions(cards: list[dict[str, Any]]) -> pd.DataFrame:
    """Merge per-card predictions; drop cross-card duplicate fight_ids and fighter pairs."""
    frames: list[pd.DataFrame] = []
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for idx, card in enumerate(cards):
        preds = card.get("predictions")
        ev = str(card.get("event_name") or f"Card {idx}")
        if not isinstance(preds, pd.DataFrame) or preds.empty:
            logger.info("Loaded Card %d: %s - 0 fights", idx, ev)
            continue
        chunk = preds.copy()
        chunk["event_name"] = ev
        keep = pd.Series(True, index=chunk.index)
        if config.FIGHT_ID_COLUMN in chunk.columns:
            ids = chunk[config.FIGHT_ID_COLUMN].astype(str)
            dup_mask = ids.isin(seen_ids)
            if dup_mask.any():
                logger.warning(
                    "Dropped %d duplicate fight_id row(s) already on another card (card %d %r)",
                    int(dup_mask.sum()),
                    idx,
                    ev,
                )
            keep &= ~dup_mask
        pair_keys = chunk.apply(_fighter_pair_key, axis=1)
        pair_dup = pair_keys.notna() & pair_keys.isin(seen_pairs)
        if pair_dup.any():
            logger.warning(
                "Dropped %d duplicate fighter-pair row(s) already on another card (card %d %r)",
                int(pair_dup.sum()),
                idx,
                ev,
            )
        keep &= ~pair_dup
        chunk = chunk.loc[keep]
        if config.FIGHT_ID_COLUMN in chunk.columns:
            seen_ids.update(chunk[config.FIGHT_ID_COLUMN].astype(str))
        for key in pair_keys.loc[keep].dropna():
            seen_pairs.add(key)
        logger.info("Loaded Card %d: %s - %d fights", idx, ev, len(chunk))
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_next_two_cards(
    *,
    explain: bool = True,
    use_cache: bool = True,
    force_refresh_cards: bool = False,
    progress: ProgressFn | None = None,
) -> tuple[list[tuple[int, str]], list[dict[str, Any]], pd.DataFrame, bool]:
    """
    Resolve and predict the two soonest upcoming cards (Refresh Next Two path).

    Logs target resolution, raw card size, and prediction row counts for debugging.
    """
    from main import load_or_refresh_data
    from src.data_loader import clear_stale_upcoming_card_caches
    from src.predictor import resolve_analysis_targets

    _log(progress, "Resolving next two events…", 0.02)
    # Manual Refresh Next Two always re-discovers UFC.com targets and drops stale card CSVs.
    cleared = clear_stale_upcoming_card_caches(
        max_age_hours=24.0,
        force=force_refresh_cards,
    )
    if cleared:
        logger.info("load_next_two_cards: cleared card cache files: %s", cleared)

    try:
        targets = resolve_analysis_targets(
            None,
            next_two=True,
            include_adjacent_week=False,
            force_refresh=True,
        )
    except SystemExit as exc:
        logger.warning("resolve_analysis_targets (next two) failed: %s", exc)
        fallback = _background_analysis_fallback()
        if fallback:
            cards = fallback.get("cards") or []
            combined = fallback.get("combined", pd.DataFrame())
            labels = [c.get("event_name", "") for c in cards]
            logger.info("Loading Next Two Cards: %s (background fallback)", labels)
            return (
                [(i, n) for i, n in enumerate(labels)],
                cards,
                combined,
                True,
            )
        raise

    labels = [name for _, name in targets]
    logger.info("Loading Next Two Cards: %s", labels)
    logger.info(
        "load_next_two_cards: resolve_analysis_targets returned %d event(s): %s",
        len(targets),
        labels,
    )
    if not targets:
        logger.warning("load_next_two_cards: no events resolved")
        fallback = _background_analysis_fallback()
        if fallback:
            cards = fallback.get("cards") or []
            combined = fallback.get("combined", pd.DataFrame())
            return ([], cards, combined, True)
        return [], [], pd.DataFrame(), False

    fights = load_or_refresh_data(refresh=False)
    logger.info("load_next_two_cards: historical fight pool rows=%d", len(fights))

    from src.data_loader import card_content_fingerprint

    cards: list[dict[str, Any]] = []
    n = len(targets)
    all_cached = bool(use_cache)
    seen_card_fps: set[str] = set()
    # Always re-fetch fight cards on this path so index 0/1 cannot stay on Freedom 250.
    do_force_cards = True

    for idx, (event_index, event_name) in enumerate(targets):
        base_pct = 0.05 + (idx / n) * 0.5
        span = 0.5 / n
        _log(progress, f"Card {idx + 1}/{n}: {event_name}", base_pct)
        try:
            card = _fetch_card_for_analysis(
                event_index,
                event_name,
                force_refresh=do_force_cards,
            )
            card_fp = card_content_fingerprint(card)
            if card_fp and card_fp in seen_card_fps:
                logger.warning(
                    "load_next_two_cards: card %d %r duplicates prior card — force refresh",
                    idx + 1,
                    event_name,
                )
                card = _fetch_card_for_analysis(
                    event_index,
                    event_name,
                    force_refresh=True,
                )
                card_fp = card_content_fingerprint(card)
            if card_fp and card_fp in seen_card_fps:
                raise SystemExit(
                    f"Card {event_name!r} has the same fights as another upcoming card. "
                    f"Delete stale cache: upcoming_card_{event_index}.csv"
                )
            if card_fp:
                seen_card_fps.add(card_fp)
        except SystemExit as exc:
            logger.warning("Card %r unavailable: %s — trying caches", event_name, exc)
            preds = _predictions_from_background_card(event_index, event_name)
            if preds is not None:
                cards.append({"event_name": event_name, "predictions": preds})
                all_cached = True
                continue
            raise
        raw_rows = len(card)
        logger.info(
            "load_next_two_cards: card %d %r raw bouts=%d",
            idx + 1,
            event_name,
            raw_rows,
        )
        if use_cache:
            hit = load_event_cache(event_name, card)
            if not hit or hit["meta"].get("explain") != explain:
                all_cached = False
        else:
            all_cached = False
        preds = predict_card_cached(
            card,
            fights,
            event_name,
            explain=explain,
            use_cache=use_cache,
            progress=progress,
            step_pct=base_pct,
            step_span=span,
        )
        logger.info("Loaded %d fights for card %r (predictions from %d bouts)", len(preds), event_name, raw_rows)
        cards.append({"event_index": event_index, "event_name": event_name, "predictions": preds})

    combined = _combine_card_predictions(cards)
    logger.info(
        "load_next_two_cards: done — %d total predictions across %d card(s)",
        len(combined),
        len(cards),
    )
    return targets, cards, combined, all_cached


def run_full_analysis(
    *,
    event_mode: str,
    profile: str,
    force_refresh_odds: bool = False,
    explain: bool = True,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
    budget_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full dashboard analysis with cached feature engineering."""
    _reload_config_flags()
    from main import fetch_event_card, resolve_event_targets, load_or_refresh_data, _model_exists
    from src.risk_manager import assess_upcoming_card_risk
    from src.strategy import bankroll_from_budget

    result: dict[str, Any] = {
        "generated_at": _utc_now(),
        "event_label": "",
        "profile": profile,
        "cards": [],
        "combined": pd.DataFrame(),
        "books": {},
        "risk_metrics": {},
        "threshold_ctx": {},
        "errors": [],
    }
    config.UFC_PROFILE = profile
    config.apply_profile_overrides()

    if not _model_exists():
        result["errors"].append("No trained model in models/.")
        return result

    next_two = event_mode in ("Next Two Cards", "Last Two Cards")
    event_query = None if event_mode in ("Next Card", "Next Two Cards", "Last Two Cards") else event_mode

    if event_mode == "Next Two Cards":
        logger.info("DEBUG: run_full_analysis(Next Two Cards) → load_next_two_cards()")
        try:
            targets, cards, combined, all_cached = load_next_two_cards(
                explain=explain,
                use_cache=use_cache,
                force_refresh_cards=True,
                progress=progress,
            )
            logger.info(
                "Loading Next Two Cards: %s",
                [name for _, name in targets] or [c.get("event_name") for c in cards],
            )
        except SystemExit as exc:
            result["errors"].append(str(exc) or "Could not resolve events.")
            fallback = _background_analysis_fallback(profile=profile)
            if fallback:
                for key in (
                    "cards",
                    "combined",
                    "books",
                    "risk_metrics",
                    "threshold_ctx",
                    "event_label",
                    "generated_at",
                    "odds_updated_at",
                    "from_cache",
                ):
                    if key in fallback:
                        result[key] = fallback[key]
                result["errors"].extend(fallback.get("errors") or [])
                return result
            return result
        result["event_label"] = " + ".join(name for _, name in targets) if targets else (
            " + ".join(c.get("event_name", "") for c in cards if c.get("event_name"))
        )
        result["cards"] = cards
        result["combined"] = combined
        result["from_cache"] = all_cached
        if combined.empty:
            fallback = _background_analysis_fallback(profile=profile)
            fb_combined = fallback.get("combined") if isinstance(fallback, dict) else None
            if isinstance(fb_combined, pd.DataFrame) and not fb_combined.empty:
                for key in ("cards", "combined", "event_label", "from_cache"):
                    if key in fallback:
                        result[key] = fallback[key]
                result["errors"].append("Live card load returned no fights — using cached snapshot.")
    else:
        _log(progress, "Resolving events…", 0.02)
        try:
            targets = resolve_event_targets(
                event_query,
                next_two=next_two,
                include_adjacent_week=not next_two and event_query is not None,
            )
        except SystemExit as exc:
            result["errors"].append(str(exc) or "Could not resolve events.")
            return result

        result["event_label"] = " + ".join(name for _, name in targets)
        fights = load_or_refresh_data(refresh=False)
        n = len(targets)
        all_cached = bool(use_cache)

        for idx, (event_index, event_name) in enumerate(targets):
            base_pct = 0.05 + (idx / n) * 0.5
            span = 0.5 / n
            _log(progress, f"Card {idx + 1}/{n}: {event_name}", base_pct)
            card = fetch_event_card(event_index, refresh=False)
            if use_cache:
                hit = load_event_cache(event_name, card)
                if not hit or hit["meta"].get("explain") != explain:
                    all_cached = False
            else:
                all_cached = False
            preds = predict_card_cached(
                card,
                fights,
                event_name,
                explain=explain,
                use_cache=use_cache,
                progress=progress,
                step_pct=base_pct,
                step_span=span,
            )
            result["cards"].append({"event_index": event_index, "event_name": event_name, "predictions": preds})

        combined = _combine_card_predictions(result["cards"])
        result["combined"] = combined
        result["from_cache"] = all_cached

    bankroll = bankroll_from_budget(budget_state)
    if budget_state is not None:
        config.apply_budget_state(budget_state)

    _log(progress, "Loading odds…", 0.58)
    result["books"] = apply_books_to_predictions(
        combined,
        force_refresh_odds=force_refresh_odds,
        event_label=result["event_label"],
        progress=progress,
        budget_state=budget_state,
    )
    _ensure_book_props(
        result["books"],
        combined,
        force_refresh_odds=force_refresh_odds,
        budget_state=budget_state,
    )
    try:
        result["arb_scan"] = _build_arb_scan(
            result["books"],
            combined,
            force_refresh_odds=force_refresh_odds,
            budget_state=budget_state,
        )
    except Exception as exc:
        logger.warning("Arb scan failed: %s", exc)
        result["arb_scan"] = {"moneyline": [], "props": [], "meta": {}, "errors": [str(exc)]}

    _log(progress, "Risk analysis…", 0.92)
    try:
        dk = result["books"].get("DraftKings", {}).get("predictions", combined)
        result["risk_metrics"] = assess_upcoming_card_risk(
            dk,
            bankroll=bankroll,
            simulations=min(config.MC_CARD_SIMULATIONS, 3000),
        )
    except Exception as exc:
        result["risk_metrics"] = {"available": False, "reason": str(exc)}
        result["errors"].append(f"Risk: {exc}")

    overview = result["books"].get("Overview", {}).get("predictions", combined)
    result["threshold_ctx"] = threshold_context_for_alerts(
        overview,
        bankroll=bankroll,
    )
    if config.ENABLE_PROPS:
        try:
            from src.backtester import load_backtest_summary

            prop_bt = load_backtest_summary()
            if prop_bt:
                result["prop_backtest"] = {
                    k.replace("prop_", ""): v
                    for k, v in prop_bt.items()
                    if k.startswith("prop_") or k.startswith("prop_acc_") or k.startswith("mixed_parlay_")
                }
        except Exception:
            result["prop_backtest"] = {}

    # Auto-log card predictions + settle any completed fights into the bank.
    if bool(getattr(config, "PREDICTION_BANK_AUTO_LOG", True)):
        try:
            from src.prediction_bank import bank_from_dashboard_payload, settle_open_predictions

            logged = bank_from_dashboard_payload(result)
            settled = settle_open_predictions()
            result["prediction_bank"] = {"logged": logged, **settled}
            if logged or settled.get("settled"):
                logger.info("Prediction bank update: %s", result["prediction_bank"])
        except Exception as exc:
            logger.warning("Prediction bank update failed: %s", exc)

    _log(progress, "Complete.", 1.0)
    if result["combined"].empty and not result.get("books"):
        fallback = _background_analysis_fallback(profile=profile)
        if fallback:
            for key in (
                "cards",
                "combined",
                "books",
                "risk_metrics",
                "threshold_ctx",
                "event_label",
                "generated_at",
                "odds_updated_at",
                "from_cache",
            ):
                if key in fallback:
                    result[key] = fallback[key]
            result["errors"].append("Analysis produced no fights — loaded background cache.")
    return result


def detect_card_change(event_index: int = 0) -> tuple[bool, str, list[str]]:
    from main import fetch_event_card

    card = fetch_event_card(event_index, refresh=True)
    fp = card_fingerprint(card)
    event_name = fp.get("event_name") or "Upcoming"
    cached = load_event_cache(event_name, card)
    if cached is None:
        return True, event_name, fp["fight_ids"]
    changed = cached["meta"].get("fingerprint") != fp
    return changed, event_name, fp["fight_ids"]
