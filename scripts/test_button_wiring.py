#!/usr/bin/env python3
"""Verify dashboard toolbar buttons map to the expected backend entry points."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check(name: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "FAIL"
    line = f"  [{mark}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        raise SystemExit(1)


def main() -> int:
    print("=== Dashboard button wiring ===\n")

    import src.ufc_dashboard as dash
    from src import dashboard_service

    app_cls = dash.UFCDashboardApp
    handlers = {
        "Refresh Next Two": ("_on_refresh", "run_dashboard_analysis / load_next_two_cards"),
        "Capture Cookies / Login": (
            "_on_refresh_capture_cookies",
            "ensure_cookies_before_refresh + run_dashboard_analysis",
        ),
        "Quick Odds + Props": ("_on_quick_odds", "run_quick_odds_refresh"),
        "Refresh Props": ("_on_refresh_props", "run_quick_props_refresh"),
        "Process New Card": ("_on_process_new_card", "detect_card_change / run_dashboard_analysis"),
        "Grok Analysis": ("_on_grok_analysis", "analyze_card_with_grok"),
    }

    for label, (method, backend) in handlers.items():
        _check(
            label,
            hasattr(app_cls, method) and callable(getattr(app_cls, method)),
            f"handler {method}() -> {backend}",
        )

    src_refresh = inspect.getsource(app_cls._on_refresh)
    _check(
        "Refresh Next Two delegates to _run_refresh_next_two",
        "_run_refresh_next_two" in src_refresh,
    )
    src_refresh_worker = inspect.getsource(app_cls._run_refresh_next_two)
    _check(
        "Refresh Next Two calls run_dashboard_analysis",
        'event_mode="Next Two Cards"' in src_refresh_worker
        and "run_dashboard_analysis" in src_refresh_worker,
    )
    _check(
        "Refresh Next Two documents load_next_two_cards chain",
        "load_next_two_cards" in src_refresh_worker,
    )
    _check(
        "Refresh runs cookie capture before odds",
        "_run_cookie_capture" in src_refresh_worker,
    )
    src_capture = inspect.getsource(app_cls._on_refresh_capture_cookies)
    _check(
        "Capture Cookies button forces cookie login",
        "force_cookie_capture=True" in src_capture,
    )
    src_on_refresh = inspect.getsource(app_cls._on_refresh)
    _check(
        "Refresh respects Login on Refresh switch",
        "login_on_refresh_var" in src_on_refresh,
    )

    src_quick = inspect.getsource(app_cls._on_quick_odds)
    _check("Quick Odds uses run_quick_odds_refresh", "run_quick_odds_refresh" in inspect.getsource(app_cls._run_quick_odds_async))

    src_props = inspect.getsource(app_cls._on_refresh_props)
    _check("Refresh Props uses run_quick_props_refresh", "run_quick_props_refresh" in inspect.getsource(app_cls._run_quick_props_async))

    src_grok = inspect.getsource(app_cls._on_grok_analysis)
    _check("Grok uses analyze_card_with_grok", "analyze_card_with_grok" in inspect.getsource(app_cls._run_grok_analysis_async))

    src_apply = inspect.getsource(app_cls._apply_payload)
    _check(
        "_apply_payload renders tabs",
        "_schedule_render_all_tabs" in src_apply and "_render_overview_section" in inspect.getsource(app_cls._schedule_render_all_tabs),
    )

    src_tabs = inspect.getsource(app_cls._build_tabs)
    _check(
        "Tab bar uses CTkTabview command hook",
        "tabs.configure(command=self._on_tab_changed)" in src_tabs,
        "must not override _segmented_button.command",
    )
    _check(
        "Tab handler exists",
        hasattr(app_cls, "_on_tab_changed") and hasattr(app_cls, "_handle_tab_selected"),
    )

    src_full = inspect.getsource(dashboard_service.run_full_analysis)
    _check(
        "run_full_analysis routes Next Two Cards to load_next_two_cards",
        'event_mode == "Next Two Cards"' in src_full and "load_next_two_cards" in src_full,
    )

    print("\n=== Backend smoke (cached, no GUI) ===\n")
    from src.dashboard_service import load_next_two_cards, run_quick_props_refresh, run_quick_odds_refresh

    _, cards, combined, _ = load_next_two_cards(explain=False, use_cache=True)
    _check("load_next_two_cards", len(cards) >= 1 and not combined.empty, f"{len(combined)} fights")

    books = run_quick_odds_refresh(combined.head(5), event_label="smoke-test", budget_state=None)
    _check("run_quick_odds_refresh", bool(books.get("books")), list(books.get("books", {}).keys()))

    props = run_quick_props_refresh(books["books"], budget_state=None)
    _check("run_quick_props_refresh", "books" in props)

    print("\nAll button wiring checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
