#!/usr/bin/env python3
"""
UFC Predictor Desktop Dashboard - customtkinter GUI with multi-book tabs.

Launch:
    python src/ufc_dashboard.py
    python src/ufc_dashboard.py --debug
    dist/ufc-dashboard.exe
    dist/ufc-dashboard.exe --debug
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Debug / console (before heavy imports)
# ---------------------------------------------------------------------------

_DEBUG_MODE = "--debug" in sys.argv

# ---------------------------------------------------------------------------
# Crash logging — as early as possible (before UI / heavy imports)
# ---------------------------------------------------------------------------

_ENTRY = Path(__file__).resolve()
_ROOT = _ENTRY.parents[1]
_CRASH_LOG: Path | None = None
_HEARTBEAT_LOGGER_READY = False


def _resolve_crash_log_path() -> Path:
    """Prefer project data/logs; fall back next to EXE when frozen."""
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = _ROOT
    return base / "data" / "logs" / "dashboard_crash.log"


def _write_crash_log(title: str, message: str) -> Path | None:
    """Append a crash / fatal record; always safe to call."""
    global _CRASH_LOG
    try:
        path = _CRASH_LOG or _resolve_crash_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        block = f"\n{'=' * 72}\n{stamp} | {title}\n{'-' * 72}\n{message}\n"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(block)
        _CRASH_LOG = path
        return path
    except Exception:
        return None


def _dashboard_heartbeat(msg: str) -> None:
    """Always-on heartbeat to crash log + optional console."""
    line = f"[heartbeat] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    _write_crash_log("HEARTBEAT", msg)
    if _HEARTBEAT_LOGGER_READY:
        try:
            import logging

            logging.getLogger("ufc_dashboard").info(line)
        except Exception:
            pass


def _install_early_fault_handler() -> None:
    """Dump native hard-crashes to dashboard_crash.log when possible."""
    try:
        import faulthandler

        path = _resolve_crash_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Keep file handle open for the process lifetime
        fh = open(path, "a", encoding="utf-8")
        fh.write(
            f"\n{'=' * 72}\n"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | faulthandler enabled\n"
        )
        fh.flush()
        faulthandler.enable(file=fh, all_threads=True)
        global _CRASH_LOG
        _CRASH_LOG = path
    except Exception as exc:
        _write_crash_log("faulthandler_init_failed", str(exc))


_install_early_fault_handler()
_dashboard_heartbeat("dashboard bootstrap start")


def _ascii_ui(text: Any) -> str:
    """Force UI / table strings to plain ASCII for Windows ttk fonts."""
    if text is None:
        return ""
    s = str(text)
    # Common Unicode that often mojibakes on Windows code pages / fonts.
    replacements = {
        "\u2265": ">=",  # >=
        "\u2264": "<=",  # <=
        "\u00b7": " | ",  # middle dot
        "\u2022": "*",  # bullet
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus
        "\u00a0": " ",  # nbsp
        "\ufeff": "",  # BOM
        "\u2026": "...",
        "\u00b1": "+/-",
        "\u2192": "->",
        "\u2190": "<-",
        "\u25bc": "v",  # black down-pointing triangle
        "\u25b2": "^",  # black up-pointing triangle
        "\u25be": "v",
        "\u25b4": "^",
        "\uff04": "$",  # fullwidth dollar
        "\u20ac": "EUR",
        "\u00a3": "GBP",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s.encode("ascii", "replace").decode("ascii")


# Display cap for absurd scraper/merge edges (strategy also uses 0.25 actionable).
_MAX_DISPLAY_EDGE = 0.30


def _has_usable_odds(row: pd.Series | dict[str, Any]) -> bool:
    get = row.get if hasattr(row, "get") else (lambda *_: None)
    if bool(get("odds_matched")):
        return True
    for col in ("f1_odds", "f2_odds"):
        v = get(col)
        try:
            if v is not None and not (isinstance(v, float) and pd.isna(v)) and float(v) > 1.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _format_edge_display(edge: float | None) -> str:
    """Show '—' when missing/suspect; never paint fake -100% edges."""
    if edge is None:
        return "—"
    try:
        e = float(edge)
    except (TypeError, ValueError):
        return "—"
    if not (e == e):  # NaN
        return "—"
    if abs(e) > _MAX_DISPLAY_EDGE:
        return "—"  # absurd scrape/merge — blank, not a real ticket cue
    if e <= -0.50:
        return "—"
    return f"{e * 100:+.1f}%"


def _format_min_bet_plain(stake: Any, *, pct: Any = None) -> str:
    """Min bet as plain dollars (floor $1 when sized); no unicode dots."""
    try:
        amt = float(stake or 0.0)
    except (TypeError, ValueError):
        amt = 0.0
    if amt <= 0:
        return "$0"
    shown = max(1.0, amt) if amt >= 0.5 else amt
    if abs(shown - round(shown)) < 1e-9:
        return f"${int(round(shown))}"
    return f"${shown:.2f}"


def _ascii_row(values: tuple | list) -> tuple:
    return tuple(_ascii_ui(v) for v in values)


def _format_min_bet(stake: Any, *, pct: Any = None) -> str:
    """Stake label: '42% ($12.50)' when pct known, else dollar-only fallback."""
    try:
        from src.strategy import format_stake_pct_dollars

        if isinstance(stake, dict):
            return format_stake_pct_dollars(stake)
        if pct is not None:
            return format_stake_pct_dollars({"stake_pct": pct, "suggested_stake": stake})
    except Exception:
        pass
    try:
        amt = float(stake or 0.0)
    except (TypeError, ValueError):
        amt = 0.0
    if amt <= 0:
        return "0% ($0)"
    if abs(amt - round(amt)) < 1e-9:
        return f"${int(round(amt))}"
    return f"${amt:.2f}"


def _format_ticket_stake(ticket: dict[str, Any]) -> str:
    """Preferred stake display for ranked bets / overview cards."""
    try:
        from src.strategy import format_stake_pct_dollars

        return format_stake_pct_dollars(ticket)
    except Exception:
        return _format_min_bet(ticket.get("suggested_stake"), pct=ticket.get("stake_pct"))


def _debug_log(msg: str) -> None:
    if _DEBUG_MODE:
        print(f"[dashboard] {_ascii_ui(msg)}", flush=True)


def _button_debug(msg: str) -> None:
    """Trace toolbar clicks -> backend (console in --debug; always in dashboard.log)."""
    line = f"DEBUG: {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    _debug_log(line)
    try:
        import logging

        logging.getLogger("ufc_dashboard").info(line)
    except Exception:
        pass


def _configure_expandable_page(page: _CTK_FRAME, content: _CTK_FRAME) -> None:
    """Let a tab page pass remaining height to its main content widget."""
    page.grid_columnconfigure(0, weight=1)
    page.grid_rowconfigure(0, weight=1)
    content.grid(row=0, column=0, sticky="nsew")


def _show_layout_widget(
    widget: _CTK_FRAME | ctk.CTkLabel,
    *,
    show: bool,
    pack_kw: dict[str, Any] | None = None,
    grid_kw: dict[str, Any] | None = None,
) -> None:
    """Show or hide a widget that uses either pack or grid in its parent."""
    if grid_kw is not None:
        if show:
            widget.grid(**grid_kw)
        else:
            widget.grid_remove()
        return
    if show:
        widget.pack(**(pack_kw or {}))
    else:
        widget.pack_forget()


def _enable_debug_console() -> None:
    """Attach a console when running a --windowed EXE with --debug."""
    if not _DEBUG_MODE or not getattr(sys, "frozen", False):
        return
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.AllocConsole()
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
        sys.stderr = open("CONERR$", "w", encoding="utf-8", errors="replace")
        _debug_log("Debug console enabled")
    except Exception:
        pass


def _suppress_console_output() -> None:
    """Windowed EXE: discard stdout/stderr so startup cannot flash a console."""
    if _DEBUG_MODE or not getattr(sys, "frozen", False):
        return
    try:
        from src.safe_io import install_safe_stdout

        install_safe_stdout()
    except Exception:
        pass
    try:
        _null = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _null
        sys.stderr = _null
    except Exception:
        pass


def _show_fatal_error(title: str, message: str) -> None:
    """Last-resort error UI when the dashboard cannot start."""
    _debug_log(f"FATAL: {title}: {message}")
    path = _write_crash_log(title, message)
    try:
        print(f"{title}\n{message}", file=sys.stderr, flush=True)
        if path:
            print(f"(also written to {path})", file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        detail = message if len(message) < 1800 else message[:1800] + "\n..."
        if path:
            detail = f"{detail}\n\nLog: {path}"
        messagebox.showerror(title, detail)
        root.destroy()
    except Exception:
        if not getattr(sys, "frozen", False):
            raise


# --- Bootstrap (EXE-safe) -----------------------------------------------------

_STARTUP_ERROR: str | None = None
_FROZEN = getattr(sys, "frozen", False)

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if _FROZEN and _DEBUG_MODE:
    _enable_debug_console()
elif _FROZEN:
    _suppress_console_output()


def _init_customtkinter():
    """Early CustomTkinter setup - must run before any CTk widgets."""
    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    if _FROZEN:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)
    return ctk


# Early bootstrap + CustomTkinter (subclass needs ctk.CTk at class definition time)
# IMPORTANT: do NOT create/destroy a temporary tk.Tk() probe here — destroying the
# default Tcl interpreter before CustomTkinter starts causes hard ACCESS_VIOLATION
# exits on Windows shortly after mainloop.
_CTK_BASE: type = object
_CTK_FRAME: type = object

try:
    from src.project_paths import bootstrap

    _ROOT = bootstrap(entry_file=_ENTRY, env_log=_debug_log if _DEBUG_MODE else None)
    _debug_log(f"Project root: {_ROOT}")
    ctk = _init_customtkinter()
    _CTK_BASE = ctk.CTk
    _CTK_FRAME = ctk.CTkFrame
    _dashboard_heartbeat("customtkinter ready (no Tk probe)")
except Exception as exc:
    _STARTUP_ERROR = f"Bootstrap / GUI init failed:\n{exc}\n\n{traceback.format_exc()}"
    ctk = None
    _show_fatal_error("UFC Dashboard - bootstrap error", _STARTUP_ERROR)


def _load_dependencies(progress: Callable[[str], None] | None = None) -> None:
    """Import ML + service deps (after CustomTkinter is ready)."""
    global np, pd, matplotlib, FigureCanvasTkAgg, Figure, ttk, config
    global generate_alerts, parse_explanation_json, build_fight_brief
    global threshold_context_for_alerts, detect_card_change, run_full_analysis
    global run_quick_odds_refresh, run_quick_props_refresh, extract_bet_candidates, kelly_stake
    global strategy_from_profile, example_threshold_table

    def _step(msg: str) -> None:
        if progress:
            progress(msg)
        _debug_log(msg)

    try:
        _step("Loading XGBoost...")
        import xgboost  # noqa: F401

        _step("Loading LightGBM...")
        import lightgbm  # noqa: F401

        _step("Loading NumPy / Pandas...")
        import numpy as np
        import pandas as pd

        _step("Loading Matplotlib...")
        import logging as _logging
        import matplotlib

        matplotlib.use("TkAgg")
        _logging.getLogger("matplotlib").setLevel(_logging.WARNING)
        _logging.getLogger("matplotlib.font_manager").setLevel(_logging.WARNING)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from tkinter import ttk

        _step("Loading config and services...")
        import config

        config.refresh_runtime_env()
        _debug_log(f"ENABLE_PROPS loaded as: {config.ENABLE_PROPS}")
        _debug_log(f"Loaded MYBOOKIE_ENABLED = {config.MYBOOKIE_ENABLED}")
        from src.alerts import generate_alerts
        from src.explainability import parse_explanation_json
        from src.fight_brief import build_fight_brief
        from src.parlay_builder import threshold_context_for_alerts
        from src.dashboard_service import (
            detect_card_change,
            run_full_analysis,
            run_quick_odds_refresh,
            run_quick_props_refresh,
        )
        from src.strategy import extract_bet_candidates, kelly_stake, strategy_from_profile
        from ufc_betting_bot.modules.dynamic_thresholds import example_threshold_table

        _step("Dependencies ready.")
    except Exception as exc:
        raise RuntimeError(f"{exc}\n\n{traceback.format_exc()}") from exc


# Module-level placeholders (filled by _load_dependencies in main)
# ctk is set during bootstrap above - do not reset here.
np = None
pd = None
matplotlib = None
FigureCanvasTkAgg = None
Figure = None
ttk = None
config = None
generate_alerts = None
parse_explanation_json = None
build_fight_brief = None
threshold_context_for_alerts = None
detect_card_change = None
run_full_analysis = None
run_quick_odds_refresh = None
run_quick_props_refresh = None
extract_bet_candidates = None
kelly_stake = None
strategy_from_profile = None
example_threshold_table = None


class SplashScreen:
    """Non-Tk splash progress.

    Using a temporary ``tk.Tk()`` splash then destroying it before CustomTkinter
    recreates the Tcl interpreter has caused Windows ACCESS_VIOLATION hard-closes
    (window opens briefly then the process dies). Progress goes to heartbeats /
    console only.
    """

    def __init__(self) -> None:
        self._status = "Starting..."
        _dashboard_heartbeat("splash: start (non-Tk)")

    def set_status(self, text: str) -> None:
        self._status = str(text or "")
        _dashboard_heartbeat(f"splash: {self._status}")
        if _DEBUG_MODE:
            _debug_log(self._status)

    def close(self) -> None:
        _dashboard_heartbeat("splash: close")

    def pump(self) -> None:
        return

# --- Data engine --------------------------------------------------------------


def _as_dataframe(value: Any) -> pd.DataFrame:
    """Coerce to DataFrame; never return None (avoids ambiguous truth checks)."""
    import pandas as _pd

    if isinstance(value, _pd.DataFrame):
        return value
    return _pd.DataFrame()


def _df_is_empty(value: Any) -> bool:
    """True when value is None or an empty DataFrame (safe boolean)."""
    import pandas as _pd

    if value is None:
        return True
    if isinstance(value, _pd.DataFrame):
        return bool(value.empty)
    return True


def _df_row_count(value: Any) -> int:
    """Safe row/item count for DataFrame / list / None."""
    import pandas as _pd

    if value is None:
        return 0
    if isinstance(value, _pd.DataFrame):
        return int(len(value))
    if isinstance(value, (list, tuple)):
        return len(value)
    try:
        return len(value)
    except TypeError:
        return 0


def _normalize_card_dicts(cards: Any) -> list[dict[str, Any]]:
    """Ensure cards is a list of dicts with DataFrame predictions."""
    if not isinstance(cards, list):
        return []
    out: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        item = dict(card)
        item["predictions"] = _as_dataframe(item.get("predictions"))
        out.append(item)
    return out


class DashboardPayload:
    """In-memory analysis snapshot for all tabs."""

    def __init__(self) -> None:
        import pandas as _pd

        self.generated_at = ""
        self.event_label = ""
        self.profile = "paper"
        self.cards: list[dict[str, Any]] = []
        self.combined: pd.DataFrame = _pd.DataFrame()
        self.books: dict[str, dict[str, Any]] = {}
        self.risk_metrics: dict[str, Any] = {}
        self.threshold_ctx: dict[str, Any] = {}
        self.errors: list[str] = []
        self.odds_updated_at = ""
        self.from_cache = False
        self.prop_backtest: dict[str, Any] = {}
        self.arb_scan: dict[str, Any] = {}

    @property
    def all_preds(self) -> pd.DataFrame:
        import pandas as _pd

        frames = [
            c["predictions"]
            for c in self.cards
            if isinstance(c.get("predictions"), _pd.DataFrame) and not c["predictions"].empty
        ]
        return _pd.concat(frames, ignore_index=True) if frames else _pd.DataFrame()


def _result_to_payload(data: dict[str, Any]) -> DashboardPayload:
    p = DashboardPayload()
    p.generated_at = data.get("generated_at", "")
    p.event_label = data.get("event_label", "")
    p.profile = config.normalize_profile(data.get("profile", "paper"))
    p.cards = _normalize_card_dicts(data.get("cards"))
    p.combined = _as_dataframe(data.get("combined"))
    p.books = data.get("books") if isinstance(data.get("books"), dict) else {}
    p.risk_metrics = data.get("risk_metrics") if isinstance(data.get("risk_metrics"), dict) else {}
    p.threshold_ctx = data.get("threshold_ctx") if isinstance(data.get("threshold_ctx"), dict) else {}
    errs = data.get("errors")
    p.errors = list(errs) if isinstance(errs, list) else []
    p.odds_updated_at = data.get("odds_updated_at", p.generated_at)
    p.from_cache = bool(data.get("from_cache", False))
    p.prop_backtest = data.get("prop_backtest") if isinstance(data.get("prop_backtest"), dict) else {}
    p.arb_scan = data.get("arb_scan") if isinstance(data.get("arb_scan"), dict) else {}
    # Normalize book prediction frames so tab render never hits ambiguous truth checks.
    for book_data in p.books.values():
        if not isinstance(book_data, dict):
            continue
        if "predictions" in book_data:
            book_data["predictions"] = _as_dataframe(book_data.get("predictions"))
    return p


def run_dashboard_analysis(
    *,
    event_mode: str,
    profile: str,
    force_refresh_odds: bool = False,
    explain: bool = True,
    use_cache: bool = True,
    progress: Callable[[str, float | None], None] | None = None,
    budget_state: dict[str, Any] | None = None,
) -> DashboardPayload:
    def _prog(msg: str, pct: float | None = None) -> None:
        if progress:
            progress(msg, pct)

    data = run_full_analysis(
        event_mode=event_mode,
        profile=profile,
        force_refresh_odds=force_refresh_odds,
        explain=explain,
        use_cache=use_cache,
        progress=_prog,
        budget_state=budget_state,
    )
    # Never use `df or []` - DataFrame truth value is ambiguous.
    _button_debug(
        f"run_dashboard_analysis({event_mode!r}) -> run_full_analysis() "
        f"returned {_df_row_count(data.get('combined'))} combined fights, "
        f"{_df_row_count(data.get('cards'))} card(s), "
        f"books={list((data.get('books') or {}).keys())}"
    )
    data["profile"] = profile
    return _result_to_payload(data)


def _ensure_props_config() -> bool:
    """Reload .env and refresh ENABLE_PROPS (props tabs + analysis)."""
    if config is None:
        return False
    try:
        from src.project_paths import reload_runtime_env

        reload_runtime_env(_ROOT, log=_debug_log if _DEBUG_MODE else None)
    except Exception as exc:
        _debug_log(f"Props config reload failed: {exc}")
        config.refresh_runtime_env()
    enabled = bool(config.ENABLE_PROPS)
    _debug_log(f"ENABLE_PROPS loaded as: {enabled}")
    return enabled


def _top_shap(row: pd.Series) -> str:
    if pd.notna(row.get("shap_explanation")):
        exp = parse_explanation_json(row.get("shap_explanation"))
        toward = exp.get("toward_pick") or exp.get("top_features") or []
        if toward:
            return str(toward[0].get("label", ""))[:48]
    return ""


def _pick_edge(row: pd.Series) -> tuple[float | None, str | None]:
    f1 = str(row.get("fighter_1", ""))
    f2 = str(row.get("fighter_2", ""))
    pick = str(row.get("predicted_winner", ""))
    if not _has_usable_odds(row):
        return None, pick or None
    if bool(row.get("edge_suspect")):
        return None, pick or None
    edge: float | None = None
    if pd.notna(row.get("edge_pct")):
        edge = float(row["edge_pct"]) / 100.0
    elif pd.notna(row.get("best_edge")):
        edge = float(row["best_edge"])
        if abs(edge) > 1.5:  # sometimes stored as percent
            edge = edge / 100.0
    elif pd.notna(row.get("edge_f1")) or pd.notna(row.get("edge_f2")):
        e1 = float(row["edge_f1"]) if pd.notna(row.get("edge_f1")) else float("-inf")
        e2 = float(row["edge_f2"]) if pd.notna(row.get("edge_f2")) else float("-inf")
        edge = max(e1, e2)
        if edge == float("-inf"):
            edge = None
    if edge is None:
        return None, pick or None
    if abs(edge) > _MAX_DISPLAY_EDGE or edge <= -0.50:
        return None, pick or None
    return edge, pick or None


def _site_odds(row: pd.Series, pick: str | None) -> str:
    if not pick:
        pick = str(row.get("predicted_winner", "") or "")
    f1 = str(row.get("fighter_1", ""))
    if not pick or not _has_usable_odds(row):
        return "—"
    if pick == f1 and pd.notna(row.get("f1_odds")):
        try:
            o = float(row["f1_odds"])
            return f"{o:.2f}" if o > 1.0 else "—"
        except (TypeError, ValueError):
            return "—"
    if pd.notna(row.get("f2_odds")):
        try:
            o = float(row["f2_odds"])
            return f"{o:.2f}" if o > 1.0 else "—"
        except (TypeError, ValueError):
            return "—"
    return "—"


def _kelly_pct(row: pd.Series, bankroll: float, strategy) -> str:
    gate = None
    try:
        from src.uncertainty_gates import PAPER_WIDE_OVERRIDE, evaluate_uncertainty_gate

        gate = evaluate_uncertainty_gate(row)
        if gate.skip:
            return f"SKIP:{gate.reason_label()[:12]}"
    except Exception:
        return "SKIP:unc"
    cand = extract_bet_candidates(row, config=strategy)
    if cand is None or bankroll <= 0:
        return "-"
    stake = kelly_stake(
        bankroll,
        prob=cand.prob,
        decimal_odds=cand.decimal_odds,
        edge=cand.edge,
        config=strategy,
        row=row,
        uncertainty_kelly_mult=float(gate.kelly_mult) if gate is not None else 1.0,
    )
    is_override = False
    try:
        from src.uncertainty_gates import PAPER_WIDE_OVERRIDE as _PWO

        is_override = gate is not None and (
            gate.primary_reason == _PWO or _PWO in (gate.reasons or [])
        )
        if is_override:
            max_frac = float(
                getattr(config, "PAPER_WIDE_OVERRIDE_MAX_STAKE_FRAC", 0.01) or 0.01
            )
            stake = min(float(stake), float(bankroll) * max(0.0, max_frac))
    except Exception:
        is_override = False
    if stake <= 0:
        return "-"
    pct = f"{stake / bankroll * 100:.2f}%"
    if is_override:
        return f"{pct} paper_wide_override"
    return pct


class _ToolTip:
    """Hover tooltip for CustomTkinter widgets (uses tk.Toplevel)."""

    def __init__(self, widget, text: str, *, delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip: Any = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return
        import tkinter as tk

        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)
        lbl = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#1e293b",
            foreground="#e2e8f0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=8,
            pady=6,
            wraplength=320,
        )
        lbl.pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def _model_prob_for_row(row: pd.Series) -> float:
    """Model win probability for the predicted pick (used for table sorting)."""
    from src.strategy import _pick_model_prob

    _pick, prob, _fight = _pick_model_prob(row)
    return float(prob) if pd.notna(prob) else 0.0


def _sort_preds_by_model_prob(preds: pd.DataFrame) -> pd.DataFrame:
    if preds is None or preds.empty:
        return preds
    scored = preds.copy()
    pick = scored.get("predicted_winner", pd.Series("", index=scored.index)).astype(str)
    f1 = scored.get("fighter_1", pd.Series("", index=scored.index)).astype(str)
    f2 = scored.get("fighter_2", pd.Series("", index=scored.index)).astype(str)
    p1 = pd.to_numeric(scored.get("prob_f1_win", scored.get("predicted_prob")), errors="coerce")
    p2 = pd.to_numeric(scored.get("prob_f2_win"), errors="coerce")
    scored["_sort_prob"] = np.where(
        pick.eq(f1),
        p1,
        np.where(pick.eq(f2), p2, p1),
    ).astype(float)
    scored["_sort_prob"] = scored["_sort_prob"].fillna(0.0)
    return scored.sort_values("_sort_prob", ascending=False).drop(columns="_sort_prob").reset_index(drop=True)


_FIGHT_TIER_SORT_RANK = {
    "blue": 0,
    "sky_blue": 1,
    "green": 2,
    "yellow": 3,
    "red": 4,
}


def _fight_table_sort_key(row: dict[str, Any]) -> tuple:
    """Stable fight-table order: color → edge↓ → model_prob↓ → fight name↑."""
    tier = str(row.get("tier") or "").strip().lower()
    edge = row.get("sort_edge")
    try:
        edge_f = float(edge) if edge is not None else -999.0
    except (TypeError, ValueError):
        edge_f = -999.0
    try:
        prob_f = float(row.get("sort_prob") or 0.0)
    except (TypeError, ValueError):
        prob_f = 0.0
    fight = str(row.get("sort_fight") or "").strip().lower()
    return (
        _FIGHT_TIER_SORT_RANK.get(tier, 4),
        -edge_f,
        -prob_f,
        fight,
    )


def _rows_for_table(
    preds: pd.DataFrame,
    bankroll: float,
    strategy,
    *,
    compact: bool = False,
    cleared_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    from src.bet_tiers import classify_bet_tier, row_clears_gates

    rows: list[dict[str, Any]] = []
    cleared = cleared_keys or set()
    min_model_prob = float(getattr(strategy, "min_model_prob", 0.70) or 0.70)
    for _, row in _sort_preds_by_model_prob(preds).iterrows():
        edge, pick = _pick_edge(row)
        f1, f2 = str(row.get("fighter_1", "")), str(row.get("fighter_2", ""))
        fight_name = f"{f1} vs {f2}"
        prob = row.get("predicted_prob", row.get("prob_f1_win"))
        if pd.notna(prob) and pick == f2 and pd.notna(row.get("prob_f2_win")):
            prob = row["prob_f2_win"]
        edge_txt = _format_edge_display(edge)
        odds_txt = _site_odds(row, pick)
        if odds_txt in {"—", "-", ""}:
            edge_txt = "—"
            edge = None
        prob_txt = f"{float(prob):.0%}" if pd.notna(prob) else "—"
        sort_prob = _model_prob_for_row(row)
        clears = row_clears_gates(row, cleared)
        kelly_txt = _kelly_pct(row, bankroll, strategy)
        # Color from pick-side math + Kelly/status — never from opponent/fight string alone.
        # SKIP status forces non-blue even if clears_gates is somehow set.
        # Cap absurd edges so they do not paint Green/Blue from scrape glitches.
        # classify_bet_tier parses stake % from Kelly text (e.g. paper_wide_override)
        # so book tables match Ollama sky-blue tickets.
        edge_for_tier = float(edge) if edge is not None else None
        tier, reason = classify_bet_tier(
            row,
            clears_gates=clears and "SKIP" not in str(kelly_txt).upper(),
            min_model_prob=min_model_prob,
            status=kelly_txt,
            edge=edge_for_tier,
            model_prob=float(prob) if pd.notna(prob) else None,
            pick=str(pick or "") or None,
            debug=True,
        )
        _debug_log(
            f"fight_row_color pick={pick!r} prob={prob_txt} edge={edge_txt} "
            f"status={kelly_txt!r} stake={'yes' if clears else 'no'} "
            f"color={tier} reason={reason}"
        )
        book = str(
            row.get("bookmaker")
            or row.get("odds_book")
            or row.get("odds_source")
            or "—"
        ).strip() or "—"
        if book.lower() in {"the_odds_api", "odds_api"}:
            book = "Odds API"
        if compact:
            values: tuple = (
                fight_name,
                pick or "—",
                prob_txt,
                odds_txt,
                edge_txt,
                book,
                kelly_txt,
            )
        else:
            brief = build_fight_brief(row, edge_pct=edge * 100 if edge else None)[:120]
            values = (
                fight_name,
                pick or "—",
                prob_txt,
                odds_txt,
                edge_txt,
                book,
                kelly_txt,
                brief,
                _top_shap(row) or "—",
            )
        rows.append(
            {
                "values": values,
                "tier": tier,
                "pick": str(pick or "") or "—",
                "sort_prob": sort_prob,
                "sort_edge": float(edge) if edge is not None else -999.0,
                "sort_fight": fight_name,
                "row_meta": row.to_dict() if hasattr(row, "to_dict") else dict(row),
            }
        )
    rows.sort(key=_fight_table_sort_key)
    order_bits = []
    for r in rows:
        edge_v = r.get("sort_edge")
        edge_s = f"{float(edge_v) * 100:+.1f}%" if edge_v is not None and float(edge_v) > -900 else "—"
        order_bits.append(f"{r.get('pick')}|{r.get('tier')}|{edge_s}")
    _debug_log(f"fight_table_sorted order: {' > '.join(order_bits)}")
    return [
        {"values": r["values"], "tier": r["tier"], "row_meta": r.get("row_meta")}
        for r in rows
    ]


def _norm_event_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def _format_card_header(name: str) -> str:
    """Short, distinct section title for Overview / book / Next Two tabs."""
    import re

    n = str(name or "Card").strip()
    low = _norm_event_name(n)
    if "freedom" in low and "250" in low:
        return "Freedom 250"
    month_pat = (
        r"(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(\d{1,2})(?:\s+(\d{4}))?"
    )
    m = re.search(month_pat, low)
    if "fight night" in low and m:
        return f"Fight Night - {m.group(1).title()} {m.group(2)}"
    if low.startswith("ufc "):
        return n[4:].strip().title() if len(n) > 4 else n
    return n


def _card_identity(card: dict[str, Any]) -> str:
    """Fingerprint for deduping card sections (URL, fight ids, or normalized name)."""
    preds = card.get("predictions")
    if isinstance(preds, pd.DataFrame) and not preds.empty:
        if "event_url" in preds.columns:
            urls = preds["event_url"].dropna().astype(str)
            if not urls.empty:
                return str(urls.iloc[0]).split("#")[0].strip()
        key = getattr(config, "FIGHT_ID_COLUMN", "fight_id")
        if key in preds.columns:
            ids = sorted(preds[key].astype(str).head(5))
            return "|".join(ids)
    path = str(card.get("event_path") or "").split("#")[0].strip()
    if path:
        return path
    return _norm_event_name(str(card.get("event_name") or ""))


def _display_cards(
    payload: "DashboardPayload",
    preds: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Cards for grouped UI - payload.cards, else split preds/combined by event_name.

    Backfills empty card ``predictions`` from combined/preds so both upcoming
    cards populate whenever the payload has rows for that event.
    """
    frame = preds if isinstance(preds, pd.DataFrame) and not preds.empty else payload.combined
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame()

    def _slice_event(ev: str) -> pd.DataFrame:
        if frame.empty or not ev or "event_name" not in frame.columns:
            return pd.DataFrame()
        col = frame["event_name"].astype(str).str.strip()
        exact = frame[col == ev]
        if not exact.empty:
            return exact
        norm = _norm_event_name(ev)
        fuzzy = frame[col.map(_norm_event_name) == norm]
        return fuzzy if not fuzzy.empty else pd.DataFrame()

    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for card in payload.cards or []:
        ev = str(card.get("event_name") or "").strip()
        ident = _card_identity(card) or _norm_event_name(ev)
        if not ev or ident in seen:
            continue
        cp = card.get("predictions", pd.DataFrame())
        if not isinstance(cp, pd.DataFrame):
            cp = pd.DataFrame()
        if cp.empty:
            filled = _slice_event(ev)
            if not filled.empty:
                cp = filled
        cleaned.append(
            {
                "event_name": ev,
                "event_index": card.get("event_index"),
                "predictions": cp,
            }
        )
        seen.add(ident)
    if len(cleaned) >= 2:
        return cleaned

    if isinstance(frame, pd.DataFrame) and not frame.empty and "event_name" in frame.columns:
        order: list[str] = []
        for raw in frame["event_name"].dropna().astype(str):
            ev = raw.strip()
            key = _norm_event_name(ev)
            if ev and key not in seen:
                order.append(ev)
                seen.add(key)
        if len(order) >= 2:
            return [
                {
                    "event_name": ev,
                    "predictions": frame[frame["event_name"].astype(str).str.strip() == ev],
                }
                for ev in order
            ]

    if len(cleaned) == 1 and isinstance(frame, pd.DataFrame) and not frame.empty:
        only = cleaned[0]
        ev = only.get("event_name", "")
        if only.get("predictions") is None or (
            isinstance(only.get("predictions"), pd.DataFrame) and only["predictions"].empty
        ):
            filled = _slice_event(str(ev))
            if not filled.empty:
                only = {**only, "predictions": filled}
        if "event_name" in frame.columns:
            groups = frame["event_name"].astype(str).str.strip().unique().tolist()
            groups = [g for g in groups if g and g != ev]
            if groups:
                out = [only]
                for g in groups:
                    chunk = frame[frame["event_name"].astype(str).str.strip() == g]
                    if not chunk.empty:
                        out.append({"event_name": g, "predictions": chunk})
                if len(out) >= 2:
                    return out

    label = str(payload.event_label or "")
    if " + " in label:
        names = [p.strip() for p in label.split(" + ") if p.strip()]
        if len(names) >= 2 and isinstance(frame, pd.DataFrame) and not frame.empty:
            out: list[dict[str, Any]] = []
            for name in names:
                chunk = _slice_event(name)
                if chunk.empty and "event_name" in frame.columns:
                    exact = frame[frame["event_name"].astype(str).str.strip() == name]
                    if not exact.empty:
                        chunk = exact
                    else:
                        chunk = frame
                if not chunk.empty:
                    out.append(
                        {
                            "event_name": name,
                            "predictions": chunk if isinstance(chunk, pd.DataFrame) else pd.DataFrame(),
                        }
                    )
            nonempty = [
                c
                for c in out
                if isinstance(c.get("predictions"), pd.DataFrame) and not c["predictions"].empty
            ]
            if len(nonempty) >= 2:
                return nonempty
            if len(out) >= 2 and any(
                isinstance(c.get("predictions"), pd.DataFrame) and not c["predictions"].empty
                for c in out
            ):
                return out

    return cleaned if cleaned else list(payload.cards or [])


def _dedupe_fight_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate bout rows (same fight_id or fighter pair)."""
    if _df_is_empty(df):
        return _as_dataframe(df)
    out = df.copy()
    if config.FIGHT_ID_COLUMN in out.columns:
        return out.drop_duplicates(subset=[config.FIGHT_ID_COLUMN], keep="first").reset_index(drop=True)
    f1 = "fighter_1" if "fighter_1" in out.columns else ("fighter1" if "fighter1" in out.columns else None)
    f2 = "fighter_2" if "fighter_2" in out.columns else ("fighter2" if "fighter2" in out.columns else None)
    if f1 and f2:
        return out.drop_duplicates(subset=[f1, f2], keep="first").reset_index(drop=True)
    return out.drop_duplicates().reset_index(drop=True)


_ODDS_DISPLAY_COLS = (
    "f1_odds",
    "f2_odds",
    "implied_prob_f1",
    "implied_prob_f2",
    "mkt_implied_prob",
    "edge_f1",
    "edge_f2",
    "edge_pct",
    "best_edge",
    "best_edge_side",
    "bookmaker_count",
    "bookmaker",
    "odds_book",
    "odds_source",
)


def _strip_book_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove book odds columns so a book tab does not show Overview/consensus lines."""
    if _df_is_empty(df):
        return _as_dataframe(df)
    out = df.copy()
    for col in _ODDS_DISPLAY_COLS:
        if col in out.columns:
            out[col] = pd.NA
    if "odds_matched" in out.columns:
        out["odds_matched"] = False
    return out


def _model_fights_from_payload(payload: "DashboardPayload") -> pd.DataFrame:
    """Model fight rows for a book tab when that book has no odds-loaded predictions."""
    frames: list[pd.DataFrame] = []
    for card in payload.cards or []:
        cp = card.get("predictions", pd.DataFrame())
        if isinstance(cp, pd.DataFrame) and not cp.empty:
            frames.append(cp)
    if frames:
        return _strip_book_odds_columns(_dedupe_fight_rows(pd.concat(frames, ignore_index=True)))
    if not _df_is_empty(payload.combined):
        return _strip_book_odds_columns(_dedupe_fight_rows(payload.combined))
    return pd.DataFrame()


def _merge_fights_with_book_odds(card_df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    """Attach this book's odds/edge columns onto card rows (by fight_id or fighter pair)."""
    if _df_is_empty(card_df):
        return _dedupe_fight_rows(_as_dataframe(book_df))
    if _df_is_empty(book_df):
        # Book empty — do not keep consensus odds from the card model frame
        return _strip_book_odds_columns(_dedupe_fight_rows(_as_dataframe(card_df)))
    odds_cols = [c for c in (*_ODDS_DISPLAY_COLS, "odds_matched") if c in book_df.columns]
    if not odds_cols:
        return _strip_book_odds_columns(_dedupe_fight_rows(card_df))
    key = config.FIGHT_ID_COLUMN
    if key in card_df.columns and key in book_df.columns:
        base = card_df.drop(columns=odds_cols, errors="ignore")
        merged = base.merge(book_df[[key, *odds_cols]], on=key, how="left")
        if "odds_matched" in merged.columns:
            merged["odds_matched"] = merged["odds_matched"].fillna(False)
        return _dedupe_fight_rows(merged)
    f1 = "fighter_1" if "fighter_1" in card_df.columns else "fighter1"
    f2 = "fighter_2" if "fighter_2" in card_df.columns else "fighter2"
    if f1 in card_df.columns and f2 in card_df.columns and f1 in book_df.columns and f2 in book_df.columns:
        base = card_df.drop(columns=odds_cols, errors="ignore")
        merged = base.merge(book_df[[f1, f2, *odds_cols]], on=[f1, f2], how="left")
        if "odds_matched" in merged.columns:
            merged["odds_matched"] = merged["odds_matched"].fillna(False)
        return _dedupe_fight_rows(merged)
    return _strip_book_odds_columns(_dedupe_fight_rows(card_df))


def _preds_for_card(
    preds: pd.DataFrame,
    card_preds: pd.DataFrame,
    event_name: str,
    *,
    strip_unmatched_odds: bool = False,
) -> pd.DataFrame:
    """Rows for one card using this book's odds merged onto card fights.

    When ``strip_unmatched_odds`` is True (book tabs), never leave consensus
    odds on the card if this book has no matched slice for the event.
    Always prefer non-empty card predictions over an empty header-only section.
    """
    ev = str(event_name or "").strip()
    norm_ev = _norm_event_name(ev)
    base = (
        _dedupe_fight_rows(card_preds)
        if isinstance(card_preds, pd.DataFrame) and not card_preds.empty
        else pd.DataFrame()
    )

    if not isinstance(preds, pd.DataFrame) or preds.empty:
        return base

    book_slice = pd.DataFrame()
    if norm_ev and "event_name" in preds.columns:
        col = preds["event_name"].astype(str).str.strip()
        book_slice = preds[col.map(_norm_event_name) == norm_ev]
        if book_slice.empty:
            book_slice = preds[col == ev]
        if book_slice.empty and norm_ev:
            # Fuzzy: "UFC 330" vs longer event title
            try:
                book_slice = preds[
                    col.map(_norm_event_name).str.contains(norm_ev, regex=False)
                    | col.str.contains(ev, case=False, regex=False)
                ]
            except Exception:
                book_slice = pd.DataFrame()
    elif not base.empty:
        return _merge_fights_with_book_odds(base, preds)

    if not book_slice.empty:
        if not base.empty:
            return _merge_fights_with_book_odds(base, book_slice)
        return _dedupe_fight_rows(book_slice)

    # No book match for this event — keep card fights; strip consensus on book tabs
    if not base.empty:
        return _strip_book_odds_columns(base) if strip_unmatched_odds else base
    return base


def _cleared_fight_keys_from_payload(payload: "DashboardPayload | None") -> set[str]:
    """Fight ids / labels that cleared HA singles gates (BLUE)."""
    from src.bet_tiers import singles_cleared_keys

    if payload is None:
        return set()
    keys: set[str] = set()
    books = getattr(payload, "books", None) or {}
    for data in books.values():
        if not isinstance(data, dict):
            continue
        keys |= singles_cleared_keys((data.get("alerts") or {}).get("singles") or [])
    return keys


def _render_grouped_fight_tables(
    parent,
    cards: list[dict[str, Any]],
    preds: pd.DataFrame,
    *,
    bankroll: float,
    strategy,
    compact: bool = True,
    table_height: int = 8,
    payload: "DashboardPayload | None" = None,
    strip_unmatched_odds: bool = False,
) -> None:
    """Render one bordered section per upcoming card, or a single flat table."""
    from src.bet_tiers import format_tier_legend

    for w in list(parent.winfo_children()):
        try:
            w.destroy()
        except Exception:
            pass

    if payload is not None:
        cards = _display_cards(payload, preds)
    elif len(cards) < 2:
        cards = _display_cards(
            type("_P", (), {"cards": cards, "combined": preds, "event_label": ""})(),
            preds,
        )

    multi = len(cards) > 1
    cleared = _cleared_fight_keys_from_payload(payload)
    _debug_log(f"Grouped fight tables: {len(cards)} card section(s), multi={multi}")

    legend = ctk.CTkLabel(
        parent,
        text=format_tier_legend(),
        font=ctk.CTkFont(size=11),
        text_color="#94a3b8",
        anchor="w",
    )
    legend.pack(fill="x", padx=8, pady=(2, 0))

    context_wrap = ctk.CTkFrame(parent, fg_color="transparent")
    context_wrap.pack(fill="x", padx=8, pady=(2, 4))
    photo_row = ctk.CTkFrame(context_wrap, fg_color="transparent")
    photo_row.pack(fill="x", pady=(0, 2))
    photo_f1 = ctk.CTkLabel(photo_row, text="", width=72, height=72)
    photo_f1.pack(side="left", padx=(0, 8))
    photo_f2 = ctk.CTkLabel(photo_row, text="", width=72, height=72)
    photo_f2.pack(side="left", padx=(0, 12))
    photo_row.pack_forget()  # show only when images load
    _photo_refs: list[Any] = []  # keep CTkImage refs alive

    context_box = ctk.CTkLabel(
        context_wrap,
        text="Select a fight row for method / market / weigh-in context.",
        anchor="w",
        justify="left",
        font=ctk.CTkFont(size=11),
        text_color="#94a3b8",
        wraplength=1000,
    )
    context_box.pack(fill="x")

    def _set_photos(f1: str, f2: str) -> None:
        photo_row.pack_forget()
        photo_f1.configure(image=None, text="")
        photo_f2.configure(image=None, text="")
        _photo_refs.clear()
        if not f1 and not f2:
            return
        try:
            from PIL import Image
            from src.weigh_in import pair_image_paths

            p1, p2 = pair_image_paths(f1, f2, fetch=True)
            imgs: list[Any] = []
            for path, label in ((p1, photo_f1), (p2, photo_f2)):
                if path is None or not path.is_file():
                    label.configure(image=None, text="")
                    continue
                im = Image.open(path).convert("RGB")
                im.thumbnail((72, 72))
                cimg = ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
                _photo_refs.append(cimg)
                label.configure(image=cimg, text="")
                imgs.append(cimg)
            if imgs:
                photo_row.pack(fill="x", pady=(0, 2), before=context_box)
        except Exception as exc:
            _debug_log(f"weigh-in photos skipped: {exc}")

    def _on_row_select(meta: dict[str, Any] | None) -> None:
        try:
            from src.fight_context import build_fight_context, format_fight_context_lines

            lines = format_fight_context_lines(build_fight_context(meta))
            context_box.configure(text=_ascii_ui("\n".join(lines)))
            if meta:
                f1 = str(meta.get("fighter_1") or meta.get("fighter1") or "")
                f2 = str(meta.get("fighter_2") or meta.get("fighter2") or "")
                _set_photos(f1, f2)
            else:
                _set_photos("", "")
        except Exception as exc:
            context_box.configure(text=_ascii_ui(f"Context unavailable ({exc})"))
            _set_photos("", "")

    if not multi:
        tbl = DataTable(
            parent, height=table_height, compact=compact, on_row_select=_on_row_select
        )
        tbl.pack(fill="both", expand=True)
        flat = _dedupe_fight_rows(preds)
        if cards:
            flat = _preds_for_card(
                preds,
                cards[0].get("predictions", pd.DataFrame()),
                cards[0].get("event_name", ""),
                strip_unmatched_odds=strip_unmatched_odds,
            )
            if (flat is None or flat.empty) and isinstance(
                cards[0].get("predictions"), pd.DataFrame
            ):
                cp0 = cards[0]["predictions"]
                if not cp0.empty:
                    flat = (
                        _strip_book_odds_columns(cp0)
                        if strip_unmatched_odds
                        else _dedupe_fight_rows(cp0)
                    )
        tbl.load_rows(
            _rows_for_table(
                flat, bankroll, strategy, compact=compact, cleared_keys=cleared
            )
        )
        return

    border_colors = ("#fbbf24", "#38bdf8", "#a78bfa", "#34d399")
    for idx, card in enumerate(cards):
        ev = card.get("event_name") or f"Card {idx + 1}"
        header = _format_card_header(ev)
        cp = card.get("predictions", pd.DataFrame())
        if not isinstance(cp, pd.DataFrame):
            cp = pd.DataFrame()
        sub_df = _preds_for_card(
            preds, cp, ev, strip_unmatched_odds=strip_unmatched_odds
        )
        # Never leave a header-only section when the card payload has fights
        if (sub_df is None or sub_df.empty) and not cp.empty:
            sub_df = (
                _strip_book_odds_columns(_dedupe_fight_rows(cp))
                if strip_unmatched_odds
                else _dedupe_fight_rows(cp)
            )
        matched_n = (
            int(sub_df.get("odds_matched", pd.Series(False)).sum())
            if isinstance(sub_df, pd.DataFrame)
            and "odds_matched" in getattr(sub_df, "columns", [])
            else 0
        )
        n_fights = len(sub_df) if isinstance(sub_df, pd.DataFrame) else 0
        border = border_colors[idx % len(border_colors)]
        section = ctk.CTkFrame(
            parent,
            fg_color="#152238",
            corner_radius=10,
            border_width=2,
            border_color=border,
        )
        section.pack(fill="x", padx=6, pady=(10 if idx else 4, 6))
        status = (
            f"{header}  -  {n_fights} fights  |  {matched_n} with book odds"
            if n_fights
            else f"{header}  -  0 fights (card unavailable — Refresh Next Two)"
        )
        ctk.CTkLabel(
            section,
            text=_ascii_ui(status),
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
            text_color=border,
        ).pack(fill="x", padx=12, pady=(10, 4))
        sub = DataTable(
            section, height=table_height, compact=compact, on_row_select=_on_row_select
        )
        sub.pack(fill="x", padx=8, pady=(0, 10))
        sub.load_rows(
            _rows_for_table(
                sub_df if isinstance(sub_df, pd.DataFrame) else pd.DataFrame(),
                bankroll,
                strategy,
                compact=compact,
                cleared_keys=cleared,
            )
        )
        _debug_log(f"  Card section {idx}: {ev!r} -> {n_fights} rows")


# --- UI helpers ---------------------------------------------------------------


def _normalize_ranked_parlays(
    parlays: list[dict[str, Any]],
    preds: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Ensure parlays have rank + fighter names on every leg."""
    from src.parlay_builder import (
        enrich_parlays_for_display,
        format_recommended_parlay_legs,
        leg_pick_label,
        leg_stats_suffix,
    )

    sorted_p = sorted(parlays, key=lambda x: x.get("expected_value", 0), reverse=True)
    enriched = enrich_parlays_for_display(sorted_p, preds)
    out: list[dict[str, Any]] = []
    for i, p in enumerate(enriched):
        item = dict(p)
        item["rank"] = i + 1
        if not item.get("min_leg_edge") and item.get("legs"):
            item["min_leg_edge"] = min(leg.get("edge", 0) for leg in item["legs"])
        item["leg_labels"] = [leg_pick_label(leg) for leg in item.get("legs", [])]
        item["_leg_rows"] = format_recommended_parlay_legs(item)
        out.append(item)
    return out


def _render_ranked_singles(
    parent,
    singles: list[dict[str, Any]],
    *,
    title: str = "Top singles",
    preds: pd.DataFrame | None = None,
) -> None:
    """Compact ranked moneyline singles for a book tab."""
    del preds
    if not singles:
        return
    ctk.CTkLabel(
        parent,
        text=title,
        font=ctk.CTkFont(size=13, weight="bold"),
        anchor="w",
    ).pack(fill="x", pady=(8, 4))
    for i, s in enumerate(singles, start=1):
        pick = str(s.get("pick") or "-")
        fight = str(s.get("fight") or "-")
        edge = float(s.get("edge_pct") or s.get("edge", 0) or 0)
        if edge <= 1.0:
            edge *= 100.0
        stake = float(s.get("suggested_stake") or 0)
        unc = str(s.get("uncertainty_action") or "").strip().lower()
        unc_bit = ""
        if unc == "tighten":
            unc_bit = f"  |  unc tighten ({s.get('uncertainty_reason') or 'elevated'})"
        elif s.get("uncertainty_reason"):
            unc_bit = f"  |  unc {s.get('uncertainty_reason')}"
        ctk.CTkLabel(
            parent,
            text=_ascii_ui(
                f"#{i}  {pick} - {fight}  |  "
                f"edge {edge:+.1f}%  |  stake {_format_ticket_stake(s)}{unc_bit}"
            ),
            anchor="w",
            text_color="#34d399" if edge > 0 else "#d1d5db",
            font=ctk.CTkFont(size=12),
            wraplength=1050,
            justify="left",
        ).pack(fill="x", padx=(4, 0), pady=(0, 4))


def _render_uncertainty_skips(
    parent,
    skipped: list[dict[str, Any]],
    *,
    title: str = "Skips",
    limit: int = 8,
) -> None:
    if not skipped:
        return
    ctk.CTkLabel(
        parent,
        text=f"{title} ({len(skipped)})",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#fbbf24",
        anchor="w",
    ).pack(fill="x", pady=(8, 2))
    for s in skipped[:limit]:
        reason = str(s.get("skip_reason") or "unknown")
        fight = str(s.get("fight") or "-")
        pick = str(s.get("pick") or "-")
        ctk.CTkLabel(
            parent,
            text=_ascii_ui(f"  SKIP [{reason}]  {pick} — {fight}"),
            anchor="w",
            text_color="#fcd34d",
            font=ctk.CTkFont(size=11),
            wraplength=1050,
            justify="left",
        ).pack(fill="x", padx=(4, 0), pady=(0, 2))


def _render_skip_scorecard_panel(
    parent,
    scorecard: dict[str, Any] | None = None,
    *,
    title: str = "Skip scorecard (7d)",
) -> None:
    """Compact noise-vs-edge rollup for Risk / Overview."""
    try:
        from src.skip_scorecard import rollup_skip_reasons, top_skip_lines

        data = scorecard or rollup_skip_reasons(
            days=int(getattr(config, "SKIP_SCORECARD_LOOKBACK_DAYS", 7) or 7),
            write_json=False,
        )
    except Exception:
        return
    ctk.CTkLabel(
        parent,
        text=f"{title} — {data.get('total_skips', 0)} skips",
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#fbbf24",
        anchor="w",
    ).pack(fill="x", pady=(8, 2))
    lines = top_skip_lines(data, limit=5)
    if not lines:
        ctk.CTkLabel(
            parent,
            text="  (no skips logged — fail-closed)",
            anchor="w",
            text_color="#9ca3af",
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=(4, 0))
        return
    for line in lines:
        ctk.CTkLabel(
            parent,
            text=_ascii_ui(f"  {line}"),
            anchor="w",
            text_color="#fcd34d",
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=(4, 0), pady=(0, 1))
    mix = (
        f"  noise {data.get('noise_filter_pct', 0):.0f}% | "
        f"edge-floor {data.get('edge_left_pct', 0):.0f}%"
    )
    ctk.CTkLabel(
        parent,
        text=_ascii_ui(mix),
        anchor="w",
        text_color="#a5b4fc",
        font=ctk.CTkFont(size=11),
    ).pack(fill="x", padx=(4, 0), pady=(2, 0))
    interp = str(data.get("interpretation") or "")
    if interp:
        ctk.CTkLabel(
            parent,
            text=_ascii_ui(f"  → {interp}"),
            anchor="w",
            text_color="#d1d5db",
            font=ctk.CTkFont(size=11),
            wraplength=1050,
            justify="left",
        ).pack(fill="x", padx=(4, 0), pady=(2, 4))


def _render_sleeve_stats_panel(
    parent,
    report: dict[str, Any] | None = None,
    *,
    title: str = "Sleeve performance (top / bottom)",
) -> None:
    """Top and bottom sleeves for Risk tab."""
    try:
        from src.sleeve_stats import format_sleeve_dashboard_lines, run_sleeve_stats

        data = report or run_sleeve_stats(write_csv=False)
        lines = format_sleeve_dashboard_lines(data, limit=3)
    except Exception:
        return
    ctk.CTkLabel(
        parent,
        text=title,
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color="#67e8f9",
        anchor="w",
    ).pack(fill="x", pady=(8, 2))
    if not lines:
        ctk.CTkLabel(
            parent,
            text="  (no settled bets yet)",
            anchor="w",
            text_color="#9ca3af",
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=(4, 0))
        return
    for i, line in enumerate(lines):
        color = "#e2e8f0" if i == 0 else "#a5f3fc"
        if line.strip().startswith("-") or "Bottom" in line:
            color = "#fca5a5"
        elif line.strip().startswith("+") or "Top" in line:
            color = "#86efac"
        ctk.CTkLabel(
            parent,
            text=_ascii_ui(f"  {line}" if not line.startswith("Sleeve") else line),
            anchor="w",
            text_color=color,
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=(4, 0), pady=(0, 1))


def _render_ranked_parlays(
    parent,
    parlays: list[dict[str, Any]],
    *,
    title: str = "Recommended Parlays",
    preds: pd.DataFrame | None = None,
) -> None:
    from src.parlay_builder import format_recommended_parlay_header

    ranked = _normalize_ranked_parlays(parlays, preds=preds)
    if not ranked:
        return
    ctk.CTkLabel(
        parent,
        text=title,
        font=ctk.CTkFont(size=13, weight="bold"),
        anchor="w",
    ).pack(fill="x", pady=(8, 4))
    for p in ranked:
        block = ctk.CTkFrame(parent, fg_color="transparent")
        block.pack(fill="x", padx=(4, 0), pady=(0, 8))
        ctk.CTkLabel(
            block,
            text=format_recommended_parlay_header(p),
            anchor="w",
            text_color="#a5b4fc",
            font=ctk.CTkFont(size=12, weight="bold"),
            wraplength=1050,
            justify="left",
        ).pack(fill="x")
        for leg_row in p.get("_leg_rows", []):
            ctk.CTkLabel(
                block,
                text=leg_row,
                anchor="w",
                text_color="#d1d5db",
                font=ctk.CTkFont(size=12),
                wraplength=1050,
                justify="left",
            ).pack(fill="x", padx=(12, 0))


class DataTable(_CTK_FRAME):
    """Scrollable ttk tree table with edge coloring and clickable column sort."""

    COLUMNS = (
        "Fight",
        "Pick",
        "Prob",
        "Odds",
        "Edge",
        "Book",
        "Kelly",
        "Brief",
        "SHAP",
    )
    COMPACT_COLUMNS = ("Fight", "Pick", "Prob", "Odds", "Edge", "Book", "Kelly")
    _NUMERIC_COLS = frozenset({"Prob", "Odds", "Edge", "Kelly"})

    def __init__(
        self,
        master,
        *,
        height: int = 12,
        compact: bool = False,
        on_row_select: Callable[[dict[str, Any] | None], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._compact = compact
        self._on_row_select = on_row_select
        columns = self.COMPACT_COLUMNS if compact else self.COLUMNS
        self._columns = columns
        self._rows_data: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
        self._row_meta: list[dict[str, Any] | None] = []
        self._sort_col: str | None = None
        self._sort_desc = True
        style = ttk.Style()
        style.theme_use("clam")
        # Tk 8.6.9+ on Windows: default Treeview style map overrides item tag
        # colors. Strip !disabled/!selected so per-row tier tags can paint fg.
        def _fixed_map(option: str):
            return [
                elm
                for elm in style.map("Treeview", query_opt=option)
                if elm[:2] != ("!disabled", "!selected")
            ]

        try:
            style.map(
                "Treeview",
                foreground=_fixed_map("foreground"),
                background=_fixed_map("background"),
            )
            style.map(
                "Dash.Treeview",
                foreground=[("disabled", "#64748b"), ("selected", "#ffffff")],
                background=[("disabled", "#1e1e1e"), ("selected", "#334155")],
            )
        except Exception:
            pass
        row_h = 26 if compact else 30
        style.configure(
            "Dash.Treeview",
            background="#1e1e1e",
            foreground="#e8e8e8",
            fieldbackground="#1e1e1e",
            rowheight=row_h,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Dash.Treeview.Heading",
            background="#2b2b2b",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(6, 4),
        )
        self.tree = ttk.Treeview(
            self,
            columns=columns,
            show="headings",
            height=height,
            style="Dash.Treeview",
        )
        if compact:
            widths = (220, 140, 56, 64, 70, 90, 72)
        else:
            widths = (200, 120, 58, 58, 68, 90, 68, 280, 130)
        for col, w in zip(columns, widths):
            self.tree.heading(
                col,
                text=col,
                anchor="w",
                command=lambda c=col: self._on_heading_click(c),
            )
            stretch = col in ("Fight", "Brief")
            self.tree.column(col, width=w, minwidth=48, anchor="w", stretch=stretch)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(0, 0), pady=(2, 0))
        vsb.grid(row=0, column=1, sticky="ns")
        if not compact:
            hsb.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # Namespaced tags + shared TIER_COLORS (avoids theme keyword collisions).
        from src.bet_tiers import TIER_COLORS

        for _tier, _hex in TIER_COLORS.items():
            self.tree.tag_configure(f"tier_{_tier}", foreground=_hex)
            self.tree.tag_configure(_tier, foreground=_hex)  # legacy tag names
        self.tree.tag_configure("pos", foreground=TIER_COLORS["green"])
        self.tree.tag_configure("neg", foreground=TIER_COLORS["red"])
        self.tree.tag_configure("neutral", foreground="#b0b0b0")
        self.tree.tag_configure("even", background="#1a1f2e")
        self.tree.tag_configure("odd", background="#1e1e1e")
        self._bind_mousewheel()
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _on_tree_select(self, _event=None) -> None:
        if self._on_row_select is None:
            return
        sel = self.tree.selection()
        if not sel:
            self._on_row_select(None)
            return
        try:
            idx = self.tree.index(sel[0])
        except Exception:
            self._on_row_select(None)
            return
        meta = None
        if 0 <= idx < len(self._row_meta):
            meta = self._row_meta[idx]
        self._on_row_select(meta)

    def _bind_mousewheel(self) -> None:
        def _on_wheel(event) -> None:
            if event.delta:
                self.tree.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.tree.bind("<MouseWheel>", _on_wheel)
        # Focus tree for wheel while hovered; restore parent scroll on leave
        self.tree.bind("<Enter>", lambda _e: self.tree.focus_set())
        self.tree.bind(
            "<Leave>",
            lambda _e: self.master.focus_set() if hasattr(self.master, "focus_set") else None,
        )

    @staticmethod
    def _cell_sort_key(value: Any, *, numeric: bool) -> tuple[int, Any]:
        """Folder-style sort key: missing values sink; numbers parse from display text."""
        s = str(value if value is not None else "").strip()
        if s in {"", "-", "—", "n/a", "N/A", "."}:
            return (1, 0.0 if numeric else "")
        if numeric:
            cleaned = s.replace("%", "").replace(",", "").replace("$", "")
            m = re.search(r"[-+]?\d*\.?\d+", cleaned)
            if m:
                try:
                    return (0, float(m.group(0)))
                except ValueError:
                    pass
            return (1, 0.0)
        return (0, s.casefold())

    def _on_heading_click(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            # Stats: best on top first click; names: A→Z first click.
            self._sort_desc = col in self._NUMERIC_COLS
        self._redraw_sorted()

    def _refresh_heading_labels(self) -> None:
        for c in self._columns:
            label = str(c)
            if c == self._sort_col:
                label = f"{c} {'v' if self._sort_desc else '^'}"
            self.tree.heading(
                c,
                text=_ascii_ui(label),
                anchor="w",
                command=lambda col=c: self._on_heading_click(col),
            )

    def _redraw_sorted(self) -> None:
        self._refresh_heading_labels()
        try:
            self.tree.delete(*self.tree.get_children())
        except Exception:
            pass
        rows = list(self._rows_data)
        metas = list(self._row_meta) if self._row_meta else [None] * len(rows)
        # Keep meta aligned with rows when sorting
        paired = list(zip(rows, metas))
        if self._sort_col and paired:
            try:
                idx = list(self._columns).index(self._sort_col)
            except ValueError:
                idx = -1
            if idx >= 0:
                numeric = self._sort_col in self._NUMERIC_COLS
                paired.sort(
                    key=lambda item: self._cell_sort_key(
                        item[0][0][idx] if idx < len(item[0][0]) else "",
                        numeric=numeric,
                    ),
                    reverse=self._sort_desc,
                )
        self._row_meta = [m for _, m in paired]
        for i, ((values, tags), _meta) in enumerate(paired):
            raw = str(tags[0] if tags else "neutral").strip().lower()
            tier_tag = raw if raw.startswith("tier_") else (
                f"tier_{raw}"
                if raw in {"blue", "sky_blue", "green", "yellow", "red"}
                else raw
            )
            zebra = "even" if i % 2 == 0 else "odd"
            self.tree.insert(
                "", "end", values=_ascii_row(values), tags=(zebra, tier_tag)
            )

    def load_rows(self, rows: list[Any]) -> None:
        try:
            self.tree.delete(*self.tree.get_children())
        except Exception:
            pass
        self._rows_data = []
        self._row_meta = []
        empty_cols = 7 if self._compact else 9
        if not rows:
            self._sort_col = None
            self._refresh_heading_labels()
            self.tree.insert(
                "",
                "end",
                values=_ascii_row(
                    ["No fights loaded - click Refresh"] + [""] * (empty_cols - 1)
                ),
            )
            return
        for row in rows:
            tier = "neutral"
            meta = None
            if isinstance(row, dict):
                values = tuple(row.get("values") or ())
                tier = str(row.get("tier") or "neutral")
                meta = row.get("row_meta")
            else:
                values = tuple(row)
                edge_txt = values[4] if len(values) > 4 else ""
                if edge_txt not in ("-", "—", ""):
                    try:
                        edge_v = float(str(edge_txt).replace("%", "").replace("+", ""))
                        tier = "green" if edge_v > 0 else "red"
                    except ValueError:
                        pass
            self._rows_data.append((values, (tier,)))
            self._row_meta.append(meta if isinstance(meta, dict) else None)
        # Preserve caller order (color → edge → prob → fight). Only re-sort after
        # the user clicks a column header (_sort_col set).
        self._redraw_sorted()


class TopRecommendedBetsPanel(_CTK_FRAME):
    """Prominent #1-#5 recommendations on the Overview tab."""

    _PANEL_BG = "#0c1222"
    _INNER_BG = "#111827"
    _ROW_BG = "#1e293b"
    _HERO_BG = "#1e1b4b"
    _HERO_BORDER = "#fbbf24"
    _BORDER = "#334155"

    def __init__(self, master, **kwargs) -> None:
        super().__init__(
            master,
            fg_color=self._PANEL_BG,
            corner_radius=12,
            border_width=2,
            border_color="#475569",
            **kwargs,
        )
        inner = ctk.CTkFrame(self, fg_color=self._INNER_BG, corner_radius=10)
        inner.pack(fill="both", expand=True, padx=4, pady=4)

        hdr = ctk.CTkFrame(inner, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(14, 10))
        title_block = ctk.CTkFrame(hdr, fg_color="transparent")
        title_block.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(
            title_block,
            text="Top Recommended Bets",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f8fafc",
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            title_block,
            text=(
                "BET THIS (Blue) = sized $ · FUN ONLY (Green) = $0 lean · "
                "CAUTION (Yellow) / DO NOT BET (Red)"
            ),
            font=ctk.CTkFont(size=11),
            text_color="#64748b",
            anchor="w",
        ).pack(fill="x", pady=(2, 0))
        self.pool_label = ctk.CTkLabel(
            hdr,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#86efac",
            anchor="e",
        )
        self.pool_label.pack(side="right", padx=(12, 0))

        # Scrollable list grows with window; page-level Overview scroll also covers overflow.
        self._bets_scroll = ctk.CTkScrollableFrame(
            inner,
            fg_color="transparent",
            height=220,
            label_text="Top picks",
        )
        self._bets_scroll.pack(fill="both", expand=True, padx=4, pady=(0, 8))
        self.list_frame = self._bets_scroll
        self.empty_label = ctk.CTkLabel(
            inner,
            text="Refresh to load top edges from your selected books.",
            text_color="#64748b",
            anchor="w",
            font=ctk.CTkFont(size=12),
        )
        self.empty_label.pack(fill="x", padx=16, pady=(0, 14))

    @staticmethod
    def _odds_line(bet: dict[str, Any]) -> str:
        book = str(bet.get("book") or "-")
        am = str(bet.get("american_odds") or "-")
        dec = str(bet.get("odds_display") or "-")
        return f"{book}  {am} ({dec})"

    def _pick_line_text(self, bet: dict[str, Any], rank: int) -> tuple[str, str]:
        """Return (display line, tier color) for one pick inside the shared bubble."""
        from src.bet_tiers import TIER_COLORS, action_label_for_bet

        # Prefer bet_tier only — bet["tier"] is often actionable/advisory, not a color.
        tier = str(bet.get("bet_tier") or "").strip().lower()
        if tier not in TIER_COLORS:
            if bet.get("fun_bet") or bet.get("advisory"):
                tier = "green"
            else:
                tier = (
                    "blue"
                    if float(bet.get("suggested_stake") or bet.get("stake_usd") or 0) > 0
                    else "yellow"
                )
        color = TIER_COLORS.get(tier, "#e2e8f0")
        label = str(
            bet.get("display_label") or bet.get("pick_line") or bet.get("pick") or "-"
        )
        edge_pct = float(bet.get("edge_pct") or 0)
        action = action_label_for_bet({**bet, "bet_tier": tier})
        line = f"#{rank}  {action}  ·  {label}  ·  {edge_pct:+.1f}%"
        return _ascii_ui(line), color

    def _render_picks_bubble(self, bets: list[dict[str, Any]]) -> None:
        """All top picks in a single bubble (colored lines, not separate cards)."""
        from src.bet_slip import dedupe_rank_top_tickets, top_recommended_label
        from src.bet_tiers import format_tier_legend, format_what_to_do_header

        bets = dedupe_rank_top_tickets(list(bets or []), limit=5)
        bubble = ctk.CTkFrame(
            self.list_frame,
            fg_color=self._ROW_BG,
            corner_radius=12,
            border_width=2,
            border_color="#475569",
        )
        bubble.pack(fill="x", padx=4, pady=(0, 6))
        ctk.CTkLabel(
            bubble,
            text=top_recommended_label(len(bets), limit=5),
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#f8fafc",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(
            bubble,
            text=_ascii_ui(format_what_to_do_header(slip=bets)),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#fde68a",
            anchor="w",
            justify="left",
            wraplength=900,
        ).pack(fill="x", padx=14, pady=(0, 6))

        for bet in bets:
            rank = int(bet.get("rank") or 0) or (bets.index(bet) + 1)
            line, color = self._pick_line_text(bet, rank)
            ctk.CTkLabel(
                bubble,
                text=line,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=color,
                anchor="w",
                justify="left",
                wraplength=900,
            ).pack(fill="x", padx=14, pady=(0, 2))

        ctk.CTkLabel(
            bubble,
            text=format_tier_legend(),
            font=ctk.CTkFont(size=10),
            text_color="#64748b",
            anchor="w",
            justify="left",
            wraplength=900,
        ).pack(fill="x", padx=14, pady=(6, 12))

    def render(
        self,
        bets: list[dict[str, Any]],
        *,
        highlight_parlay: dict[str, Any] | None = None,
        pool_status: str | None = None,
        empty_reason: str | None = None,
    ) -> None:
        del highlight_parlay  # unified list via aggregate_overview_recommendations
        for w in self.list_frame.winfo_children():
            w.destroy()

        if not bets:
            self.pool_label.configure(text=_ascii_ui(pool_status or ""))
            self._bets_scroll.pack_forget()
            msg = empty_reason or "Refresh to load top edges from your selected books."
            self.empty_label.configure(text=_ascii_ui(msg))
            self.empty_label.pack(fill="x", padx=16, pady=(0, 14))
            return

        self.empty_label.pack_forget()
        self._bets_scroll.pack(fill="x", padx=4, pady=(0, 8))
        if pool_status:
            self.pool_label.configure(text=_ascii_ui(pool_status))
        else:
            allocated = sum(
                float(b.get("suggested_stake") or 0)
                for b in bets
                if not b.get("fun_bet")
            )
            sum_pct = sum(
                float(b.get("stake_pct") or 0) for b in bets if not b.get("fun_bet")
            )
            auto = float(bets[0].get("card_pool_usd") or 0)
            self.pool_label.configure(
                text=_ascii_ui(
                    f"Auto card ${auto:,.2f} · Allocated ${allocated:,.2f} "
                    f"({sum_pct:.0f}%) · Picks {len(bets)}"
                )
            )

        self._render_picks_bubble(bets)


def _format_grok_user_error(err: Any) -> str:
    """User-facing Ollama/analysis error text."""
    text = str(err or "Unknown error").strip()
    low = text.lower()
    if "timed out" in low or "timeout" in low:
        return text if "ollama timed out" in low else (
            f"Ollama timed out. {text} "
            f"(timeout={getattr(config, 'OLLAMA_TIMEOUT_SEC', 60)}s). "
            "Pick a faster model or raise OLLAMA_TIMEOUT_SEC in .env."
        )
    if "model missing" in low or ("pull" in low and "model" in low):
        return text
    if "ollama not running" in low or "host unreachable" in low or "connection refused" in low:
        return (
            text
            if "ollama" in low
            else (
                "Ollama not running. Start it with `ollama serve`, then "
                f"`ollama pull {getattr(config, 'OLLAMA_MODEL', 'qwen2.5-coder:14b')}`."
            )
        )
    if "403" in text or "forbidden" in low or "permission-denied" in low:
        return (
            "xAI/Grok is disabled. This dashboard uses local Ollama. "
            "Start Ollama and click Run Ollama Analysis."
        )
    return text or "Unknown error"


class GrokAnalysisPanel(_CTK_FRAME):
    """Optional Ollama narrative read on top fights/props (runs in background thread)."""

    def __init__(
        self,
        master,
        *,
        on_run: Callable[[], None] | None = None,
        on_chat: Callable[[str], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="#111827", corner_radius=10, **kwargs)
        self._on_run = on_run
        self._on_chat = on_chat
        self._think_after_id: str | None = None
        self._think_started_at: float | None = None
        self._chat_busy = False
        self._chat_ask_busy = False
        self._last_result: dict[str, Any] | None = None

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(
            hdr,
            text="Ollama Analysis",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f8fafc",
        ).pack(side="left")
        self.status_label = ctk.CTkLabel(
            hdr,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            anchor="e",
        )
        self.status_label.pack(side="right", fill="x", expand=True, padx=(8, 0))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 4))
        self.run_btn = ctk.CTkButton(
            btn_row,
            text="Run Ollama Analysis",
            width=170,
            fg_color="#0f766e",
            hover_color="#14b8a6",
            command=on_run,
        )
        self.run_btn.pack(side="left")

        from src.ollama_client import list_model_choices, model_speed_hint

        model_choices = list_model_choices()
        current = str(getattr(config, "OLLAMA_MODEL", model_choices[0]) or model_choices[0])
        if current not in model_choices:
            model_choices = [current, *model_choices]
        self.model_var = ctk.StringVar(value=current)
        ctk.CTkLabel(
            btn_row,
            text="Model",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
        ).pack(side="left", padx=(14, 4))
        self.model_menu = ctk.CTkOptionMenu(
            btn_row,
            values=model_choices,
            variable=self.model_var,
            width=200,
            command=self._on_model_selected,
        )
        self.model_menu.pack(side="left")

        timeout_s = int(getattr(config, "OLLAMA_TIMEOUT_SEC", 60) or 60)
        self.hint_label = ctk.CTkLabel(
            btn_row,
            text=(
                f"Model = narrative only (HA stakes unchanged) · "
                f"{model_speed_hint(current)} · timeout {timeout_s}s"
            ),
            font=ctk.CTkFont(size=11),
            text_color="#64748b",
            anchor="w",
        )
        self.hint_label.pack(side="left", padx=(12, 0))

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate", height=8)
        self.progress.pack(fill="x", padx=12, pady=(0, 4))
        self.progress.set(0)
        self.progress.pack_forget()

        self.scroll = ctk.CTkScrollableFrame(self, label_text="Top 5 recommended")
        self.scroll.pack(fill="both", expand=True, padx=10, pady=(4, 6))

        # --- Communication / chat bar ---
        chat_wrap = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=8)
        chat_wrap.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(
            chat_wrap,
            text="Ask Ollama",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#e2e8f0",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 2))
        self.chat_log = ctk.CTkTextbox(
            chat_wrap,
            height=120,
            font=ctk.CTkFont(size=12),
            fg_color="#111827",
            text_color="#cbd5e1",
            wrap="word",
            activate_scrollbars=True,
        )
        self.chat_log.pack(fill="x", padx=10, pady=(0, 6))
        self.chat_log.insert(
            "1.0",
            "Ask which bets look best, or click Best bets for an automatic HA/stats briefing.\n",
        )
        self.chat_log.configure(state="disabled")

        bar = ctk.CTkFrame(chat_wrap, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=(0, 10))
        self.chat_entry = ctk.CTkEntry(
            bar,
            placeholder_text="e.g. Which are the best bets on this card?",
            height=32,
        )
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.chat_entry.bind("<Return>", self._on_chat_return)
        self.best_bets_btn = ctk.CTkButton(
            bar,
            text="Best bets",
            width=90,
            fg_color="#1d4ed8",
            hover_color="#2563eb",
            command=self._on_best_bets_click,
        )
        self.best_bets_btn.pack(side="left", padx=(0, 6))
        self.ask_btn = ctk.CTkButton(
            bar,
            text="Ask",
            width=70,
            fg_color="#0f766e",
            hover_color="#14b8a6",
            command=self._on_ask_click,
        )
        self.ask_btn.pack(side="left")

    def _on_model_selected(self, value: str) -> None:
        from src.ollama_client import last_model_warn, model_speed_hint, set_active_model

        requested = str(value or "").strip()
        chosen = set_active_model(requested)
        self.model_var.set(chosen)
        self.refresh_model_choices()
        timeout_s = int(getattr(config, "OLLAMA_TIMEOUT_SEC", 60) or 60)
        warn = last_model_warn()
        if warn:
            note = warn
            status = warn
        else:
            note = model_speed_hint(chosen)
            status = f"Model set to {chosen}"
        try:
            self.hint_label.configure(
                text=(
                    f"Model = narrative only (HA stakes unchanged) · "
                    f"{note} · timeout {timeout_s}s"
                )
            )
        except Exception:
            pass
        self.status_label.configure(text=status)
        _debug_log(f"Ollama model selected: requested={requested!r} active={chosen!r}")

    def sync_active_model(self) -> str:
        """Apply the dropdown selection to runtime config before any Ollama call."""
        from src.ollama_client import (
            _is_heavy_model,
            _model_param_b,
            last_model_warn,
            ollama_installed_models,
            set_active_model,
        )

        requested = str(self.model_var.get() or "").strip()
        installed = ollama_installed_models()
        # Prefer 7b when UI still has a heavy model selected (common timeout cause).
        if requested and installed and (
            _is_heavy_model(requested) or _model_param_b(requested) >= 14
        ):
            for pref in ("qwen2.5-coder:7b", "qwen2.5:7b", "llama3.2:3b"):
                if pref in installed:
                    requested = pref
                    break
        chosen = set_active_model(requested)
        try:
            self.model_var.set(chosen)
            self.refresh_model_choices()
        except Exception:
            pass
        warn = last_model_warn()
        if warn:
            try:
                self.status_label.configure(text=warn)
            except Exception:
                pass
        return chosen

    def refresh_model_choices(self) -> None:
        try:
            from src.ollama_client import list_model_choices

            choices = list_model_choices()
            current = str(self.model_var.get() or getattr(config, "OLLAMA_MODEL", "")).strip()
            if current and current not in choices:
                choices = [current, *choices]
            self.model_menu.configure(values=choices)
            if current:
                self.model_var.set(current)
        except Exception:
            pass

    def _on_chat_return(self, _event: Any = None) -> str:
        self._on_ask_click()
        return "break"

    def _on_ask_click(self) -> None:
        if self._chat_busy:
            return
        q = str(self.chat_entry.get() or "").strip()
        if not q:
            q = "Which are the best bets on this card? Include stakes and edge stats."
        self.chat_entry.delete(0, "end")
        self.append_chat("You", q)
        if self._on_chat:
            self._on_chat(q)

    def _on_best_bets_click(self) -> None:
        if self._chat_busy:
            return
        q = "Which are the best bets on this card? Include stakes and edge stats."
        self.chat_entry.delete(0, "end")
        self.append_chat("You", q)
        if self._on_chat:
            self._on_chat(q)

    def append_chat(self, role: str, text: str) -> None:
        body = _ascii_ui(str(text or "").strip())
        if not body:
            return
        stamp = datetime.now().strftime("%H:%M")
        try:
            self.chat_log.configure(state="normal")
            self.chat_log.insert("end", f"\n[{stamp}] {role}\n{body}\n")
            self.chat_log.see("end")
            self.chat_log.configure(state="disabled")
        except Exception:
            pass

    def set_chat_busy(self, busy: bool) -> None:
        self._chat_busy = bool(busy)
        state = "disabled" if busy else "normal"
        try:
            self.ask_btn.configure(state=state)
            self.best_bets_btn.configure(state=state)
            self.chat_entry.configure(state=state)
        except Exception:
            pass

    def _cancel_think_tick(self) -> None:
        if self._think_after_id is not None:
            try:
                self.after_cancel(self._think_after_id)
            except Exception:
                pass
            self._think_after_id = None

    def _think_tick(self) -> None:
        if self._think_started_at is None:
            return
        elapsed = max(0, int(time.time() - self._think_started_at))
        timeout_s = int(getattr(config, "OLLAMA_TIMEOUT_SEC", 60) or 60)
        model = str(self.model_var.get() or getattr(config, "OLLAMA_MODEL", "ollama"))
        self.status_label.configure(
            text=f"Ollama is thinking... {elapsed}s / {timeout_s}s  |  {model}"
        )
        self._think_after_id = self.after(1000, self._think_tick)

    def set_busy(self, busy: bool, message: str = "") -> None:
        state = "disabled" if busy else "normal"
        try:
            self.run_btn.configure(state=state)
            self.model_menu.configure(state=state)
        except Exception:
            pass
        if busy:
            self.set_chat_busy(True)
        elif not self._chat_ask_busy:
            self.set_chat_busy(False)
        self._cancel_think_tick()
        if busy:
            self._think_started_at = time.time()
            try:
                self.progress.pack(fill="x", padx=12, pady=(0, 4), before=self.scroll)
                self.progress.start()
            except Exception:
                pass
            self.status_label.configure(text=message or "Ollama is thinking...")
            self._think_tick()
        else:
            self._think_started_at = None
            try:
                self.progress.stop()
                self.progress.pack_forget()
            except Exception:
                pass
            if message:
                self.status_label.configure(text=message)

    def render(self, result: dict[str, Any] | None, *, available: bool) -> None:
        for w in self.scroll.winfo_children():
            w.destroy()
        self.refresh_model_choices()
        if isinstance(result, dict):
            self._last_result = result

        health_banner = ""
        error_class = ""
        latency_ms = None
        if isinstance(result, dict):
            health_banner = str(result.get("health_banner") or "").strip()
            error_class = str(result.get("error_class") or "").strip()
            latency_ms = result.get("latency_ms")
            if result.get("ollama_latency_ms") is not None:
                latency_ms = result.get("ollama_latency_ms")

        if not available and not (
            result
            and (result.get("bet_slip") or result.get("recommended_parlays"))
        ):
            # Offline with no HA slip yet — clear banner, fail-closed (no invented bets)
            banner = health_banner or "Ollama offline — showing model tickets only"
            self.status_label.configure(text=banner)
            self._pack_message(banner, "Run Refresh Next Two, then Run Ollama Analysis. "
                "Model tickets will still show when available; Ollama never invents bets.",
                color="#fbbf24")
            return

        if not result:
            self.status_label.configure(text="Not run yet")
            self._pack_message(
                "No Ollama analysis yet",
                "Click Run Ollama Analysis after Refresh Next Two. "
                "Output = HA tickets already gated + conf/odds sized (what to bet and how much).",
            )
            return

        cache_note = " (cached)" if result.get("from_cache") else ""
        model_name = result.get("model") or getattr(config, "OLLAMA_MODEL", "ollama")
        profile = str(result.get("profile") or config.UFC_PROFILE or "paper").upper()
        latency_note = f"  |  {int(latency_ms)}ms" if latency_ms is not None else ""
        class_note = f"  |  {error_class}" if error_class and error_class != "ok" else ""
        self.status_label.configure(
            text=f"{result.get('generated_at', '-')}{cache_note}  |  {profile}  |  {model_name}"
            f"{latency_note}{class_note}"
        )

        if health_banner and (
            not result.get("ok")
            or result.get("narrative_degraded")
            or error_class in {"offline", "timeout", "model_missing", "disabled"}
            or str(result.get("ollama_error_class") or "")
            in {"offline", "timeout", "model_missing", "disabled"}
        ):
            self._pack_message(health_banner, "", color="#fbbf24", title_size=13)

        if not result.get("ok") and not result.get("bet_slip") and result.get("error"):
            self.status_label.configure(text="Last run failed")
            err = _format_grok_user_error(result.get("error"))
            self._pack_message("Ollama error", err, color="#f87171")
            if result.get("no_usable_odds") or (
                "no usable odds" in str(result.get("summary") or "").lower()
            ):
                self._pack_message(
                    "NO BET",
                    "NO BET — no usable odds (fail-closed)",
                    color="#fbbf24",
                )
            return

        if result.get("ok") is False and result.get("error") and not health_banner:
            self._pack_message(
                "Ollama warning",
                _format_grok_user_error(result.get("error"))
                + " Showing HA-sized slip with model reasons.",
                color="#fbbf24",
            )

        summary = str(result.get("summary") or "").strip()
        summary_is_no_bet = "NO BET" in summary.upper()
        if summary and not summary_is_no_bet:
            self._pack_message("Card summary", summary, color="#e2e8f0", title_size=13)

        # Always show Top 5 structure warning when there is a slip
        slip = list(result.get("bet_slip") or [])
        try:
            from src.bet_slip import dedupe_rank_top_tickets, top_recommended_label

            slip = dedupe_rank_top_tickets(slip, limit=5)
            result["bet_slip"] = slip
            try:
                self.scroll.configure(label_text=top_recommended_label(len(slip), limit=5))
            except Exception:
                pass
        except Exception:
            slip = slip[:5]
        warn = str(result.get("top5_warning") or "").strip()
        if slip and warn:
            n_act = result.get("n_actionable")
            n_adv = result.get("n_advisory")
            counts = ""
            if n_act is not None or n_adv is not None:
                counts = f" ({int(n_act or 0)} actionable / {int(n_adv or 0)} advisory)"
            self._pack_message(
                f"Warning{counts}",
                warn,
                color="#fbbf24",
                title_size=12,
            )

        # Budget line — auto card SSO T (never leftover $12 pool wording)
        try:
            from src.strategy import format_card_allocation_status

            br = float(result.get("bankroll") or getattr(config, "DEFAULT_TOTAL_BANKROLL", 100))
            auto = float(
                config.default_card_budget_usd(
                    br, profile=str(result.get("profile") or config.UFC_PROFILE)
                )
            )
            card_disp = float(result.get("card_budget") or auto)
            allocated = sum(float(b.get("stake_usd") or b.get("suggested_stake") or 0) for b in slip)
            if result.get("total_stake_usd") is not None:
                allocated = float(result.get("total_stake_usd") or allocated)
            if abs(card_disp - auto) < 0.02:
                budget_line = format_card_allocation_status(
                    auto_card_usd=auto,
                    allocated_usd=allocated,
                    n_tickets=len(slip),
                    overridden=False,
                )
            else:
                budget_line = format_card_allocation_status(
                    auto_card_usd=auto,
                    allocated_usd=allocated,
                    n_tickets=len(slip),
                    overridden=True,
                    card_budget_usd=card_disp,
                )
            self._pack_message("Sizing", _ascii_ui(budget_line), color="#94a3b8", title_size=12)
        except Exception:
            br = result.get("bankroll")
            card = result.get("card_budget")
            total_pct = result.get("total_stake_pct")
            total_usd = result.get("total_stake_usd")
            budget_bits = []
            if card is not None:
                budget_bits.append(f"Auto card ${float(card):.2f}")
            if total_pct is not None and total_usd is not None:
                budget_bits.append(
                    f"Allocated ${float(total_usd):.2f} ({float(total_pct):.0f}%)"
                )
            budget_bits.append(f"Tickets {len(slip)}")
            if budget_bits:
                self._pack_message(
                    "Sizing", " · ".join(budget_bits), color="#94a3b8", title_size=12
                )

        if result.get("no_bet") and not slip:
            no_bet_body = summary if (
                summary and "NO BET" in summary.upper()
            ) else (
                "NO BET — no usable odds (fail-closed)"
                if result.get("no_usable_odds")
                else "Nothing cleared HA gates for this card (fail-closed on missing odds / high uncertainty)."
            )
            self._pack_message("NO BET", no_bet_body, color="#fbbf24")
        elif slip:
            self._render_bet_slip_bubble(slip)
            actionable = [
                b for b in slip if not b.get("advisory") and not b.get("fun_bet")
            ]
            sum_pct = sum(float(b.get("stake_pct") or 0) for b in actionable)
            sum_usd = sum(float(b.get("stake_usd") or 0) for b in actionable)
            if actionable:
                self._pack_message(
                    "Card total (actionable only)",
                    f"{sum_pct:.1f}% of card  ·  ${sum_usd:.2f}",
                    color="#86efac",
                    title_size=13,
                )
            if result.get("no_bet") and not actionable:
                self._pack_message(
                    "Sized bankroll",
                    "WHAT TO BET (sized): NONE. Any Green lines above are FUN ONLY ($0) — not bankroll tickets.",
                    color="#fbbf24",
                    title_size=12,
                )

        self._render_auto_parlay_recs(list(result.get("recommended_parlays") or []))

        skips = list(result.get("skipped") or [])
        if skips:
            lines = [
                f"* {s.get('pick') or '-'} | {s.get('fight') or '-'} — {s.get('skip_reason') or 'skipped'}"
                for s in skips[:8]
            ]
            self._pack_message("Skips", "\n".join(lines), color="#f87171", title_size=12)

    def _render_auto_parlay_recs(self, parlays: list[dict[str, Any]]) -> None:
        """Show auto 2-leg + 3-leg parlays in the same bubble style as Top picks."""
        from src.bet_tiers import TIER_COLORS, TIER_GREEN, TIER_LABELS, TIER_YELLOW

        if not parlays:
            return
        by_n: dict[int, dict[str, Any]] = {}
        for p in parlays:
            try:
                n = int(p.get("n_legs") or 0)
            except (TypeError, ValueError):
                continue
            if n in (2, 3) and n not in by_n:
                by_n[n] = p
        ordered = [by_n[n] for n in (2, 3) if n in by_n] or list(parlays)[:2]
        if not ordered:
            return

        bubble = ctk.CTkFrame(
            self.scroll,
            fg_color="#0f172a",
            corner_radius=10,
            border_width=2,
            border_color="#475569",
        )
        bubble.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(
            bubble,
            text=f"Parlay recommendations ({len(ordered)})",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f8fafc",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(
            bubble,
            text="Auto 2-leg + 3-leg · research $0 (not HA-sized) · Green = stronger legs / Yellow = thinner",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

        for i, p in enumerate(ordered, start=1):
            n = int(p.get("n_legs") or 0)
            side = str(p.get("picks") or p.get("pick_line") or p.get("display_label") or "-")
            market = f"{n}-leg parlay" if n else str(p.get("market") or "parlay")
            comb = p.get("combined_prob")
            try:
                comb_s = f"{float(comb):.0%}" if comb is not None else "n/a"
            except (TypeError, ValueError):
                comb_s = "n/a"
            # Stronger presentation when HA-qualified 2-leg or high combined prob
            if p.get("ha_qualified") or (comb is not None and float(comb) >= 0.40):
                bet_tier = TIER_GREEN
            else:
                bet_tier = TIER_YELLOW
            color = TIER_COLORS.get(bet_tier, "#e2e8f0")
            tier_lbl = TIER_LABELS.get(bet_tier, bet_tier).upper()
            tag = "HA legs" if p.get("ha_qualified") else "research $0"
            line = (
                f"#{i}  [{tier_lbl}]  {side}  ·  {market}  ·  "
                f"combined {comb_s}  ·  {tag}"
            )
            ctk.CTkLabel(
                bubble,
                text=_ascii_ui(line),
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=color,
                anchor="w",
                justify="left",
                wraplength=980,
            ).pack(fill="x", padx=12, pady=(0, 2))
            reason = str(p.get("reason") or "").strip()
            if reason:
                # Match actionable density: one muted reason under the pick line
                ctk.CTkLabel(
                    bubble,
                    text=_ascii_ui(reason),
                    font=ctk.CTkFont(size=11),
                    text_color="#94a3b8",
                    anchor="w",
                    justify="left",
                    wraplength=960,
                ).pack(fill="x", padx=24, pady=(0, 6))

        ctk.CTkFrame(bubble, fg_color="transparent", height=8).pack(fill="x")

    def _pack_message(
        self,
        title: str,
        body: str,
        *,
        color: str = "#94a3b8",
        title_size: int = 14,
    ) -> None:
        frame = ctk.CTkFrame(self.scroll, fg_color="#1e293b", corner_radius=8)
        frame.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(
            frame,
            text=_ascii_ui(title),
            font=ctk.CTkFont(size=title_size, weight="bold"),
            text_color="#f1f5f9",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4 if body else 10))
        if body:
            ctk.CTkLabel(
                frame,
                text=_ascii_ui(body),
                font=ctk.CTkFont(size=12),
                text_color=color,
                anchor="w",
                justify="left",
                wraplength=1000,
            ).pack(fill="x", padx=12, pady=(0, 10))

    def _render_bet_slip_bubble(self, slip: list[dict[str, Any]]) -> None:
        """All recommended picks in one bubble (tier-colored lines)."""
        from src.bet_slip import dedupe_rank_top_tickets, top_recommended_label
        from src.bet_tiers import (
            TIER_COLORS,
            TIER_LABELS,
            action_label_for_bet,
            format_tier_legend,
            format_what_to_do_header,
        )

        # Fail-safe: never show duplicates / >5 even if upstream missed a merge
        slip = dedupe_rank_top_tickets(list(slip or []), limit=5)

        bubble = ctk.CTkFrame(
            self.scroll,
            fg_color="#0f172a",
            corner_radius=10,
            border_width=2,
            border_color="#475569",
        )
        bubble.pack(fill="x", padx=4, pady=6)
        ctk.CTkLabel(
            bubble,
            text=top_recommended_label(len(slip), limit=5),
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#f8fafc",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(
            bubble,
            text=_ascii_ui(format_what_to_do_header(slip=slip)),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#fde68a",
            anchor="w",
            justify="left",
            wraplength=980,
        ).pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            bubble,
            text=format_tier_legend(),
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            anchor="w",
            justify="left",
            wraplength=980,
        ).pack(fill="x", padx=12, pady=(0, 8))

        for i, bet in enumerate(slip, start=1):
            rank = bet.get("rank") or i
            side = str(bet.get("side") or bet.get("pick_line") or bet.get("pick") or "—")
            market = str(bet.get("market") or "moneyline")
            book = str(bet.get("book") or "n/a")
            edge = bet.get("edge_pct")
            if edge is None and bet.get("edge") is not None:
                try:
                    edge = float(bet.get("edge")) * 100.0
                except (TypeError, ValueError):
                    edge = None
            bet_tier = str(bet.get("bet_tier") or "").strip().lower()
            dollars = float(bet.get("stake_usd") or 0)
            if bet_tier not in TIER_COLORS:
                if bet.get("fun_bet") or bet.get("advisory"):
                    bet_tier = "green"
                elif dollars > 0:
                    bet_tier = "blue"
                else:
                    bet_tier = "yellow"
            color = TIER_COLORS.get(bet_tier, "#e2e8f0")
            action = action_label_for_bet({**bet, "bet_tier": bet_tier})
            edge_s = f"{float(edge):+.1f}%" if edge is not None else "n/a"
            line = (
                f"#{rank}  {action}  ·  {side}  ·  {market} @ {book}  ·  edge {edge_s}"
            )
            ctk.CTkLabel(
                bubble,
                text=_ascii_ui(line),
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=color,
                anchor="w",
                justify="left",
                wraplength=980,
            ).pack(fill="x", padx=12, pady=(0, 3))

        ctk.CTkFrame(bubble, fg_color="transparent", height=8).pack(fill="x")

    def _render_bet_slip_row(self, bet: dict[str, Any]) -> None:
        """Legacy single-row path — fold into the shared bubble."""
        self._render_bet_slip_bubble([bet])

    def _render_pick_card(self, pick: dict[str, Any]) -> None:
        """Legacy narrative card — maps into bet-slip bubble when possible."""
        self._render_bet_slip_bubble(
            [
                {
                    "rank": pick.get("rank") or "",
                    "side": pick.get("side") or pick.get("id") or pick.get("pick") or "Pick",
                    "market": pick.get("market") or pick.get("pick_type") or "moneyline",
                    "book": pick.get("book") or "n/a",
                    "stake_pct": pick.get("stake_pct") or 0,
                    "stake_usd": pick.get("stake_usd") or 0,
                    "reason": pick.get("reason") or pick.get("narrative_edge") or "",
                    "confidence": pick.get("conviction") or "",
                    "odds_display": pick.get("odds_display") or "",
                }
            ]
        )

class BookTab(_CTK_FRAME):
    """Reusable betting book tab layout."""

    _WARN_PACK = {"fill": "x", "padx": 12, "pady": (6, 0)}
    _STAKE_PACK = {"fill": "x", "padx": 12, "pady": (0, 2)}

    def __init__(self, master, title: str, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.title = title
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Full-page scroll so many fights + singles never clip when the window is short.
        self.page = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.page.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.page.grid_columnconfigure(0, weight=1)

        self.warning_box = ctk.CTkLabel(
            self.page,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#fbbf24",
            wraplength=1100,
        )
        self.warning_box.pack_forget()
        self.summary = ctk.CTkLabel(
            self.page,
            text="Run Refresh to load data.",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#cbd5e1",
        )
        self.summary.pack(fill="x", padx=12, pady=(6, 2))
        self.stake_box = ctk.CTkLabel(
            self.page,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=11),
            text_color="#93c5fd",
        )
        self.stake_box.pack_forget()
        self.fights_area = ctk.CTkFrame(
            self.page,
            fg_color="transparent",
        )
        self.fights_area.pack(fill="x", expand=False, padx=10, pady=2)
        self.bets_frame = ctk.CTkFrame(
            self.page,
            fg_color="transparent",
        )
        self.bets_frame.pack(fill="x", expand=False, padx=10, pady=(2, 8))

    def render(
        self,
        book_data: dict[str, Any],
        threshold_ctx: dict[str, Any],
        *,
        budget_state: dict[str, Any] | None = None,
        profile: str | None = None,
    ) -> None:
        try:
            self._render_inner(book_data, threshold_ctx, budget_state=budget_state, profile=profile)
        except Exception as exc:
            tb = traceback.format_exc()
            _debug_log(f"BookTab [{self.title}] render error: {tb}")
            try:
                import logging

                logging.getLogger("ufc_dashboard").error(
                    "BookTab [%s] render failed: %s\n%s", self.title, exc, tb
                )
            except Exception:
                pass
            self.summary.configure(text=f"{self.title} tab error: {exc}")
            for w in list(self.fights_area.winfo_children()):
                try:
                    w.destroy()
                except Exception:
                    pass
            for w in list(self.bets_frame.winfo_children()):
                try:
                    w.destroy()
                except Exception:
                    pass

    def _render_inner(
        self,
        book_data: dict[str, Any],
        threshold_ctx: dict[str, Any],
        *,
        budget_state: dict[str, Any] | None = None,
        profile: str | None = None,
    ) -> None:
        from src.strategy import (
            allocate_card_budget_per_book,
            book_display_name,
            budget_aware_alerts,
            collect_dashboard_risk_warnings,
        )

        preds: pd.DataFrame = book_data.get("predictions", pd.DataFrame())
        alerts: dict = budget_aware_alerts(
            book_data.get("alerts") or {},
            budget_state,
            self.title,
            profile=profile,
        )
        matched = book_data.get("odds_matched", 0)
        total = book_data.get("odds_total", 0)
        source = book_data.get("source", self.title)
        warning = book_data.get("warning") or book_data.get("error") or ""

        bankroll = float(
            (budget_state or {}).get("total_bankroll")
            or alerts.get("bankroll")
            or config.INITIAL_BANKROLL
        )
        strategy = strategy_from_profile(bankroll=bankroll)

        risk_warnings = collect_dashboard_risk_warnings(alerts, budget_state, bankroll=bankroll)
        if warning:
            risk_warnings.insert(0, ("warn", warning))
        _apply_risk_warning_label(
            self.warning_box,
            risk_warnings,
            pack={**self._WARN_PACK, "before": self.summary},
        )

        book_pool = 0.0
        alloc_line = ""
        if budget_state:
            from src.strategy import resolve_display_card_budget, available_card_budget_usd

            plan = allocate_card_budget_per_book(budget_state, profile=profile)
            info = plan.get(self.title, {})
            card_disp, overridden = resolve_display_card_budget(
                budget_state, profile=profile
            )
            card_label = "Card" if overridden else "Auto card"
            if self.title == "Odds API":
                book_pool = float(available_card_budget_usd(budget_state, profile=profile) or 0)
                if book_pool <= 0:
                    book_pool = float(card_disp or 0)
                alloc_line = (
                    f"{card_label} ${card_disp:.2f}  |  "
                    f"Odds API pool ${book_pool:.2f}  |  "
                    f"source the_odds_api"
                )
            elif info.get("enabled"):
                book_pool = float(info.get("allocation") or 0)
                alloc_line = (
                    f"{card_label} ${card_disp:.2f}  |  "
                    f"This book ${book_pool:.2f} "
                    f"({float(info.get('share_pct') or 0):.0f}%)  |  "
                    f"Balance ${float(info.get('balance') or 0):.2f}"
                )
            else:
                alloc_line = f"{book_display_name(self.title)} disabled in book toggles (Advanced)."

        if alloc_line:
            self.stake_box.configure(text=_ascii_ui(alloc_line))
            self.stake_box.pack(**self._STAKE_PACK, after=self.summary)
        else:
            self.stake_box.pack_forget()

        if int(matched or 0) == 0 and warning:
            # Hard fail-closed (auth/quota/no API events) → empty actionable state.
            # Roster mismatch still shows the card fights with a warning banner.
            hard_fail = (
                "NO BET" in str(warning)
                or "key rejected" in str(warning).lower()
                or "quota exhausted" in str(warning).lower()
                or "returned no events" in str(warning).lower()
                or "set THE_ODDS_API_KEY" in str(warning)
            ) and "lines matched" not in str(warning).lower()
            if hard_fail:
                summary_line = _ascii_ui(str(warning))
                self.summary.configure(text=summary_line)
                for w in list(self.fights_area.winfo_children()):
                    try:
                        w.destroy()
                    except Exception:
                        pass
                for w in list(self.bets_frame.winfo_children()):
                    try:
                        w.destroy()
                    except Exception:
                        pass
                empty = ctk.CTkLabel(
                    self.bets_frame,
                    text=_ascii_ui(
                        warning
                        if "NO BET" in str(warning)
                        else "NO BET — no usable odds (fail-closed)"
                    ),
                    anchor="w",
                    justify="left",
                    text_color="#fbbf24",
                    font=ctk.CTkFont(size=13, weight="bold"),
                    wraplength=1000,
                )
                empty.pack(fill="x", padx=8, pady=12)
                return

        match_meta = book_data.get("odds_match_meta") or {}
        if match_meta.get("status_line") and self.title == "Odds API":
            summary_line = (
                f"{match_meta.get('status_line')}  |  "
                f"{len(alerts.get('singles') or [])} singles  |  "
                f"source {source or 'the_odds_api'}"
            )
            rem = match_meta.get("requests_remaining")
            if rem is not None and str(rem) != "":
                summary_line += f"  |  credits remaining={rem}"
        else:
            if int(matched or 0) == 0:
                summary_line = (
                    f"{source}  |  no lines ({matched}/{total})  |  "
                    f"{len(alerts.get('singles') or [])} singles  |  "
                    f"{len(alerts.get('parlays') or [])} parlays"
                )
            else:
                summary_line = (
                    f"{source}  |  Odds {matched}/{total}  |  "
                    f"{len(alerts.get('singles') or [])} singles  |  "
                    f"{len(alerts.get('parlays') or [])} parlays"
                )

        skipped_n = int(alerts.get("skipped_count") or len(alerts.get("skipped") or []))
        if skipped_n:
            summary_line += f"  |  Unc skips {skipped_n}"
        dl = alerts.get("decision_layer") or {}
        td = threshold_ctx.get("thresholds") or alerts.get("threshold_detail")
        if dl:
            summary_line += (
                f"  |  Decision: edge>={100*float(dl.get('min_edge') or 0):.1f}% "
                f"prob>={100*float(dl.get('singles_min_model_prob') or 0):.0f}% "
                f"conf>={dl.get('singles_min_confidence') or '?'} "
                f"max={dl.get('max_tickets_per_card') or dl.get('max_bets_per_card')}/card "
                f"2-leg | Over1.5 only | RR=off"
            )
            unc = dl.get("uncertainty") or {}
            if unc:
                summary_line += (
                    f"  |  Unc skip w>={float(unc.get('interval_width_skip') or 0):.2f}"
                )
            if dl.get("strategy_line"):
                summary_line += f"\n{_ascii_ui(str(dl['strategy_line']))}"
        elif td:
            summary_line += f"  |  Min edge {td.get('alert_min_edge', 0):.1%}"
        elif config.is_live_profile() or config.is_paper_profile():
            summary_line += f"  |  Min edge {config.ALERT_MIN_EDGE:.1%}"

        self.summary.configure(text=_ascii_ui(summary_line))

        cards = book_data.get("cards") or []
        payload_stub = book_data.get("_payload")
        _render_grouped_fight_tables(
            self.fights_area,
            cards,
            preds,
            bankroll=bankroll,
            strategy=strategy,
            compact=True,
            table_height=11,
            payload=payload_stub,
            strip_unmatched_odds=True,
        )

        for w in list(self.bets_frame.winfo_children()):
            try:
                w.destroy()
            except Exception:
                pass
        singles = alerts.get("singles") or []
        max_singles = config.profile_int("max_singles_show")
        if singles:
            singles_box = ctk.CTkFrame(self.bets_frame, fg_color="transparent")
            singles_box.pack(fill="x", pady=(0, 6))
            _render_ranked_singles(
                singles_box,
                singles[:max_singles],
                preds=preds,
                title=f"Top singles ({source})",
            )
        skipped = alerts.get("skipped") or []
        if skipped:
            skip_box = ctk.CTkFrame(self.bets_frame, fg_color="transparent")
            skip_box.pack(fill="x", pady=(0, 6))
            _render_uncertainty_skips(skip_box, skipped)
            try:
                _render_skip_scorecard_panel(skip_box, None, title="Skip scorecard (7d)")
            except Exception:
                pass
        parlays = alerts.get("parlays") or []
        max_parlays = config.profile_int("max_parlays_show")
        _render_ranked_parlays(
            self.bets_frame,
            parlays[:max_parlays],
            preds=preds,
            title="Parlays" if parlays else "",
        )


class PropsTable(_CTK_FRAME):
    """Prop singles table: type+fight, fighter, odds, source, edge, min bet $."""

    COLUMNS = ("Prop Type", "Fighter", "Odds", "Source", "Edge", "Min Bet")
    _NUMERIC_COLS = frozenset({"Odds", "Edge", "Min Bet"})

    def __init__(self, master, *, height: int = 14, book_name: str = "", **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.book_name = book_name
        self._rows_data: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
        self._sort_col: str | None = None
        self._sort_desc = True
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Props.Treeview",
            background="#1e1e1e",
            foreground="#e8e8e8",
            fieldbackground="#1e1e1e",
            rowheight=30,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Props.Treeview.Heading",
            background="#2b2b2b",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(6, 4),
        )
        self.tree = ttk.Treeview(
            self,
            columns=self.COLUMNS,
            show="headings",
            height=height,
            style="Props.Treeview",
        )
        widths = (260, 120, 110, 88, 88, 100)
        for col, w in zip(self.COLUMNS, widths):
            self.tree.heading(
                col,
                text=col,
                anchor="w",
                command=lambda c=col: self._on_heading_click(c),
            )
            stretch = col == "Prop Type"
            self.tree.column(col, width=w, minwidth=52, anchor="w", stretch=stretch)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.tree.tag_configure("pos", foreground="#34d399")
        self.tree.tag_configure("neg", foreground="#f87171")
        self.tree.tag_configure("neutral", foreground="#b0b0b0")
        self.tree.tag_configure("synth", foreground="#94a3b8")
        self.tree.tag_configure("relaxed", foreground="#94a3b8")
        self.tree.tag_configure("even", background="#1a1f2e")
        self.tree.tag_configure("odd", background="#1e1e1e")
        self.tree.bind("<MouseWheel>", self._on_wheel)
        self.tree.bind("<Enter>", lambda _e: self.tree.focus_set())
        self.tree.bind(
            "<Leave>",
            lambda _e: self.master.focus_set() if hasattr(self.master, "focus_set") else None,
        )

    def _on_wheel(self, event) -> None:
        if event.delta:
            self.tree.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_heading_click(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            self._sort_desc = col in self._NUMERIC_COLS
        self._redraw_sorted()

    def _refresh_heading_labels(self) -> None:
        for c in self.COLUMNS:
            label = str(c)
            if c == self._sort_col:
                label = f"{c} {'v' if self._sort_desc else '^'}"
            self.tree.heading(
                c,
                text=_ascii_ui(label),
                anchor="w",
                command=lambda col=c: self._on_heading_click(col),
            )

    def _redraw_sorted(self) -> None:
        self._refresh_heading_labels()
        self.tree.delete(*self.tree.get_children())
        rows = list(self._rows_data)
        if self._sort_col and rows:
            try:
                idx = list(self.COLUMNS).index(self._sort_col)
            except ValueError:
                idx = -1
            if idx >= 0:
                numeric = self._sort_col in self._NUMERIC_COLS
                rows.sort(
                    key=lambda item: DataTable._cell_sort_key(
                        item[0][idx] if idx < len(item[0]) else "",
                        numeric=numeric,
                    ),
                    reverse=self._sort_desc,
                )
        for i, (values, tags) in enumerate(rows):
            edge_tag = tags[0] if tags else "neutral"
            zebra = "even" if i % 2 == 0 else "odd"
            self.tree.insert("", "end", values=_ascii_row(values), tags=(edge_tag, zebra))

    @staticmethod
    def _format_odds(s: dict[str, Any], book_name: str) -> str:
        from src.parlay_builder import decimal_to_american

        dec = float(s.get("odds", 0) or 0)
        if dec <= 1:
            return "—"
        am = decimal_to_american(dec)
        return f"{am} ({dec:.2f})"

    @staticmethod
    def _format_source(s: dict[str, Any], book_name: str = "") -> str:
        source = str(
            s.get("source_label") or s.get("odds_source") or "synthetic"
        ).strip().lower()
        if source in {"synthetic", "", "model"}:
            return "Synthetic"
        # Props tab is book-scoped — never label another book's lines as this tab.
        if book_name:
            return str(book_name).replace(".eu", "")
        if source in {"live", "the_odds_api"}:
            return "Odds API" if source == "the_odds_api" else "Live"
        if source in {"mybookie", "draftkings", "betnow", "betnow.eu"}:
            return source.replace("betnow.eu", "BetNow").title()
        return source.title() if source else "Synthetic"

    def load_singles(self, singles: list[dict[str, Any]]) -> None:
        self.tree.delete(*self.tree.get_children())
        self._rows_data = []
        if not singles:
            self._sort_col = None
            self._refresh_heading_labels()
            self.tree.insert(
                "",
                "end",
                values=_ascii_row(("No props match current filters", "-", "—", "—", "—", "—")),
                tags=("neutral", "even"),
            )
            return
        for s in singles:
            fight = str(s.get("fight", ""))
            prop_title = str(
                s.get("prop_short") or s.get("prop_type") or s.get("prop_key", "")
            ).strip()
            if fight:
                prop_cell = f"{prop_title} - {fight}"
            else:
                prop_cell = prop_title

            fighter = str(s.get("fighter", "-"))
            if fighter in ("", "-"):
                fighter = "-"

            source = str(s.get("odds_source", "synthetic")).lower()
            source_txt = self._format_source(s, self.book_name)
            edge_pct = s.get("edge_pct")
            edge_txt = "—"
            edge_tag = "neutral"
            # Live edge only when live odds exist — never invent edge on synthetic
            if source in {"live", "the_odds_api"} and edge_pct is not None:
                try:
                    ep = float(edge_pct)
                    if abs(ep) <= 1.0:
                        ep = ep * 100.0
                    if abs(ep) > _MAX_DISPLAY_EDGE * 100.0:
                        edge_txt = "—"
                        edge_tag = "neutral"
                    else:
                        edge_txt = f"{ep:+.1f}%"
                        edge_tag = (
                            "pos" if ep > 0 else "neg" if ep < 0 else "neutral"
                        )
                except (TypeError, ValueError):
                    edge_txt = "—"
            elif source not in {"live", "the_odds_api"}:
                edge_txt = f"{float(s.get('prob', 0)):.0%} model"
                edge_tag = "synth"

            if s.get("book_disabled"):
                stake_txt = "Book off"
            else:
                stake_txt = _format_min_bet_plain(
                    s.get("suggested_stake") or s.get("stake_usd") or 0
                )

            row = (
                prop_cell,
                fighter,
                self._format_odds(s, self.book_name),
                source_txt,
                edge_txt,
                stake_txt,
            )
            self._rows_data.append((row, (edge_tag,)))
        if self._sort_col is None:
            self._sort_col = "Edge"
            self._sort_desc = True
        self._redraw_sorted()


class ArbScannerTab(_CTK_FRAME):
    """Cross-book moneyline and totals arb opportunities with optional live watch."""

    COLUMNS = (
        "Market",
        "Fight",
        "Side A",
        "Book A",
        "Side B",
        "Book B",
        "Margin",
        "Stakes",
    )

    def __init__(
        self,
        master,
        *,
        payload_getter: Callable[[], "DashboardPayload | None"] | None = None,
        budget_getter: Callable[[], dict[str, Any] | None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._payload_getter = payload_getter
        self._budget_getter = budget_getter
        self._seen_alert_keys: set[str] = set()
        self._poll_after_id: str | None = None
        self._poll_busy = False
        self._last_payload: "DashboardPayload | None" = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self.summary = ctk.CTkLabel(
            self,
            text="Cross-book arb scan - run Refresh Next Two or Quick Odds + Props.",
            anchor="w",
            justify="left",
            wraplength=1100,
        )
        self.summary.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))

        self.detail = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=11),
            text_color="#64748b",
            wraplength=1100,
        )
        self.detail.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))

        self.controls = ctk.CTkFrame(self, fg_color="transparent")
        self.controls.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))

        self.watch_var = ctk.BooleanVar(value=True)
        self.watch_switch = ctk.CTkSwitch(
            self.controls,
            text="Watch for Arbs (DK vs MyBookie)",
            variable=self.watch_var,
            command=self._on_watch_toggle,
        )
        self.watch_switch.pack(side="left", padx=(0, 12))

        self.watch_status = ctk.CTkLabel(
            self.controls,
            text="",
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
        )
        self.watch_status.pack(side="left", fill="x", expand=True)

        self.alert_banner = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#34d399",
            fg_color="#14532d",
            corner_radius=6,
            wraplength=1100,
        )
        self.alert_banner.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))
        self.alert_banner.grid_remove()

        self.scroll = ctk.CTkScrollableFrame(self, label_text="Arb opportunities")
        self.scroll.grid(row=4, column=0, sticky="nsew", padx=10, pady=(2, 8))

        self._update_watch_status("Watch ON - waiting for card data")

    def reset_alerts(self) -> None:
        """Clear dedupe keys after a full card refresh."""
        self._seen_alert_keys.clear()

    def start_watch(self) -> None:
        if bool(self.watch_var.get()):
            self._schedule_poll(2_000)

    def stop_watch(self) -> None:
        self._cancel_poll()

    def _on_watch_toggle(self) -> None:
        if bool(self.watch_var.get()):
            self._update_watch_status("Watch ON - polling for arbs...")
            self._schedule_poll(500)
        else:
            self.stop_watch()
            self._update_watch_status("Watch OFF")
            self.alert_banner.grid_remove()

    def _update_watch_status(self, text: str) -> None:
        threshold = config.ARB_ALERT_THRESHOLD_PCT
        poll = config.ARB_ALERT_POLL_SEC
        self.watch_status.configure(
            text=f"{text}  |  alert >={threshold:.1f}% profit  |  poll every {poll}s"
        )

    def _cancel_poll(self) -> None:
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None

    def _schedule_poll(self, delay_ms: int) -> None:
        self._cancel_poll()
        self._poll_after_id = self.after(delay_ms, self._poll_tick)

    def _poll_tick(self) -> None:
        if not bool(self.watch_var.get()):
            return
        if not self._poll_busy:
            threading.Thread(target=self._poll_worker, daemon=True).start()
        delay_ms = max(30, int(config.ARB_ALERT_POLL_SEC)) * 1000
        self._poll_after_id = self.after(delay_ms, self._poll_tick)

    def _poll_worker(self) -> None:
        if self._poll_busy:
            return
        self._poll_busy = True
        try:
            payload = self._payload_getter() if self._payload_getter else None
            if payload is None or _df_is_empty(getattr(payload, "combined", None)):
                self.after(0, lambda: self._update_watch_status("Watch ON - load a card first"))
                return
            from src.arb_scanner import scan_cross_book_arbs

            scan = scan_cross_book_arbs(
                books=payload.books,
                combined=payload.combined,
                force_refresh=True,
                budget_state=self._budget_getter() if self._budget_getter else None,
            )
            self.after(0, lambda: self._on_poll_result(scan, payload))
        except Exception as exc:
            self.after(0, lambda: self._update_watch_status(f"Watch error: {exc}"))
        finally:
            self._poll_busy = False

    def _on_poll_result(self, scan: dict[str, Any], payload: "DashboardPayload") -> None:
        payload.arb_scan = scan
        self._last_payload = payload
        self.render(payload, from_poll=True)
        scanned = (scan.get("meta") or {}).get("scanned_at", "now")
        self._update_watch_status(f"Last check {scanned}")

    def _process_new_alerts(self, scan: dict[str, Any]) -> None:
        from src.arb_scanner import arb_row_alert_key, format_arb_alert_message, strong_arb_rows

        strong = strong_arb_rows(scan, dk_mybookie_only=True)
        if strong:
            top = strong[0]
            profit = float(top.get("profit_pct", 0))
            self.alert_banner.configure(
                text=f"Strong DK <-> MyBookie arb: {top.get('fight', '')} (+{profit:.2f}%)"
            )
            self.alert_banner.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))
        else:
            self.alert_banner.grid_remove()

        for row in strong:
            key = arb_row_alert_key(row)
            if key in self._seen_alert_keys:
                continue
            self._seen_alert_keys.add(key)
            self._show_arb_toast(format_arb_alert_message(row))

    def _show_arb_toast(self, message: str) -> None:
        if config.ARB_ALERT_SOUND:
            try:
                import winsound

                winsound.Beep(880, 160)
            except Exception:
                pass

        root = self.winfo_toplevel()
        toast = ctk.CTkToplevel(root)
        toast.title("Arb alert")
        toast.attributes("-topmost", True)
        toast.resizable(False, False)
        toast.configure(fg_color="#14532d")

        frame = ctk.CTkFrame(toast, fg_color="#14532d", corner_radius=8)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        ctk.CTkLabel(
            frame,
            text="Cross-book arb detected",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#bbf7d0",
        ).pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(
            frame,
            text=message,
            anchor="w",
            justify="left",
            wraplength=420,
            font=ctk.CTkFont(size=12),
            text_color="#ecfdf5",
        ).pack(anchor="w", padx=12, pady=(0, 10))
        ctk.CTkButton(
            frame,
            text="Dismiss",
            width=90,
            command=toast.destroy,
        ).pack(anchor="e", padx=12, pady=(0, 10))

        toast.update_idletasks()
        rw = int(root.winfo_width() or 900)
        tw = int(toast.winfo_width() or 460)
        x = int(root.winfo_rootx() + max(12, rw - tw - 24))
        y = int(root.winfo_rooty() + 72)
        toast.geometry(f"+{x}+{y}")
        toast.after(14_000, lambda: toast.destroy() if toast.winfo_exists() else None)

    def render(
        self,
        payload: "DashboardPayload | None",
        *,
        from_poll: bool = False,
    ) -> None:
        for w in self.scroll.winfo_children():
            w.destroy()

        if payload is None or not payload.books:
            self.summary.configure(text="No card loaded - click Refresh Next Two first.")
            self.detail.configure(text="")
            return

        self._last_payload = payload
        scan = payload.arb_scan or {}
        if not scan and not from_poll:
            try:
                from src.dashboard_service import _build_arb_scan

                scan = _build_arb_scan(
                    payload.books,
                    payload.combined,
                    budget_state=self._budget_getter() if self._budget_getter else None,
                )
                payload.arb_scan = scan
            except Exception as exc:
                self.summary.configure(text=f"Arb scan failed: {exc}")
                return

        if from_poll:
            self._process_new_alerts(scan)

        meta = scan.get("meta") or {}
        ml_rows = list(scan.get("moneyline") or [])
        prop_rows = list(scan.get("props") or [])
        all_rows = ml_rows + prop_rows
        threshold = config.ARB_ALERT_THRESHOLD_PCT
        from src.arb_scanner import strong_arb_rows

        strong_n = len(strong_arb_rows(scan, threshold_pct=threshold, dk_mybookie_only=True))
        true_n = int(meta.get("true_arb_count", sum(1 for r in all_rows if r.get("is_arb"))))
        near_n = int(meta.get("near_count", sum(1 for r in all_rows if r.get("is_near"))))
        books = ", ".join(meta.get("books_scanned") or []) or "-"
        stake = float(meta.get("stake_total") or config.ARB_STAKE_TOTAL)
        near_pct = float(meta.get("near_margin_pct") or config.ARB_NEAR_MARGIN_PCT)

        self.summary.configure(
            text=(
                f"Scanned {books}  |  {true_n} true arb(s)  |  {strong_n} strong DK<->MyBookie (>={threshold:.1f}%)  "
                f"|  {near_n} near-arb (<={near_pct:.0f}% overround)  "
                f"|  Updated {meta.get('scanned_at', payload.odds_updated_at or '-')}"
            )
        )
        err_txt = "; ".join(scan.get("errors") or [])
        if err_txt:
            self.detail.configure(text=f"Book warnings: {err_txt[:400]}")
        elif not all_rows:
            self.detail.configure(
                text=(
                    "No cross-book arbs on this card - markets are efficiently priced. "
                    f"Alerts fire when DK vs MyBookie profit >={threshold:.1f}%."
                )
            )
        else:
            self.detail.configure(
                text=(
                    f"Bright green rows = arb profit >={threshold:.1f}%. "
                    f"Amber = near-arb. Stakes assume ${stake:.0f} total."
                )
            )

        if not all_rows:
            frame = ctk.CTkFrame(self.scroll, fg_color="#1a1f2e", corner_radius=8)
            frame.pack(fill="x", padx=8, pady=12)
            ctk.CTkLabel(
                frame,
                text="No arb or near-arb lines found",
                anchor="w",
                text_color="#9ca3af",
                font=ctk.CTkFont(size=13),
            ).pack(fill="x", padx=14, pady=12)
            return

        table_frame = ctk.CTkFrame(self.scroll, fg_color="transparent")
        table_frame.pack(fill="x", padx=2, pady=(2, 6))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Arb.Treeview",
            background="#1e1e1e",
            foreground="#e8e8e8",
            fieldbackground="#1e1e1e",
            rowheight=30,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Arb.Treeview.Heading",
            background="#2b2b2b",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(6, 4),
        )
        tree = ttk.Treeview(
            table_frame,
            columns=self.COLUMNS,
            show="headings",
            height=min(16, len(all_rows) + 1),
            style="Arb.Treeview",
        )
        widths = (88, 220, 120, 92, 120, 92, 72, 130)
        for col, w in zip(self.COLUMNS, widths):
            tree.heading(col, text=col, anchor="w")
            stretch = col == "Fight"
            tree.column(col, width=w, minwidth=56, anchor="w", stretch=stretch)
        tree.pack(fill="x")
        tree.tag_configure("pos", foreground="#34d399")
        tree.tag_configure("strong", foreground="#ecfdf5", background="#166534")
        tree.tag_configure("synth", foreground="#94a3b8")
        tree.tag_configure("even", background="#1a1f2e")
        tree.tag_configure("odd", background="#1e1e1e")

        for i, row in enumerate(all_rows):
            market = "ML" if row.get("market") == "moneyline" else "O/U 1.5"
            sa = row.get("side_a") or {}
            sb = row.get("side_b") or {}
            profit = float(row.get("profit_pct", 0) or 0)
            if row.get("is_arb") and profit >= threshold:
                margin = f"+{profit:.2f}%"
                edge_tag = "strong"
            elif row.get("is_arb"):
                margin = f"+{profit:.2f}%"
                edge_tag = "pos"
            else:
                margin = f"{float(row.get('overround_pct', 0)):.1f}%"
                edge_tag = "synth"
            stakes = (
                f"{_format_min_bet(row.get('stake_a', 0))} / "
                f"{_format_min_bet(row.get('stake_b', 0))}"
            )
            values = (
                market,
                str(row.get("fight", "")),
                f"{sa.get('fighter', '')} {sa.get('american', '')}".strip(),
                str(sa.get("book", "")).replace(".eu", ""),
                f"{sb.get('fighter', '')} {sb.get('american', '')}".strip(),
                str(sb.get("book", "")).replace(".eu", ""),
                margin,
                stakes,
            )
            zebra = "even" if i % 2 == 0 else "odd"
            tree.insert("", "end", values=_ascii_row(values), tags=(edge_tag, zebra))

        if bool(self.watch_var.get()) and self._poll_after_id is None:
            self.start_watch()


class BookPropsTab(_CTK_FRAME):
    """Prop bets for one book - singles; DraftKings also shows parlays."""

    _BOOK_EMPTY_HINTS: dict[str, str] = {
        "Odds API": (
            "No Odds API prop markets returned. Set THE_ODDS_API_KEY and ENABLE_PROPS=true, "
            "then Soft Update. HA actionable props are Over 1.5 only (Odds API totals)."
        ),
        "BetNow.eu": (
            "No ranked props for BetNow yet. Set BETNOW_COOKIE in .env for live method/total lines, "
            "then Refresh Next Two. Synthetic props need >={min_prob:.0%} model probability."
        ),
        "DraftKings": (
            "No ranked props for DraftKings yet. Live totals and method markets load on Refresh when "
            "ENABLE_PROPS=true. Synthetic props need >={min_prob:.0%} model probability."
        ),
        "MyBookie": (
            "No ranked props for MyBookie yet. Enable MYBOOKIE_ENABLED=true and refresh for live lines. "
            "Synthetic props need >={min_prob:.0%} model probability."
        ),
    }

    _RISK_GRID = {"row": 1, "column": 0, "sticky": "ew", "padx": 12, "pady": (0, 4)}
    _BACKTEST_GRID = {"row": 4, "column": 0, "sticky": "ew", "padx": 12, "pady": (0, 4)}

    def __init__(
        self,
        master,
        *,
        book_name: str,
        book_note: str,
        show_parlays: bool = False,
        show_all_var: Any = None,
        profile_getter: Callable[[], str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self.book_name = book_name
        self.book_note = book_note
        self.show_parlays = show_parlays
        self.show_all_var = show_all_var
        self.profile_getter = profile_getter
        self._last_payload: DashboardPayload | None = None
        self._last_budget_state: dict[str, Any] | None = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        self.summary = ctk.CTkLabel(
            self,
            text="Props - click Refresh Next Two to load ranked prop lines.",
            anchor="w",
            justify="left",
        )
        self.summary.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        self.risk_warning_box = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#f87171",
            wraplength=1100,
        )
        self.risk_warning_box.grid_remove()
        self.filter_label = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#9ca3af",
        )
        self.filter_label.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))
        self.controls = ctk.CTkFrame(self, fg_color="transparent")
        self.controls.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 4))
        self.show_all_switch = ctk.CTkSwitch(
            self.controls,
            text="Show all props (relaxed)",
            variable=show_all_var if show_all_var is not None else ctk.BooleanVar(value=False),
            command=self._on_show_all_toggle,
        )
        self.show_all_hint = ctk.CTkLabel(
            self.controls,
            text=(
                "Strict filter shows props with live edge or high model confidence. "
                "Relaxed includes lower-confidence model lines for research."
            ),
            anchor="w",
            font=ctk.CTkFont(size=11),
            text_color="#64748b",
            wraplength=900,
            justify="left",
        )
        self.backtest_box = ctk.CTkLabel(
            self,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=11),
            text_color="#64748b",
        )
        self.backtest_box.grid_remove()
        self.scroll = ctk.CTkScrollableFrame(
            self,
            label_text="Ranked prop singles",
            height=420,
        )
        self.scroll.grid(row=5, column=0, sticky="nsew", padx=10, pady=(2, 6))

    def _on_show_all_toggle(self) -> None:
        print(f"Button [Show all props ({self.book_name})] clicked", flush=True)
        val = bool(self.show_all_var.get()) if self.show_all_var is not None else False
        _debug_log(f"Show all props toggled ({self.book_name}): {val}")
        if self._last_payload is not None:
            profile = self.profile_getter() if self.profile_getter else None
            self.render(self._last_payload, budget_state=self._last_budget_state, profile=profile)

    def _is_paper_profile(self) -> bool:
        if self.profile_getter:
            return config.normalize_profile(self.profile_getter()) == "paper"
        return config.is_paper_profile()

    def _update_show_all_visibility(self) -> None:
        if self._is_paper_profile():
            self.show_all_switch.pack(side="left", padx=(0, 8))
            self.show_all_hint.pack(side="left", fill="x", expand=True)
        else:
            self.show_all_switch.pack_forget()
            self.show_all_hint.pack_forget()
            if self.show_all_var is not None:
                self.show_all_var.set(False)

    def _update_show_all_controls(self, meta: dict[str, Any], shown: int) -> None:
        if not self._is_paper_profile():
            return
        total = int(meta.get("total_found", 0))
        strict = int(meta.get("strict_count", 0))
        relaxed = int(meta.get("relaxed_count", max(0, total - strict)))
        min_prob = config.PROP_MIN_MODEL_PROB
        relaxed_floor = config.PROP_SHOW_ALL_MIN_PROB
        if relaxed:
            self.show_all_switch.configure(
                text=f"Show all props (+{relaxed} relaxed)"
            )
        else:
            self.show_all_switch.configure(text="Show all props (relaxed)")
        show_all = bool(self.show_all_var.get()) if self.show_all_var is not None else False
        if show_all:
            self.show_all_hint.configure(
                text=(
                    f"Showing all {shown} ranked props - strict ({strict}) plus relaxed "
                    f"(model >={relaxed_floor:.0%}, below strict {min_prob:.0%}). "
                    "Stakes scaled to your auto card budget (bankroll × profile risk %)."
                ),
                text_color="#94a3b8",
            )
        else:
            self.show_all_hint.configure(
                text=(
                    f"Strict only: {strict} props with live edge >={config.PROP_MIN_EDGE:.0%} "
                    f"or model >={min_prob:.0%}. "
                    f"Toggle on to add {relaxed} relaxed research lines (>={relaxed_floor:.0%} model)."
                ),
                text_color="#64748b",
            )

    def _paper_display_cap(self) -> int:
        return min(36, int(config.PROP_MAX_RESULTS))

    def _filter_singles(self, singles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        show_all = bool(self.show_all_var.get()) if self.show_all_var is not None else False
        if show_all and self._is_paper_profile():
            return singles
        return [s for s in singles if s.get("strict_qualified", True)]

    def _render_props_table(self, singles: list[dict[str, Any]]) -> None:
        cap = self._paper_display_cap() if self._is_paper_profile() else min(12, config.PROP_MAX_RESULTS)
        display = singles[:cap]
        table = PropsTable(
            self.scroll,
            height=min(cap, max(6, len(display) + 1)),
            book_name=self.book_name,
        )
        table.pack(fill="x", padx=2, pady=(2, 6))
        table.load_singles(display)
        if len(singles) > len(display):
            ctk.CTkLabel(
                self.scroll,
                text=f"Showing top {len(display)} of {len(singles)} props (Paper cap {cap}).",
                anchor="w",
                text_color="#64748b",
                font=ctk.CTkFont(size=11),
            ).pack(fill="x", padx=4, pady=(0, 4))

    def _empty_state_hint(
        self,
        *,
        total_found: int,
        strict_count: int,
        book_warning: str,
        book_disabled: bool = False,
    ) -> str:
        if book_disabled:
            return (
                f"{self.book_name.replace('.eu', '')} is unchecked in Advanced book toggles - "
                "enable it to see prop stakes and recommendations for this book."
            )
        min_prob = config.PROP_MIN_MODEL_PROB
        relaxed_floor = config.PROP_SHOW_ALL_MIN_PROB
        show_all = bool(self.show_all_var.get()) if self.show_all_var is not None else False
        relaxed = max(0, total_found - strict_count)

        if total_found and not show_all and self._is_paper_profile() and relaxed:
            return (
                f"No strict props to display ({strict_count} pass >={min_prob:.0%} model / "
                f">={config.PROP_MIN_EDGE:.0%} live edge).\n\n"
                f"{relaxed} relaxed candidates (model >={relaxed_floor:.0%}) are hidden - "
                "turn on **Show all props (relaxed)** above."
            )
        if total_found and show_all:
            return (
                f"All {total_found} ranked props are below the table display cap - "
                "try Refresh Next Two after odds load."
            )

        template = self._BOOK_EMPTY_HINTS.get(
            self.book_name,
            "No ranked props for this book yet. Refresh Next Two after ENABLE_PROPS=true.",
        )
        hint = template.format(min_prob=min_prob)
        if book_warning:
            hint = f"{book_warning}\n\n{hint}"
        return hint

    def _pack_empty_state(self, title: str, body: str, *, color: str = "#9ca3af") -> None:
        frame = ctk.CTkFrame(self.scroll, fg_color="#1a1f2e", corner_radius=8)
        frame.pack(fill="x", padx=8, pady=12)
        ctk.CTkLabel(
            frame,
            text=title,
            anchor="w",
            text_color="#e2e8f0",
            font=ctk.CTkFont(size=14, weight="bold"),
            wraplength=1000,
            justify="left",
        ).pack(fill="x", padx=14, pady=(12, 4))
        ctk.CTkLabel(
            frame,
            text=body,
            anchor="w",
            text_color=color,
            wraplength=1000,
            justify="left",
            font=ctk.CTkFont(size=12),
        ).pack(fill="x", padx=14, pady=(0, 12))

    def _render_backtest(self, payload: "DashboardPayload") -> None:
        bt = payload.prop_backtest or {}
        if bt:
            roi = bt.get("roi_pct")
            acc = bt.get("acc_mean_prop_accuracy") or bt.get("mean_prop_accuracy")
            parts = []
            if roi is not None:
                parts.append(f"backtest prop ROI {float(roi):+.1f}%")
            if acc is not None:
                parts.append(f"mean prop accuracy {float(acc):.0%}")
            if parts:
                self.backtest_box.configure(text="Historical: " + "  |  ".join(parts))
                self.backtest_box.grid(**self._BACKTEST_GRID)
            else:
                self.backtest_box.configure(text="")
                self.backtest_box.grid_remove()
        else:
            self.backtest_box.configure(text="")
            self.backtest_box.grid_remove()

    def _format_single_line(self, s: dict[str, Any]) -> tuple[str, str]:
        """Legacy one-line formatter (parlay legs)."""
        from src.parlay_builder import decimal_to_american

        am = decimal_to_american(float(s.get("odds", 0) or 0))
        source = str(s.get("odds_source", "synthetic")).lower()
        if source in {"live", "the_odds_api"}:
            badge = "Odds API" if source == "the_odds_api" else "Live odds"
            badge_color = "#34d399"
        else:
            badge = "Synthetic"
            badge_color = "#fbbf24"
        edge_pct = s.get("edge_pct")
        if edge_pct is not None:
            edge_part = f"edge {float(edge_pct):+.1f}%"
        else:
            edge_part = f"model {float(s.get('prob', 0)):.0%}"
        text = (
            f"* {s.get('label', '')}  |  "
            f"prob {s.get('prob', 0):.0%}  |  "
            f"{am} ({s.get('odds', 0):.2f})  |  "
            f"{edge_part}  |  "
            f"{badge}"
        )
        line_color = badge_color if source in {"live", "the_odds_api"} else "#3dd68c"
        return text, line_color

    def render(
        self,
        payload: "DashboardPayload",
        *,
        budget_state: dict[str, Any] | None = None,
        profile: str | None = None,
    ) -> None:
        self._last_payload = payload
        self._last_budget_state = budget_state
        _button_debug(f"Loading props for {self.book_name}")
        for w in self.scroll.winfo_children():
            w.destroy()
        self._update_show_all_visibility()

        overview = payload.books.get("Overview", {})
        alerts = overview.get("alerts") or {}
        from src.strategy import attach_prop_stakes, collect_dashboard_risk_warnings

        risk_warnings = collect_dashboard_risk_warnings(alerts, budget_state)
        _apply_risk_warning_label(self.risk_warning_box, risk_warnings, grid=self._RISK_GRID)

        props_on = _ensure_props_config()
        if not props_on:
            self.summary.configure(text="Prop betting disabled - set ENABLE_PROPS=true in .env")
            self.filter_label.configure(text="")
            self.backtest_box.configure(text="")
            self._pack_empty_state(
                "Props disabled",
                "Add ENABLE_PROPS=true to .env, then click Refresh Next Two "
                "to load method and total markets from each book.",
                color="#fbbf24",
            )
            return

        if self.book_name == "MyBookie" and not config.MYBOOKIE_ENABLED:
            self.summary.configure(text="MyBookie disabled - set MYBOOKIE_ENABLED=true in .env")
            self.backtest_box.configure(text="")
            self._pack_empty_state(
                "MyBookie props unavailable",
                "Set MYBOOKIE_ENABLED=true in .env and refresh to fetch live method/total lines.",
            )
            return

        if self.book_name == "Odds API" and not str(getattr(config, "ODDS_API_KEY", "") or "").strip():
            self.summary.configure(text="Odds API props unavailable — set THE_ODDS_API_KEY")
            self.backtest_box.configure(text="")
            self._pack_empty_state(
                "NO BET — no usable odds (fail-closed)",
                "Set THE_ODDS_API_KEY in .env, then Soft Update to load Over 1.5 markets.",
                color="#fbbf24",
            )
            return

        book_disabled = False
        pool_line = ""
        if budget_state:
            from src.strategy import (
                allocate_card_budget_per_book,
                available_card_budget_usd,
                book_display_name,
                resolve_display_card_budget,
            )

            plan = allocate_card_budget_per_book(budget_state, profile=profile)
            info = plan.get(self.book_name, {})
            if self.book_name == "Odds API":
                book_disabled = False
                pool = float(available_card_budget_usd(budget_state, profile=profile) or 0)
                if pool <= 0:
                    pool, _ = resolve_display_card_budget(budget_state, profile=profile)
                if pool > 0:
                    pool_line = f"Prop stake pool: ${pool:.2f} (Odds API / card budget)."
            else:
                book_disabled = not info.get("enabled", True)
                pool = float(info.get("allocation") or 0)
                if book_disabled:
                    pool_line = f"{book_display_name(self.book_name)} disabled in Advanced book toggles."
                elif pool > 0:
                    pool_line = f"Prop stake pool: ${pool:.2f} (auto card budget)."

        has_cards = bool(payload.cards) or not _df_is_empty(payload.combined)
        if not has_cards:
            self.summary.configure(text=f"{self.book_name} - {self.book_note}")
            self.backtest_box.configure(text="")
            self._pack_empty_state(
                "No card loaded",
                "Click Refresh Next Two to pull fights, model predictions, and prop markets.",
                color="#fbbf24",
            )
            return

        cap = self._paper_display_cap() if self._is_paper_profile() else config.PROP_MAX_RESULTS
        self.summary.configure(
            text=_ascii_ui(
                f"{self.book_name.replace('.eu', '')} props  |  "
                f"Up to {cap} lines  |  "
                f"Strict: >={config.PROP_MIN_MODEL_PROB:.0%} model or >={config.PROP_MIN_EDGE:.0%} live edge"
                + (f"  |  {pool_line}" if pool_line else "")
            )
        )
        self._render_backtest(payload)

        book = payload.books.get(self.book_name, {})
        props = book.get("props") or {}
        singles_all = props.get("singles") or []
        if not singles_all and props_on and not book_disabled:
            try:
                from src.dashboard_service import _build_props_payload

                book_preds = book.get("predictions", payload.combined)
                if isinstance(book_preds, pd.DataFrame) and not book_preds.empty:
                    props = _build_props_payload(
                        book_preds,
                        self.book_name,
                        force_refresh_odds=False,
                        budget_state=budget_state,
                    )
                    singles_all = props.get("singles") or []
                    payload.books.setdefault(self.book_name, {})["props"] = props
                    _debug_log(
                        f"Props rebuild for {self.book_name}: {len(singles_all)} ranked "
                        f"(live rows {props.get('prop_odds_rows', 0)})"
                    )
            except Exception as exc:
                _debug_log(f"Props rebuild failed for {self.book_name}: {exc}")
        singles = self._filter_singles(singles_all)
        singles = attach_prop_stakes(singles, budget_state, self.book_name, profile=profile)
        meta = props.get("singles_meta") or {}
        parlays = props.get("parlays") or [] if self.show_parlays else []
        book_warning = str(book.get("warning") or book.get("error") or "").strip()
        props_warn = str(props.get("warning") or "").strip()
        if props_warn and props_warn not in book_warning:
            book_warning = f"{book_warning} | {props_warn}".strip(" |") if book_warning else props_warn

        total_found = int(meta.get("total_found", len(singles_all)))
        strict_count = int(meta.get("strict_count", 0))
        shown = len(singles)
        self._update_show_all_controls(meta, shown)
        synth_n = int(meta.get("synthetic_count", 0))
        live_prop_n = int(meta.get("live_count", 0))
        if total_found:
            parts = [f"Showing {shown} of {total_found} ranked props"]
            if synth_n:
                parts.append(f"{synth_n} synthetic (slate, model %)")
            if live_prop_n:
                parts.append(f"{live_prop_n} live edge")
            self.filter_label.configure(text="  |  ".join(parts))
        else:
            self.filter_label.configure(text="")

        if book_warning:
            ctk.CTkLabel(
                self.scroll,
                text=f"Note: {book_warning}",
                anchor="w",
                text_color="#fbbf24",
                font=ctk.CTkFont(size=12),
                wraplength=1100,
                justify="left",
            ).pack(fill="x", padx=8, pady=(4, 2))

        live_lines = props.get("live_prop_lines") or {}
        live_n = int(live_lines.get("live", 0))
        prop_rows = int(props.get("prop_odds_rows", 0))
        if prop_rows:
            ctk.CTkLabel(
                self.scroll,
                text=f"Live prop lines fetched: {live_n} (of {prop_rows} market rows)",
                anchor="w",
                text_color="#9ca3af",
                font=ctk.CTkFont(size=12),
            ).pack(fill="x", padx=8)

        if not singles and not parlays:
            relaxed_hidden = max(0, total_found - strict_count)
            hint = self._empty_state_hint(
                total_found=total_found,
                strict_count=strict_count,
                book_warning=book_warning,
                book_disabled=book_disabled,
            )
            title = "No props to show"
            if book_disabled:
                title = "Book disabled in Advanced toggles"
            elif total_found and relaxed_hidden and not bool(self.show_all_var.get()):
                title = f"{relaxed_hidden} relaxed props hidden"
            self._pack_empty_state(title, hint.replace("**", ""))
            return

        if singles:
            self._render_props_table(singles)

        if parlays:
            ctk.CTkLabel(self.scroll, text="Prop / mixed parlays", anchor="w").pack(fill="x", padx=4, pady=(6, 0))
            for p in parlays[:5]:
                hdr = (
                    f"Parlay #{p.get('rank', 0)}  |  {p.get('n_legs', 0)}-Team  |  "
                    f"prob {p.get('combined_prob', 0):.0%}  |  "
                    f"odds {p.get('combined_odds', 0):.2f}  |  "
                    f"EV {p.get('expected_value', 0):+.0%}"
                )
                if p.get("correlation_adjusted"):
                    hdr += "  |  corr-adj"
                ctk.CTkLabel(self.scroll, text=hdr, anchor="w", text_color="#60a5fa").pack(fill="x", padx=8)
                for line in p.get("_leg_rows") or []:
                    ctk.CTkLabel(self.scroll, text=line, anchor="w").pack(fill="x", padx=16)
        elif self.show_parlays and not config.BOOK_PROP_RULES.get(self.book_name, {}).get(
            "allow_prop_parlays", False
        ):
            ctk.CTkLabel(
                self.scroll,
                text=f"{self.book_name}: prop singles only (parlays disabled by BOOK_PROP_RULES).",
                anchor="w",
                text_color="#94a3b8",
                font=ctk.CTkFont(size=12),
            ).pack(fill="x", padx=8, pady=(6, 0))


def _apply_risk_warning_label(
    label: ctk.CTkLabel,
    warnings: list[tuple[str, str]],
    *,
    grid: dict[str, Any] | None = None,
    pack: dict[str, Any] | None = None,
) -> None:
    """Show or hide a tab warning label from unified risk warnings."""
    from src.strategy import format_risk_warnings

    text, color = format_risk_warnings(warnings)
    if text:
        label.configure(text=text, text_color=color)
        if grid is not None:
            label.grid(**grid)
        else:
            label.pack(**(pack or {"fill": "x", "padx": 12, "pady": (8, 0)}))
    elif grid is not None:
        label.grid_remove()
    else:
        label.pack_forget()


def _budget_badge_style(pool_usd: float, *, books_enabled: bool) -> tuple[str, str]:
    """Badge colors for Available-this-card (uses strategy helper when present)."""
    try:
        from src.strategy import budget_availability_badge_style

        return budget_availability_badge_style(pool_usd, books_enabled=books_enabled)
    except ImportError:
        if not books_enabled:
            return "#451a1a", "#fca5a5"
        if pool_usd > 50:
            return "#14532d", "#86efac"
        if pool_usd >= 20:
            return "#713f12", "#fde047"
        return "#451a1a", "#fca5a5"


class BudgetManagerBar(_CTK_FRAME):
    """Slim bankroll control — no Budget Manager chrome.

    Card budget is internal (bankroll × profile risk %) unless Advanced override.
    """

    def __init__(
        self,
        master,
        *,
        on_save: Callable[[dict[str, Any]], None],
        on_change: Callable[[dict[str, Any]], None] | None = None,
        profile_getter: Callable[[], str],
        leading_frame: _CTK_FRAME | None = None,
        **kwargs,
    ) -> None:
        super().__init__(master, fg_color="transparent", **kwargs)
        self._on_save = on_save
        self._on_change = on_change
        self._profile_getter = profile_getter
        self._persisted_state = config.default_budget_state()
        self._refreshing = False
        self._refresh_after_id: str | None = None
        self._pending_notify_parent = True
        self._card_budget_overridden = False
        self._syncing_card_from_bankroll = False
        self._advanced_open = False
        self._persist_after_id: str | None = None

        self.total_bankroll_var = ctk.StringVar(value=f"{config.DEFAULT_TOTAL_BANKROLL:g}")
        self.card_budget_var = ctk.StringVar(value=f"{config.DEFAULT_CARD_BUDGET:g}")
        self.use_betnow_var = ctk.BooleanVar(value=True)
        self.use_dk_var = ctk.BooleanVar(value=True)
        self.use_myb_var = ctk.BooleanVar(value=True)

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=(2, 2))

        # leading_frame is unused (profile/event live in the parent control header)
        _ = leading_frame

        _BANKROLL_TIP = (
            "Total betting bankroll (persisted). "
            "Card budget auto = bankroll × profile card-risk % (Paper/Live). "
            "Conf/odds sizing allocates % of that auto card budget."
        )
        self.bankroll_label = ctk.CTkLabel(
            row,
            text="Bankroll $",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#f8fafc",
            anchor="w",
        )
        self.bankroll_label.pack(side="left", padx=(0, 4))
        self.bankroll_entry = ctk.CTkEntry(
            row,
            textvariable=self.total_bankroll_var,
            width=140,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.bankroll_entry.pack(side="left", padx=(0, 6))
        self.bankroll_entry.bind("<KeyRelease>", self._on_bankroll_edited)
        self.bankroll_entry.bind("<FocusOut>", lambda _e: self._persist_bankroll())
        _ToolTip(self.bankroll_label, _BANKROLL_TIP)
        _ToolTip(self.bankroll_entry, _BANKROLL_TIP)

        self.auto_card_hint = ctk.CTkLabel(
            row,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            anchor="w",
        )
        self.auto_card_hint.pack(side="left", padx=(4, 10))

        self.advanced_btn = ctk.CTkButton(
            row,
            text="Advanced",
            width=88,
            height=28,
            fg_color="#334155",
            hover_color="#475569",
            command=self._toggle_advanced,
        )
        self.advanced_btn.pack(side="left", padx=(0, 6))

        ctk.CTkButton(row, text="Save", width=56, height=28, command=self._save).pack(
            side="left", padx=(0, 4)
        )

        # Optional Advanced collapse (card override + book toggles)
        self.advanced_frame = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=8)
        adv = self.advanced_frame
        adv_row = ctk.CTkFrame(adv, fg_color="transparent")
        adv_row.pack(fill="x", padx=10, pady=8)

        _CARD_TIP = (
            "Optional override of auto card budget. "
            "Leave alone for bankroll × profile risk %. Clear override with Reset auto."
        )
        self.card_budget_label = ctk.CTkLabel(adv_row, text="Card $", anchor="w", width=52)
        self.card_budget_label.pack(side="left", padx=(0, 2))
        self.card_budget_entry = ctk.CTkEntry(adv_row, textvariable=self.card_budget_var, width=110)
        self.card_budget_entry.pack(side="left", padx=(0, 6))
        self.card_budget_entry.bind("<KeyRelease>", self._on_card_budget_edited)
        _ToolTip(self.card_budget_label, _CARD_TIP)
        _ToolTip(self.card_budget_entry, _CARD_TIP)

        self.card_cap_hint = ctk.CTkLabel(adv_row, text="", text_color="#9ca3af", anchor="w")
        self.card_cap_hint.pack(side="left", padx=(0, 8))
        self.live_cap_warning = ctk.CTkLabel(
            adv_row, text="", text_color="#f87171", font=ctk.CTkFont(size=11), anchor="w"
        )
        self.live_cap_warning.pack(side="left", padx=(0, 8))

        ctk.CTkCheckBox(
            adv_row, text="BetNow", variable=self.use_betnow_var, width=80, command=self._schedule_refresh
        ).pack(side="left", padx=(0, 2))
        ctk.CTkCheckBox(
            adv_row, text="DK", variable=self.use_dk_var, width=56, command=self._schedule_refresh
        ).pack(side="left", padx=(0, 2))
        ctk.CTkCheckBox(
            adv_row, text="MyBookie", variable=self.use_myb_var, width=90, command=self._schedule_refresh
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            adv_row,
            text="Reset auto",
            width=80,
            height=26,
            fg_color="#4b5563",
            hover_color="#6b7280",
            command=self._reset_card_auto,
        ).pack(side="right")

        # Optional attrs kept for older callers (never required for sizing)
        self.available_badge = None
        self.available_label = None
        self.warning_box = None

    def _toggle_advanced(self) -> None:
        self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self.advanced_frame.pack(fill="x", padx=4, pady=(0, 4))
            self.advanced_btn.configure(text="Advanced (on)")
            self._refresh_live_ui()
        else:
            self.advanced_frame.pack_forget()
            self.advanced_btn.configure(text="Advanced")

    def _parse_float(self, var: ctk.StringVar, default: float = 0.0) -> float:
        try:
            return max(float(str(var.get()).strip().replace("$", "").replace(",", "")), 0.0)
        except ValueError:
            return default

    def _profile_key(self) -> str:
        try:
            return config.normalize_profile(self._profile_getter())
        except Exception:
            return "paper"

    def _default_card_budget(self, bankroll: float | None = None) -> float:
        br = float(bankroll) if bankroll is not None else self._parse_float(
            self.total_bankroll_var, config.DEFAULT_TOTAL_BANKROLL
        )
        return float(config.default_card_budget_usd(br, profile=self._profile_key()))

    def _sync_card_budget_from_bankroll(self) -> None:
        if self._card_budget_overridden:
            return
        self._syncing_card_from_bankroll = True
        try:
            default = self._default_card_budget()
            self.card_budget_var.set(f"{default:.2f}".rstrip("0").rstrip("."))
        finally:
            self._syncing_card_from_bankroll = False

    def _on_bankroll_edited(self, _event=None) -> None:
        self._sync_card_budget_from_bankroll()
        self._schedule_refresh()
        # Debounced persist
        if self._persist_after_id is not None:
            try:
                self.after_cancel(self._persist_after_id)
            except Exception:
                pass
        self._persist_after_id = self.after(800, self._persist_bankroll)

    def _on_card_budget_edited(self, _event=None) -> None:
        if not self._syncing_card_from_bankroll:
            self._card_budget_overridden = True
        self._schedule_refresh()

    def _reset_card_auto(self) -> None:
        self._card_budget_overridden = False
        self._sync_card_budget_from_bankroll()
        self._schedule_refresh(notify_parent=True)

    def apply_profile_defaults(self, *, force_card: bool = False) -> None:
        if force_card:
            self._card_budget_overridden = False
        self._sync_card_budget_from_bankroll()
        self._schedule_refresh(notify_parent=True)

    def get_state(self) -> dict[str, Any]:
        br = self._parse_float(self.total_bankroll_var, config.DEFAULT_TOTAL_BANKROLL)
        n_enabled = max(
            sum(
                [
                    bool(self.use_betnow_var.get()),
                    bool(self.use_dk_var.get()),
                    bool(self.use_myb_var.get()),
                ]
            ),
            1,
        )
        per_book = br / n_enabled
        if self._card_budget_overridden:
            card = self._parse_float(self.card_budget_var, self._default_card_budget(br))
        else:
            card = self._default_card_budget(br)
            self._syncing_card_from_bankroll = True
            try:
                self.card_budget_var.set(f"{card:.2f}".rstrip("0").rstrip("."))
            finally:
                self._syncing_card_from_bankroll = False
        return {
            "total_bankroll": br,
            "card_budget": card,
            "card_budget_overridden": bool(self._card_budget_overridden),
            "betnow_balance": per_book,
            "draftkings_balance": per_book,
            "mybookie_balance": per_book,
            "use_betnow": bool(self.use_betnow_var.get()),
            "use_draftkings": bool(self.use_dk_var.get()),
            "use_mybookie": bool(self.use_myb_var.get()),
        }

    def load(self, state: dict[str, Any]) -> None:
        normalized = config.normalize_budget_state(state)
        self._persisted_state = normalized
        self.total_bankroll_var.set(f"{normalized['total_bankroll']:.2f}".rstrip("0").rstrip("."))
        br = float(normalized["total_bankroll"])
        default_card = self._default_card_budget(br)
        saved_card = float(normalized["card_budget"])
        if "card_budget_overridden" in state:
            self._card_budget_overridden = bool(state.get("card_budget_overridden"))
        else:
            # Migrate: prefer auto card budget (no separate editor in main UI)
            self._card_budget_overridden = False
        if self._card_budget_overridden:
            self.card_budget_var.set(f"{saved_card:.2f}".rstrip("0").rstrip("."))
        else:
            self.card_budget_var.set(f"{default_card:.2f}".rstrip("0").rstrip("."))
        self.use_betnow_var.set(bool(normalized["use_betnow"]))
        self.use_dk_var.set(bool(normalized["use_draftkings"]))
        self.use_myb_var.set(bool(normalized["use_mybookie"]))
        self._refresh_live(notify_parent=False)

    def refresh_warnings(self) -> None:
        self._refresh_live(notify_parent=False)

    def _schedule_refresh(self, *, notify_parent: bool = True) -> None:
        self._pending_notify_parent = notify_parent
        if self._refresh_after_id is not None:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
        self._refresh_after_id = self.after(10, self._run_scheduled_refresh)

    def _run_scheduled_refresh(self) -> None:
        self._refresh_after_id = None
        self._refresh_live(notify_parent=self._pending_notify_parent)

    def _refresh_live(self, *, notify_parent: bool = True) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self._refresh_live_ui()
            if notify_parent and self._on_change:
                state = config.normalize_budget_state(self.get_state())
                state["card_budget_overridden"] = bool(self._card_budget_overridden)
                self._on_change(state)
        finally:
            self._refreshing = False

    def _refresh_live_ui(self) -> None:
        """Refresh auto-card hint / advanced caps. Safe if optional widgets are missing."""
        try:
            profile = self._profile_getter()
            state = self.get_state()
            br = float(state["total_bankroll"])
            card = float(state["card_budget"])
            risk_frac = config.profile_card_risk_fraction(profile=profile)
            default_card = self._default_card_budget(br)
            mode = "override" if self._card_budget_overridden else "auto"
            hint = getattr(self, "auto_card_hint", None)
            if hint is not None:
                hint.configure(
                    text=f"Auto card ${default_card:g} (x{100 * risk_frac:.0f}%)"
                    + (f" | using ${card:g}" if self._card_budget_overridden else "")
                )

            if not getattr(self, "_advanced_open", False):
                return
            cap_hint = getattr(self, "card_cap_hint", None)
            live_warn = getattr(self, "live_cap_warning", None)
            is_live = config.normalize_profile(profile) == "live"
            if is_live:
                live_cap = config.live_card_budget_cap_usd(br)
                over = card > live_cap
                if cap_hint is not None:
                    cap_hint.configure(
                        text=f"Live cap ${live_cap:g} · {mode}",
                        text_color="#f87171" if over else "#9ca3af",
                    )
                if live_warn is not None:
                    live_warn.configure(text="Over Live cap" if over else "")
            else:
                safe = config.max_card_stake_cap(br)
                if cap_hint is not None:
                    cap_hint.configure(
                        text=f"Safe ~${safe:g} · {mode}",
                        text_color="#9ca3af",
                    )
                if live_warn is not None:
                    live_warn.configure(text="")
        except Exception as exc:
            _debug_log(f"budget bar _refresh_live_ui skipped: {exc}")

    def _persist_bankroll(self) -> None:
        """Quietly persist bankroll + auto card budget."""
        self._persist_after_id = None
        try:
            state = config.normalize_budget_state(self.get_state())
            state["card_budget_overridden"] = bool(self._card_budget_overridden)
            saved = config.save_budget(state)
            try:
                import json

                path = config.BUDGET_JSON_PATH
                raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                if isinstance(raw, dict):
                    raw["card_budget_overridden"] = bool(self._card_budget_overridden)
                    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            except Exception:
                pass
            self._persisted_state = saved
        except Exception as exc:
            _debug_log(f"bankroll persist skipped: {exc}")

    def _save(self) -> None:
        print("Button [Bankroll Save] clicked", flush=True)
        state = config.normalize_budget_state(self.get_state())
        state["card_budget_overridden"] = bool(self._card_budget_overridden)
        profile = self._profile_getter()
        if config.normalize_profile(profile) == "live":
            live_cap = config.live_card_budget_cap_usd(state["total_bankroll"])
            if state["card_budget"] > live_cap:
                state["card_budget"] = live_cap
                self.card_budget_var.set(f"{live_cap:g}")
        saved = config.save_budget(state)
        try:
            import json

            path = config.BUDGET_JSON_PATH
            raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if isinstance(raw, dict):
                raw["card_budget_overridden"] = bool(self._card_budget_overridden)
                path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        except Exception:
            pass
        self._persisted_state = saved
        self.load({**saved, "card_budget_overridden": self._card_budget_overridden})
        self._on_save(saved)

    def _reset_defaults(self) -> None:
        self._card_budget_overridden = False
        defaults = config.default_budget_state()
        br = float(defaults["total_bankroll"])
        defaults["card_budget"] = self._default_card_budget(br)
        self.load({**defaults, "card_budget_overridden": False})
        self._schedule_refresh(notify_parent=True)


class GaneFoulScenarioPanel(_CTK_FRAME):
    """Prominent speculative panel for Pereira vs Gane foul/eye-poke tail risk."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, fg_color="#2a1215", corner_radius=8, **kwargs)
        self._border = ctk.CTkFrame(self, fg_color="#7f1d1d", corner_radius=10)
        self._border.pack(fill="x", padx=10, pady=(10, 6))
        self._inner = ctk.CTkFrame(self._border, fg_color="#1f1012", corner_radius=8)
        self._inner.pack(fill="x", padx=2, pady=2)

        ctk.CTkLabel(
            self._inner,
            text="WARNING:  GANE EYE POKE / FOUL SCENARIO",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#fca5a5",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(
            self._inner,
            text="HIGH RISK / SPECULATIVE - tail-event hedge only, not a model core pick",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#fbbf24",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 6))

        self.fight_label = ctk.CTkLabel(
            self._inner,
            text="Alex Pereira vs Ciryl Gane",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#e5e7eb",
            anchor="w",
        )
        self.fight_label.pack(fill="x", padx=12, pady=(0, 4))

        self.best_bet_label = ctk.CTkLabel(
            self._inner, text="", anchor="w", justify="left", wraplength=1050, text_color="#93c5fd"
        )
        self.best_bet_label.pack(fill="x", padx=12, pady=2)

        self.odds_label = ctk.CTkLabel(
            self._inner, text="", anchor="w", justify="left", wraplength=1050, text_color="#d1d5db"
        )
        self.odds_label.pack(fill="x", padx=12, pady=2)

        self.model_label = ctk.CTkLabel(
            self._inner, text="", anchor="w", justify="left", wraplength=1050, text_color="#3dd68c"
        )
        self.model_label.pack(fill="x", padx=12, pady=2)

        self.stake_label = ctk.CTkLabel(
            self._inner, text="", anchor="w", justify="left", text_color="#fde68a"
        )
        self.stake_label.pack(fill="x", padx=12, pady=2)

        self.explain_label = ctk.CTkLabel(
            self._inner,
            text="",
            anchor="w",
            justify="left",
            wraplength=1050,
            font=ctk.CTkFont(size=11),
            text_color="#9ca3af",
        )
        self.explain_label.pack(fill="x", padx=12, pady=(4, 10))

        self.empty_label = ctk.CTkLabel(
            self._inner,
            text="Load card data (Refresh) to evaluate the Gane foul scenario.",
            anchor="w",
            text_color="#6b7280",
        )
        self.empty_label.pack(fill="x", padx=12, pady=(0, 10))

    def render(self, scenario: dict[str, Any] | None) -> None:
        scenario = scenario or {}
        if not scenario.get("found"):
            self.fight_label.configure(text=scenario.get("fight_label", "Alex Pereira vs Ciryl Gane"))
            self.best_bet_label.configure(text="")
            self.odds_label.configure(text="")
            self.model_label.configure(text="")
            self.stake_label.configure(text="")
            self.explain_label.configure(text="")
            self.empty_label.configure(
                text=scenario.get("message", "Load card data (Refresh) to evaluate the Gane foul scenario.")
            )
            self.empty_label.pack(fill="x", padx=12, pady=(0, 10))
            return

        self.empty_label.pack_forget()
        best = scenario.get("best_bet") or {}
        book_quotes = scenario.get("book_quotes") or {}
        ml_prob = float(scenario.get("gane_ml_prob") or 0)
        ko_prob = float(scenario.get("gane_ko_prob") or 0)
        prop_key = str(best.get("prop_key", ""))
        model_p = float(best.get("model_prob") or (ml_prob if prop_key == "moneyline" else ko_prob))
        edge = best.get("edge")
        edge_txt = f"{float(edge):+.1%}" if edge is not None else "-"

        ev = str(scenario.get("event_name", "")).strip()
        fight_txt = scenario.get("fight_label", "Alex Pereira vs Ciryl Gane")
        if ev:
            fight_txt = f"{fight_txt}  ({ev})"
        self.fight_label.configure(text=fight_txt)

        self.best_bet_label.configure(
            text=(
                f"Best available proxy: {best.get('prop_label', '-')} @ {best.get('book', '-')}  "
                f"({best.get('american', '-')}, {float(best.get('decimal_odds', 0)):.2f} decimal)"
            )
        )

        lines = []
        for book in ("BetNow.eu", "DraftKings", "MyBookie"):
            q = book_quotes.get(book, {})
            short = book.replace(".eu", "")
            ml = q.get("moneyline_american", "-")
            ml_dec = q.get("moneyline_decimal")
            ml_part = f"ML {ml}" + (f" ({float(ml_dec):.2f})" if ml_dec else "")
            if q.get("method_available"):
                meth = q.get("method_american", "-")
                meth_dec = q.get("method_decimal")
                meth_part = f"KO/TKO {meth}" + (f" ({float(meth_dec):.2f})" if meth_dec else "")
                lines.append(f"{short}: {ml_part}  |  {meth_part}")
            else:
                lines.append(f"{short}: {ml_part}  |  KO/TKO -")
        self.odds_label.configure(text="Odds by book:  " + "     ".join(lines))

        self.model_label.configure(
            text=(
                f"Model: Gane ML {ml_prob:.1%}  |  Gane by KO/TKO (method proxy) {ko_prob:.1%}  |  "
                f"Edge on best line ({best.get('prop_label', '')}): {edge_txt}"
            )
        )

        stake = float(scenario.get("suggested_stake_usd") or 2.5)
        self.stake_label.configure(
            text=(
                f"Suggested speculative stake: ${stake:.2f} "
                f"(target range {scenario.get('stake_range', '$2-$3')} - lottery hedge only)"
            )
        )

        self.explain_label.configure(text=str(scenario.get("explanation", "")))


# --- Main application ---------------------------------------------------------


class UFCDashboardApp(_CTK_BASE):
    def __init__(self) -> None:
        if ctk is None or _CTK_BASE is object:
            raise RuntimeError("CustomTkinter failed to initialize")
        try:
            super().__init__()
        except Exception as exc:
            raise RuntimeError(f"Failed to create main window: {exc}") from exc

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("UFC Predictor Dashboard")
        self.resizable(True, True)
        self._fullscreen = False

        self._payload: DashboardPayload | None = None
        self._busy = False
        self._auto_watch = False
        self._last_full_refresh_ts: float | None = None
        self._last_odds_ts: float | None = None
        self._next_odds_ts: float | None = None
        self._next_card_ts: float | None = None
        self._model_ready = self._check_model_ready()
        self._render_token = 0
        self._tab_render_seq = 0
        self._pending_tab_after_busy: str | None = None
        self._busy_watchdog_id: str | None = None
        self._rendered_sections: set[str] = set()
        self._grok_result: dict[str, Any] | None = None
        self._grok_busy = False
        self._updating_budget = False
        self._budget_after_id: str | None = None

        config.UFC_PROFILE = "paper"
        config.apply_profile_overrides()
        self._budget_state = config.apply_budget_state()

        self.show_all_props_var = ctk.BooleanVar(value=config.PAPER_PROPS_SHOW_ALL_DEFAULT)
        self.profile_var = ctk.StringVar(value="Paper")
        self.event_var = ctk.StringVar(value="Next Two Cards")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_mode_banner()
        self.mode_banner.grid(row=0, column=0, sticky="ew")

        self.control_header = ctk.CTkFrame(self, fg_color="transparent")
        self._build_control_header()
        self.control_header.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 2))

        self._build_tabs()
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)

        self._status_frame = self._build_status_area()
        self._status_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))

        self._fit_initial_geometry()
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)
        self._schedule_status_tick()
        self.after(100, self._ensure_controls_enabled)
        self.after(200, self._load_background_cache_on_startup)

    def _wrap_button_click(self, name: str, handler: Callable[..., None]) -> Callable[..., None]:
        """Bind a button command with debug logging and safe error handling."""

        def wrapped(*args, **kwargs) -> None:
            print(f"Button [{name}] clicked", flush=True)
            _button_debug(f"{name} clicked")
            try:
                handler(*args, **kwargs)
            except Exception as exc:
                tb = traceback.format_exc()
                _debug_log(f"Button {name} failed: {tb}")
                self._show_error(f"{name}: {exc}")
                self._finish_busy()

        return wrapped

    def _ensure_controls_enabled(self) -> None:
        """Guarantee toolbar controls are interactive after startup/layout."""
        if self._busy:
            _debug_log("Controls check: busy - refresh buttons stay disabled until work finishes")
            return
        self._set_busy(False)
        _debug_log("Controls enabled (state=normal)")
        _dashboard_heartbeat("dashboard controls enabled (mainloop alive)")

    def _load_background_cache_on_startup(self) -> None:
        """Load background snapshot on startup only if it matches live upcoming events."""
        if self._busy or self._payload is not None:
            return

        def worker() -> None:
            try:
                from src.background_runner import load_background_snapshot
                from src.dashboard_service import _snapshot_matches_live_events

                data = load_background_snapshot(max_age_hours=24)
                if data is None:
                    data = load_background_snapshot(max_age_hours=72)
                    if data:
                        _debug_log("Startup: considering stale background cache (24-72h)")
                if data is not None and not _snapshot_matches_live_events(data):
                    _debug_log(
                        "Startup: skipping background cache - events do not match live slate "
                        f"({data.get('event_label')!r})"
                    )
                    data = None
                if data is None:
                    self.after(
                        0,
                        lambda: self._set_status(
                            'Ready - click "Refresh Next Two" (no background cache).'
                        ),
                    )
                    self.after(800, self._auto_refresh_if_empty)
                    return

                payload = _result_to_payload(data)
                manifest = data.get("_manifest") or {}
                full_ts = self._iso_to_epoch(manifest.get("full_run_at") or manifest.get("saved_at"))
                odds_ts = self._iso_to_epoch(manifest.get("odds_updated_at"))

                def apply() -> None:
                    if self._payload is not None or self._busy:
                        return
                    self.event_var.set("Next Two Cards")
                    if full_ts is not None:
                        self._last_full_refresh_ts = full_ts
                    if odds_ts is not None:
                        self._last_odds_ts = odds_ts
                    trigger = manifest.get("trigger", "background")
                    run_type = manifest.get("run_type", "full")
                    self._sync_profile_menu(payload.profile)
                    self._log_loaded_fights("background cache", payload)
                    self._apply_payload(payload, full_refresh=bool(full_ts), odds_refresh=bool(odds_ts))
                    self._set_status(
                        f"Loaded background cache ({run_type}/{trigger}) - {payload.event_label}"
                    )

                self.after(0, apply)
            except Exception as exc:
                _debug_log(f"Background cache load failed: {exc}")
                self.after(
                    0,
                    lambda: self._set_status(
                        f"Ready - click Refresh Next Two (cache load failed: {exc})."
                    ),
                )
                self.after(800, self._auto_refresh_if_empty)

        threading.Thread(target=worker, daemon=True).start()

    def _auto_refresh_if_empty(self) -> None:
        """First launch with no cache: run Next Two Cards refresh automatically."""
        if self._payload is not None or self._busy:
            return
        _debug_log("No payload on startup - auto-running Refresh Next Two")
        self.event_var.set("Next Two Cards")
        self._on_refresh()

    @staticmethod
    def _iso_to_epoch(ts: str | None) -> float | None:
        if not ts:
            return None
        try:
            from src.background_runner import _parse_iso

            dt = _parse_iso(ts)
            return dt.timestamp() if dt else None
        except (ValueError, TypeError):
            return None

    def _check_model_ready(self) -> bool:
        try:
            from main import _model_exists

            return _model_exists()
        except Exception as exc:
            _debug_log(f"Model check failed: {exc}")
            return False

    def _fit_initial_geometry(self) -> None:
        """Size the window to the current display (taller default, fully resizable)."""
        self.update_idletasks()
        sw = int(self.winfo_screenwidth() or 1366)
        sh = int(self.winfo_screenheight() or 768)
        # Prefer ~88% of screen height so fight lists + top bets fit without clipping.
        width = max(1024, min(1440, sw - 48))
        height = max(700, min(int(sh * 0.88), sh - 56))
        min_w = min(900, max(760, sw - 80))
        min_h = min(560, max(480, sh - 120))
        self.minsize(min_w, min_h)
        self.maxsize(sw, sh)
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        _debug_log(f"Window geometry {width}x{height} (screen {sw}x{sh}, min {min_w}x{min_h})")

    def _toggle_fullscreen(self, _event: Any = None) -> str:
        """F11 / button: toggle OS fullscreen."""
        self._fullscreen = not getattr(self, "_fullscreen", False)
        try:
            self.attributes("-fullscreen", self._fullscreen)
        except Exception as exc:
            _debug_log(f"Fullscreen toggle failed: {exc}")
            self._fullscreen = False
        if hasattr(self, "fullscreen_btn"):
            self.fullscreen_btn.configure(
                text="Exit Fullscreen" if self._fullscreen else "Fullscreen"
            )
        _debug_log(f"Fullscreen={'on' if self._fullscreen else 'off'}")
        return "break"

    def _exit_fullscreen(self, _event: Any = None) -> str | None:
        """Escape exits fullscreen only (does not close the app)."""
        if not getattr(self, "_fullscreen", False):
            return None
        self._fullscreen = False
        try:
            self.attributes("-fullscreen", False)
        except Exception:
            pass
        if hasattr(self, "fullscreen_btn"):
            self.fullscreen_btn.configure(text="Fullscreen")
        return "break"

    def _build_status_area(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self, fg_color="transparent")

        self.status = ctk.CTkLabel(frame, text="Ready - click Refresh to analyze.", anchor="w")
        self.status.pack(fill="x")

        self.status_bar = ctk.CTkLabel(
            frame,
            text=self._format_status_bar(),
            anchor="w",
            text_color="#9ca3af",
            font=ctk.CTkFont(size=12),
        )
        self.status_bar.pack(fill="x", pady=(4, 0))

        self.progress = ctk.CTkProgressBar(frame, height=10)
        self.progress.set(0)
        self._progress_visible = False
        return frame

    def _format_status_bar(self) -> str:
        model_txt = "Model ready" if self._model_ready else "Model missing"
        if self._last_full_refresh_ts is None:
            full_txt = "Last full refresh: -"
        else:
            mins = max(0, int((time.time() - self._last_full_refresh_ts) / 60))
            full_txt = f"Last full refresh: {mins} min ago" if mins else "Last full refresh: just now"
        if self._last_odds_ts is None:
            odds_txt = "Odds updated: -"
        else:
            mins = max(0, int((time.time() - self._last_odds_ts) / 60))
            odds_txt = f"Odds updated: {mins} min ago" if mins else "Odds updated: just now"
        return f"{model_txt}  |  {full_txt}  |  {odds_txt}"

    def _schedule_status_tick(self) -> None:
        self._update_status_bar()
        self.after(30_000, self._schedule_status_tick)

    def _set_status(self, text: str) -> None:
        """Status line with ASCII-safe text for Windows."""
        if hasattr(self, "status"):
            self.status.configure(text=_ascii_ui(text))

    def _update_status_bar(self) -> None:
        if hasattr(self, "status_bar"):
            self.status_bar.configure(text=_ascii_ui(self._format_status_bar()))

    def _show_progress(self, pct: float | None) -> None:
        if pct is None:
            if self._progress_visible:
                self.progress.pack_forget()
                self._progress_visible = False
            return
        if not self._progress_visible:
            self.progress.pack(fill="x", pady=(6, 0))
            self._progress_visible = True
        self.progress.set(max(0.0, min(1.0, pct)))

    def _build_mode_banner(self) -> None:
        self.mode_banner = ctk.CTkFrame(self, height=36, corner_radius=0)
        self.mode_banner.pack_propagate(False)
        self.mode_banner_label = ctk.CTkLabel(
            self.mode_banner,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="center",
            justify="center",
        )
        self.mode_banner_label.pack(fill="both", expand=True, padx=12, pady=2)
        self._update_mode_banner()

    def _update_mode_banner(self) -> None:
        """Status banner: Bankroll · Auto card · Profile (safe if budget bar missing)."""
        try:
            br = float(
                (self._budget_state or {}).get("total_bankroll")
                or getattr(config, "INITIAL_BANKROLL", 100)
                or 100
            )
        except Exception:
            br = 100.0
        try:
            profile_key = (
                self._profile_from_menu(self.profile_var.get())
                if hasattr(self, "profile_var")
                else ("live" if config.is_live_profile() else "paper")
            )
        except Exception:
            profile_key = "paper"
        profile = "Live" if profile_key == "live" else "Paper"
        try:
            auto_card = float(config.default_card_budget_usd(br, profile=profile_key))
        except Exception:
            auto_card = round(br * (0.18 if profile_key == "live" else 0.55), 2)

        card = auto_card
        overridden = False
        if hasattr(self, "budget_bar") and self.budget_bar is not None:
            try:
                st = self.budget_bar.get_state()
                br = float(st.get("total_bankroll") or br)
                auto_card = float(config.default_card_budget_usd(br, profile=profile_key))
                overridden = bool(st.get("card_budget_overridden"))
                card = float(st.get("card_budget") or auto_card)
            except Exception as exc:
                _debug_log(f"mode banner budget_bar skipped: {exc}")

        card_txt = (
            f"Auto card ${auto_card:,.0f}"
            if not overridden
            else f"Card ${card:,.0f} (override; auto ${auto_card:,.0f})"
        )
        line = f"Bankroll ${br:,.0f}  |  {card_txt}  |  {profile}"

        ha_line = ""
        try:
            from src.high_accuracy_strategy import format_strategy_rules_line

            ha_line = format_strategy_rules_line()
        except Exception:
            ha_line = ""

        if not hasattr(self, "mode_banner") or not hasattr(self, "mode_banner_label"):
            return
        try:
            if config.is_live_profile():
                self.mode_banner.configure(fg_color="#7f1d1d")
                self.mode_banner_label.configure(
                    text_color="#fecaca",
                    text=_ascii_ui(f"LIVE  |  {line}\n{ha_line}" if ha_line else f"LIVE  |  {line}"),
                )
            else:
                self.mode_banner.configure(fg_color="#1e3a5f")
                self.mode_banner_label.configure(
                    text_color="#bfdbfe",
                    text=_ascii_ui(
                        f"PAPER  |  {line}\n{ha_line}" if ha_line else f"PAPER  |  {line}"
                    ),
                )
        except Exception as exc:
            _debug_log(f"mode banner update failed: {exc}")

    def _current_budget_state(self) -> dict[str, Any]:
        if hasattr(self, "budget_bar"):
            return config.normalize_budget_state(self.budget_bar.get_state())
        return self._budget_state

    def _allowed_fight_keys(self, combined: pd.DataFrame) -> set[str]:
        """Fight ids + 'F1 vs F2' labels for filtering overview picks to loaded cards."""
        keys: set[str] = set()
        if _df_is_empty(combined):
            return keys
        key = getattr(config, "FIGHT_ID_COLUMN", "fight_id")
        if key in combined.columns:
            keys.update(combined[key].astype(str))
        f1c = "fighter_1" if "fighter_1" in combined.columns else "fighter1"
        f2c = "fighter_2" if "fighter_2" in combined.columns else "fighter2"
        if f1c in combined.columns and f2c in combined.columns:
            for _, row in combined[[f1c, f2c]].iterrows():
                keys.add(f"{row[f1c]} vs {row[f2c]}")
        return keys

    def _overview_recommendations(self) -> list[dict[str, Any]]:
        """Top picks: BLUE HA tickets first, then GREEN/YELLOW fun bets (never loosens gates)."""
        if self._payload is None:
            return []
        profile = self._profile_from_menu(self.profile_var.get())
        items: list[dict[str, Any]] = []
        try:
            from src.strategy import (
                aggregate_overview_recommendations,
                apply_narrative_tilt_after_model_sizing,
            )

            recs = aggregate_overview_recommendations(
                self._payload.books,
                self._current_budget_state(),
                limit=5,
                profile=profile,
                allowed_fights=self._allowed_fight_keys(self._payload.combined),
            )
            tilted = apply_narrative_tilt_after_model_sizing(recs, self._grok_result)
            if isinstance(tilted, dict):
                items = list(tilted.get("items") or [])
                if not items:
                    items = (
                        list(tilted.get("singles") or [])
                        + list(tilted.get("prop_singles") or [])
                        + list(tilted.get("parlays") or [])
                    )
                pool = float(
                    tilted.get("card_pool_usd")
                    or (items[0].get("card_pool_usd") if items else 0)
                    or 0
                )
            else:
                items = list(tilted or [])
                pool = float(items[0].get("card_pool_usd") or 0) if items else 0.0
            if items and pool > 0:
                try:
                    from src.strategy import allocate_card_budget_pct

                    items = allocate_card_budget_pct(items, pool, profile=profile, inplace=True)
                    for i, t in enumerate(items, start=1):
                        t["rank"] = i
                except Exception as exc:
                    _debug_log(f"Overview stake re-alloc skipped: {exc}")
        except Exception as exc:
            _debug_log(f"Overview recommendations fallback: {exc}")
            overview = self._payload.books.get("Overview", {}).get("alerts") or {}
            singles = overview.get("singles") or []
            items = []
            for i, s in enumerate(singles[:5], start=1):
                fight = str(s.get("fight") or "")
                pick = str(s.get("pick") or "")
                items.append(
                    {
                        "rank": i,
                        "pick_line": f"{pick} over {fight.split(' vs ')[-1] if ' vs ' in fight else fight}",
                        "display_label": f"{pick} ML",
                        "edge_pct": float(s.get("edge_pct") or float(s.get("edge") or 0) * 100),
                        "book": "Overview",
                        "american_odds": "-",
                        "odds_display": "-",
                        "suggested_stake": float(s.get("suggested_stake") or 0),
                        "card_pool_usd": float(
                            config.default_card_budget_usd(
                                float(self._budget_state.get("total_bankroll") or 100),
                                profile=profile,
                            )
                        ),
                    }
                )

        for t in items:
            from src.bet_tiers import TIER_BLUE, TIER_SKY_BLUE, is_sky_blue_ticket

            def _sf(v: Any) -> float | None:
                try:
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return float(v)
                except (TypeError, ValueError):
                    return None

            if is_sky_blue_ticket(
                stake_pct=_sf(t.get("stake_pct")),
                stake_usd=_sf(t.get("suggested_stake")) or _sf(t.get("stake_usd")),
                uncertainty_reason=str(t.get("uncertainty_reason") or ""),
            ):
                t["bet_tier"] = TIER_SKY_BLUE
                t["tier"] = TIER_SKY_BLUE
            else:
                t.setdefault("bet_tier", TIER_BLUE)
                t.setdefault("tier", t.get("bet_tier") or TIER_BLUE)
            t["fun_bet"] = False

        # Fill remaining slots with GREEN/YELLOW fun rankings (HA gates unchanged).
        try:
            from src.bet_tiers import (
                TIER_BLUE,
                TIER_GREEN,
                TIER_SKY_BLUE,
                TIER_YELLOW,
                collect_props_from_books,
                merge_bet_tier_dicts,
                rank_card_bet_tiers,
                rank_prop_bet_tiers,
            )

            overview_alerts = (self._payload.books.get("Overview") or {}).get("alerts") or {}
            cleared = list(overview_alerts.get("singles") or [])
            for book_data in (self._payload.books or {}).values():
                if isinstance(book_data, dict):
                    cleared.extend((book_data.get("alerts") or {}).get("singles") or [])
            preds = self._payload.combined
            if _df_is_empty(preds):
                preds = (self._payload.books.get("Overview") or {}).get("predictions")
            tiers = rank_card_bet_tiers(
                preds if isinstance(preds, pd.DataFrame) else None,
                cleared_singles=cleared,
                limit_per_tier=5,
            )
            try:
                prop_rows = collect_props_from_books(self._payload.books or {}, limit=6)
                prop_tiers = rank_prop_bet_tiers(prop_rows, limit_per_tier=5)
                tiers = merge_bet_tier_dicts(tiers, prop_tiers, limit_per_tier=8)
            except Exception:
                pass
            for fun in list(tiers.get(TIER_GREEN) or []) + list(tiers.get(TIER_YELLOW) or []):
                items.append(fun)
            if not items:
                items = list(tiers.get("best_fun") or [])
            for t in items:
                if t.get("bet_tier") in {TIER_GREEN, TIER_YELLOW}:
                    t["suggested_stake"] = 0.0
                    t["stake_usd"] = 0.0
                    t["stake_pct"] = 0.0
                    t["fun_bet"] = True
                elif not t.get("bet_tier"):
                    t["bet_tier"] = TIER_BLUE
                elif t.get("bet_tier") == TIER_SKY_BLUE:
                    t["fun_bet"] = False
                    t["advisory"] = False
        except Exception as exc:
            _debug_log(f"Fun tier fill skipped: {exc}")

        # Single merge: dedupe + rank + true Top 5 (never concat HA + fun past N)
        try:
            from src.bet_slip import dedupe_rank_top_tickets

            items = dedupe_rank_top_tickets(
                items,
                limit=5,
                event=str(getattr(self._payload, "event_label", "") or ""),
            )
            _debug_log(
                f"Overview recommendations shown={len(items)} "
                f"(blue={sum(1 for t in items if t.get('bet_tier')=='blue')}, "
                f"sky={sum(1 for t in items if t.get('bet_tier')=='sky_blue')}, "
                f"fun={sum(1 for t in items if t.get('fun_bet'))})"
            )
        except Exception as exc:
            _debug_log(f"Overview dedupe/rank skipped: {exc}")
            items = items[:5]
            for i, t in enumerate(items, start=1):
                t["rank"] = i
        return items

    def _payload_has_usable_odds(self) -> bool:
        if self._payload is None:
            return False
        try:
            from src.grok_analysis import books_have_usable_odds

            return books_have_usable_odds(self._payload.books)
        except Exception:
            return False

    def _card_allocation_status_line(self, bets: list[dict[str, Any]]) -> str:
        """SSO T status: Auto card $X · Allocated $Y (Z%) · Tickets N."""
        from src.strategy import format_card_allocation_status, resolve_display_card_budget

        profile = self._profile_from_menu(self.profile_var.get())
        budget = self._current_budget_state()
        # Preserve override flag from budget bar when present
        if hasattr(self, "budget_bar") and self.budget_bar is not None:
            try:
                st = self.budget_bar.get_state()
                budget = {**budget, **st}
                budget["card_budget_overridden"] = bool(
                    getattr(self.budget_bar, "_card_budget_overridden", False)
                )
            except Exception:
                pass
        card, overridden = resolve_display_card_budget(budget, profile=profile)
        auto = float(
            config.default_card_budget_usd(
                float(budget.get("total_bankroll") or card), profile=profile
            )
        )
        allocated = sum(float(b.get("suggested_stake") or 0) for b in bets)
        return format_card_allocation_status(
            auto_card_usd=auto,
            allocated_usd=allocated,
            n_tickets=len(bets),
            overridden=overridden,
            card_budget_usd=card if overridden else None,
        )

    def _refresh_top_recommendations(self) -> None:
        if hasattr(self, "top_bets_panel"):
            items = self._overview_recommendations()
            status = self._card_allocation_status_line(
                [b for b in items if not b.get("fun_bet")]
            )
            empty_reason = None
            if not items and self._payload is not None and not self._payload_has_usable_odds():
                empty_reason = "NO BET — no usable odds (fail-closed)"
            elif not items and self._payload is not None:
                empty_reason = "No BLUE/GREEN edges on this card."
            elif items and all(b.get("fun_bet") for b in items):
                status = (
                    (status + " · " if status else "")
                    + "No BLUE HA tickets — showing GREEN/YELLOW fun picks ($0)"
                )
            self.top_bets_panel.render(
                items, pool_status=status, empty_reason=empty_reason
            )

    def _build_control_header(self) -> None:
        """Slim top bar: Profile · Event · actions · Bankroll."""
        top = ctk.CTkFrame(self.control_header, fg_color="transparent")
        top.pack(fill="x", padx=4, pady=(4, 2))

        ctk.CTkLabel(top, text="Profile", font=ctk.CTkFont(weight="bold")).pack(
            side="left", padx=(4, 4)
        )
        self.profile_menu = ctk.CTkOptionMenu(
            top,
            variable=self.profile_var,
            values=["Paper", "Live"],
            width=90,
            command=self._wrap_button_click("Profile", self._on_profile_change),
        )
        self.profile_menu.pack(side="left", padx=(0, 8))

        ctk.CTkLabel(top, text="Event", font=ctk.CTkFont(weight="bold")).pack(
            side="left", padx=(4, 4)
        )
        self.event_menu = ctk.CTkOptionMenu(
            top,
            variable=self.event_var,
            values=["Next Two Cards", "Next Card", "UFC 329"],
            width=140,
        )
        self.event_menu.pack(side="left", padx=(0, 10))

        # Actions stay on the first row so they never get pushed below the fold.
        self.refresh_btn = ctk.CTkButton(
            top,
            text="Refresh",
            width=90,
            height=28,
            state="normal",
            fg_color="#2563eb",
            hover_color="#3b82f6",
            command=self._wrap_button_click("Refresh Next Two", self._on_refresh),
        )
        self.refresh_btn.pack(side="left", padx=(0, 4))
        _ToolTip(
            self.refresh_btn,
            "Refresh Next Two — load UFC.com cards (predictions). Reuses odds cache when ODDS_FETCH_ONCE.",
        )

        self.soft_update_btn = ctk.CTkButton(
            top,
            text="Soft Update",
            width=100,
            height=28,
            state="normal",
            fg_color="#1d4ed8",
            hover_color="#2563eb",
            command=self._wrap_button_click("Soft Update", self._on_soft_update),
        )
        self.soft_update_btn.pack(side="left", padx=4)
        _ToolTip(
            self.soft_update_btn,
            "Reload .env and attach Odds API lines/props (reuses first download when ODDS_FETCH_ONCE).",
        )

        self.restart_app_btn = ctk.CTkButton(
            top,
            text="Restart",
            width=80,
            height=28,
            state="normal",
            fg_color="#b91c1c",
            hover_color="#dc2626",
            command=self._wrap_button_click("Restart App", self._on_restart_app),
        )
        self.restart_app_btn.pack(side="left", padx=4)
        _ToolTip(
            self.restart_app_btn,
            "Quit and relaunch so .env / code changes load cleanly.",
        )

        self.fullscreen_btn = ctk.CTkButton(
            top,
            text="Full",
            width=56,
            height=28,
            fg_color="#334155",
            hover_color="#475569",
            command=self._wrap_button_click("Fullscreen", self._toggle_fullscreen),
        )
        self.fullscreen_btn.pack(side="left", padx=(4, 10))

        self.meta_label = ctk.CTkLabel(top, text="", text_color="#9ca3af", width=1)
        self.meta_label.pack(side="right", padx=(4, 4))

        self.budget_bar = BudgetManagerBar(
            top,
            on_save=self._on_budget_saved,
            on_change=self._on_budget_live_change,
            profile_getter=lambda: self._profile_from_menu(self.profile_var.get()),
        )
        self.budget_bar.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.budget_bar.load(self._budget_state)

    def _build_budget_bar(self, master=None) -> None:
        """Legacy hook - budget bar is built via _build_control_header."""
        return

    def _profile_from_menu(self, display: str) -> str:
        return config.normalize_profile(display)

    def _sync_profile_menu(self, profile: str) -> None:
        p = config.normalize_profile(profile)
        self.profile_var.set("Live" if p == "live" else "Paper")

    def _build_action_bar(self, master=None) -> None:
        """Legacy no-op — actions live on the control header row."""
        return
    def _build_tabs(self) -> None:
        self.tabs = ctk.CTkTabview(self)

        self.tab_overview = self.tabs.add("Overview")
        self.tab_odds_api = self.tabs.add("Odds API")
        # Optional scraper books — only when enabled in .env
        self.tab_betnow = None
        self.tab_dk = None
        self.tab_mybookie = None
        if getattr(config, "BETNOW_ENABLED", False):
            self.tab_betnow = self.tabs.add("BetNow.eu")
        if getattr(config, "DRAFTKINGS_ENABLED", False):
            self.tab_dk = self.tabs.add("DraftKings")
        if config.MYBOOKIE_ENABLED:
            self.tab_mybookie = self.tabs.add("MyBookie")
        self.tab_arb = self.tabs.add("Arb Scanner")
        self.tab_next_two = self.tabs.add("Next Two Cards")
        self.tab_props_odds_api = self.tabs.add("Odds API Props")
        self.tab_props_betnow = None
        self.tab_props_dk = None
        self.tab_props_mybookie = None
        if getattr(config, "BETNOW_ENABLED", False):
            self.tab_props_betnow = self.tabs.add("Props - BetNow")
        if getattr(config, "DRAFTKINGS_ENABLED", False):
            self.tab_props_dk = self.tabs.add("Props - DraftKings")
        if config.MYBOOKIE_ENABLED:
            self.tab_props_mybookie = self.tabs.add("Props - MyBookie")
        self.tab_risk = self.tabs.add("Risk Analysis")
        self.tab_grok = self.tabs.add("Ollama Analysis")

        # Overview: one full-page scroll hosts fights + top bets (no clipped bottom panel).
        self.tab_overview.grid_columnconfigure(0, weight=1)
        self.tab_overview.grid_rowconfigure(0, weight=1)
        self.overview_page = ctk.CTkScrollableFrame(
            self.tab_overview,
            fg_color="transparent",
            label_text="Overview",
        )
        self.overview_page.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.overview_summary = ctk.CTkLabel(
            self.overview_page,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
        )
        self.overview_summary.pack(fill="x", padx=12, pady=(6, 2))
        self.overview_risk_box = ctk.CTkLabel(
            self.overview_page,
            text="",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#f87171",
            wraplength=1100,
        )
        self.overview_risk_box.pack_forget()
        ctk.CTkLabel(
            self.overview_page,
            text="All Fights - scroll this page (header stays put)",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
            text_color="#cbd5e1",
        ).pack(fill="x", padx=12, pady=(2, 2))
        # Single page scroll — pack card sections directly (no nested scroll frame).
        self.overview_fights_scroll = ctk.CTkFrame(
            self.overview_page,
            fg_color="#0f172a",
        )
        self.overview_fights_scroll.pack(fill="x", padx=10, pady=(0, 6))
        self.top_bets_panel = TopRecommendedBetsPanel(self.overview_page)
        self.top_bets_panel.pack(fill="x", padx=8, pady=(2, 6))
        self.gane_foul_panel = GaneFoulScenarioPanel(self.overview_page)
        self.gane_foul_panel.pack_forget()

        self.grok_panel = GrokAnalysisPanel(
            self.tab_grok,
            on_run=self._wrap_button_click("Run Ollama Analysis Tab", self._on_grok_analysis),
            on_chat=self._wrap_button_click("Ollama Chat Ask", self._on_ollama_chat),
        )
        _configure_expandable_page(self.tab_grok, self.grok_panel)
        self._grok_result: dict[str, Any] | None = None

        self.odds_api_tab = BookTab(self.tab_odds_api, "Odds API")
        _configure_expandable_page(self.tab_odds_api, self.odds_api_tab)
        self.betnow_tab = None
        self.dk_tab = None
        self.mybookie_tab = None
        if self.tab_betnow is not None:
            self.betnow_tab = BookTab(self.tab_betnow, "BetNow.eu")
            _configure_expandable_page(self.tab_betnow, self.betnow_tab)
        if self.tab_dk is not None:
            self.dk_tab = BookTab(self.tab_dk, "DraftKings")
            _configure_expandable_page(self.tab_dk, self.dk_tab)
        if self.tab_mybookie is not None:
            self.mybookie_tab = BookTab(self.tab_mybookie, "MyBookie")
            _configure_expandable_page(self.tab_mybookie, self.mybookie_tab)

        self.arb_tab = ArbScannerTab(
            self.tab_arb,
            payload_getter=lambda: self._payload,
            budget_getter=self._current_budget_state,
        )
        _configure_expandable_page(self.tab_arb, self.arb_tab)

        self.tab_next_two.grid_columnconfigure(0, weight=1)
        self.tab_next_two.grid_rowconfigure(0, weight=1)
        self.next_two_scroll = ctk.CTkScrollableFrame(
            self.tab_next_two, label_text="Upcoming cards (closest first)"
        )
        self.next_two_scroll.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.props_odds_api_tab = BookPropsTab(
            self.tab_props_odds_api,
            book_name="Odds API",
            book_note="Over 1.5 totals from The Odds API (HA actionable)",
            show_parlays=False,
            show_all_var=self.show_all_props_var,
            profile_getter=lambda: self._profile_from_menu(self.profile_var.get()),
        )
        _configure_expandable_page(self.tab_props_odds_api, self.props_odds_api_tab)
        self.props_betnow_tab = None
        self.props_dk_tab = None
        self.props_mybookie_tab = None
        if self.tab_props_betnow is not None:
            self.props_betnow_tab = BookPropsTab(
                self.tab_props_betnow,
                book_name="BetNow.eu",
                book_note="Singles only - props cannot be parlayed",
                show_parlays=False,
                show_all_var=self.show_all_props_var,
                profile_getter=lambda: self._profile_from_menu(self.profile_var.get()),
            )
            _configure_expandable_page(self.tab_props_betnow, self.props_betnow_tab)
        if self.tab_props_dk is not None:
            self.props_dk_tab = BookPropsTab(
                self.tab_props_dk,
                book_name="DraftKings",
                book_note="Singles + 2-3 leg prop/mixed parlays (correlation-adjusted)",
                show_parlays=True,
                show_all_var=self.show_all_props_var,
                profile_getter=lambda: self._profile_from_menu(self.profile_var.get()),
            )
            _configure_expandable_page(self.tab_props_dk, self.props_dk_tab)
        if self.tab_props_mybookie is not None:
            self.props_mybookie_tab = BookPropsTab(
                self.tab_props_mybookie,
                book_name="MyBookie",
                book_note="Singles + 2-3 leg prop/mixed parlays (live method/round props when scraped)",
                show_parlays=True,
                show_all_var=self.show_all_props_var,
                profile_getter=lambda: self._profile_from_menu(self.profile_var.get()),
            )
            _configure_expandable_page(self.tab_props_mybookie, self.props_mybookie_tab)

        # Use CTkTabview's command hook - do NOT override _segmented_button.command
        # (that bypasses internal tab switching and breaks all tab clicks).
        self.tabs.configure(command=self._on_tab_changed)

        self.tab_risk.grid_columnconfigure(0, weight=1)
        self.tab_risk.grid_rowconfigure(0, weight=1)
        self.risk_page = ctk.CTkScrollableFrame(
            self.tab_risk, fg_color="transparent", label_text="Risk Analysis"
        )
        self.risk_page.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.risk_summary = ctk.CTkLabel(self.risk_page, text="", anchor="w", justify="left")
        self.risk_summary.pack(fill="x", padx=12, pady=(8, 4))
        self.skip_scorecard_frame = ctk.CTkFrame(self.risk_page, fg_color="transparent")
        self.skip_scorecard_frame.pack(fill="x", padx=12, pady=(0, 4))
        self.sleeve_stats_frame = ctk.CTkFrame(self.risk_page, fg_color="transparent")
        self.sleeve_stats_frame.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            self.risk_page,
            text="Model Insights - discovered interaction features",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
            text_color="#cbd5e1",
        ).pack(fill="x", padx=12, pady=(2, 2))
        self.model_insights_box = ctk.CTkLabel(
            self.risk_page,
            text="Train the model to populate interaction discoveries.",
            anchor="w",
            justify="left",
            font=ctk.CTkFont(size=12),
            text_color="#94a3b8",
            wraplength=1100,
        )
        self.model_insights_box.pack(fill="x", padx=12, pady=(0, 6))
        self.risk_scroll = ctk.CTkFrame(self.risk_page, fg_color="transparent")
        self.risk_scroll.pack(fill="x", padx=10, pady=(0, 8))
        self.risk_chart_frame = ctk.CTkFrame(self.risk_scroll)
        self.risk_chart_frame.pack(fill="x", padx=4, pady=6)
        self._risk_fig = Figure(figsize=(6.5, 2.6), dpi=100, facecolor="#1a1a1a")
        self._risk_canvas = FigureCanvasTkAgg(self._risk_fig, master=self.risk_chart_frame)
        self._risk_canvas.get_tk_widget().pack(fill="x")
        self.threshold_chart_frame = ctk.CTkFrame(self.risk_scroll)
        self.threshold_chart_frame.pack(fill="x", padx=4, pady=6)
        self._thresh_fig = Figure(figsize=(6.5, 2.2), dpi=100, facecolor="#1a1a1a")
        self._thresh_canvas = FigureCanvasTkAgg(self._thresh_fig, master=self.threshold_chart_frame)
        self._thresh_canvas.get_tk_widget().pack(fill="x")

    def _on_profile_change(self, value: str) -> None:
        _debug_log(f"Profile changed: {value}")
        config.UFC_PROFILE = self._profile_from_menu(value)
        config.apply_profile_overrides()
        self._update_mode_banner()
        if config.is_live_profile():
            self.show_all_props_var.set(False)
        elif config.is_paper_profile():
            self.show_all_props_var.set(config.PAPER_PROPS_SHOW_ALL_DEFAULT)
        if hasattr(self, "budget_bar"):
            # Refresh card-budget default for new Paper/Live risk % (keep override if set)
            self.budget_bar.apply_profile_defaults(force_card=False)
            self.budget_bar.refresh_warnings()
        if self._payload is not None:
            self._render_all_tabs(self._payload)

    def _reload_runtime_config(self, *, apply_bankroll_defaults: bool = True) -> dict[str, Any]:
        """Re-read .env + refresh_runtime_env; optionally sync bankroll UI defaults."""
        from src.project_paths import reload_runtime_env

        root = reload_runtime_env(
            log=_debug_log if _DEBUG_MODE else None,
        )
        config.apply_profile_overrides()
        key_len = len(str(getattr(config, "ODDS_API_KEY", "") or ""))
        key_src = str(getattr(config, "ODDS_API_KEY_SOURCE", "") or "")
        from src.odds_providers.odds_api_client import (
            clear_odds_api_session,
            key_last4,
            refresh_odds_api_runtime,
        )

        # Always re-pick newest project/dist .env key and drop stale HTTP session
        odds_meta = refresh_odds_api_runtime(root=root)
        clear_odds_api_session()
        key_len = int(odds_meta.get("key_length") or key_len)
        key_src = str(odds_meta.get("key_source") or key_src)
        last4 = str(odds_meta.get("key_last4") or key_last4(config.ODDS_API_KEY))
        summary = {
            "root": str(root),
            "profile": config.UFC_PROFILE,
            "odds_api_key_len": key_len,
            "odds_api_key_source": key_src,
            "odds_api_key_last4": last4,
            "odds_api_sport": str(getattr(config, "ODDS_API_SPORT", "")),
            "odds_api_regions": str(getattr(config, "ODDS_API_REGIONS", "")),
            "initial_bankroll": float(getattr(config, "INITIAL_BANKROLL", 0) or 0),
            "card_budget": float(getattr(config, "CARD_BUDGET", 0) or 0),
            "action_network": bool(getattr(config, "ACTION_NETWORK_ENABLED", True)),
            "betnow": bool(getattr(config, "BETNOW_ENABLED", False)),
            "mybookie": bool(getattr(config, "MYBOOKIE_ENABLED", False)),
            "draftkings": bool(getattr(config, "DRAFTKINGS_ENABLED", False)),
            "cache_ttl_min": int(getattr(config, "ODDS_CACHE_TTL_MINUTES", 20) or 20),
        }
        _debug_log(
            "Reload Config: "
            f"profile={summary['profile']} bankroll={summary['initial_bankroll']} "
            f"odds_key_len={key_len} last4={last4 or '-'} source={key_src or '-'} "
            f"sport={summary['odds_api_sport']} regions={summary['odds_api_regions']} "
            f"ttl={summary['cache_ttl_min']}m "
            f"AN={summary['action_network']} BN={summary['betnow']} "
            f"DK={summary['draftkings']} MB={summary['mybookie']}"
        )
        _dashboard_heartbeat(
            f"config reloaded profile={summary['profile']} "
            f"bankroll={summary['initial_bankroll']}"
        )

        self._sync_profile_menu(config.UFC_PROFILE)
        if apply_bankroll_defaults and hasattr(self, "budget_bar"):
            try:
                br = float(config.INITIAL_BANKROLL)
                self.budget_bar.total_bankroll_var.set(
                    f"{br:.2f}".rstrip("0").rstrip(".")
                )
                self.budget_bar.apply_profile_defaults(force_card=False)
                self.budget_bar.refresh_warnings()
                self._budget_state = self.budget_bar.get_state()
            except Exception as exc:
                _debug_log(f"Reload Config bankroll UI sync skipped: {exc}")
        self._update_mode_banner()
        if hasattr(self, "grok_panel"):
            try:
                self.grok_panel.refresh_model_choices()
            except Exception:
                pass
        if hasattr(self, "grok_btn"):
            try:
                from src.grok_analysis import grok_available

                ok = grok_available()
                self.grok_btn.configure(
                    fg_color="#0f766e" if ok else "#374151",
                    hover_color="#14b8a6" if ok else "#4b5563",
                )
            except Exception:
                pass
        return summary

    def _on_reload_config(self) -> None:
        try:
            self._reload_runtime_config(apply_bankroll_defaults=True)
            self._set_status("Config reloaded")
        except Exception as exc:
            tb = traceback.format_exc()
            _debug_log(f"Reload Config failed: {tb}")
            self._set_status(f"Config reload failed: {exc}")
            self._show_error(f"Reload Config: {exc}")

    def _on_soft_update(self) -> None:
        """Reload config + Quick Odds in one click."""
        _debug_log("Soft Update: reload config then Quick Odds")
        _dashboard_heartbeat("soft update start")
        try:
            self._reload_runtime_config(apply_bankroll_defaults=True)
        except Exception as exc:
            _debug_log(f"Soft Update config reload failed: {exc}")
            self._set_status(f"Soft Update failed (config): {exc}")
            return
        if self._payload is None or not self._payload.books:
            self._set_status("Config reloaded — run Refresh Next Two before Soft Update odds")
            _dashboard_heartbeat("soft update: config only (no card)")
            return
        self._set_status("Config reloaded — Quick Odds refresh...")
        _dashboard_heartbeat("soft update: quick odds")
        self._run_quick_odds_async(auto=False)

    def _restart_command(self) -> list[str]:
        """Build argv to relaunch the same entrypoint (python script or frozen EXE)."""
        if getattr(sys, "frozen", False):
            return [sys.executable, *sys.argv[1:]]
        script = Path(__file__).resolve()
        args = [a for a in sys.argv[1:]]
        return [sys.executable, "-u", str(script), *args]

    def _on_restart_app(self) -> None:
        """Cleanly quit and relaunch so .env is re-read on startup."""
        import subprocess

        cmd = self._restart_command()
        cwd = str(getattr(config, "ROOT_DIR", _ROOT) or _ROOT)
        _debug_log(f"Restart App: spawning {cmd!r} cwd={cwd}")
        _dashboard_heartbeat(f"restart app: {' '.join(str(c) for c in cmd[:3])}...")
        self._set_status("Restarting dashboard...")
        try:
            if hasattr(self, "budget_bar"):
                try:
                    self.budget_bar._persist_bankroll()
                except Exception:
                    pass
            popen_kwargs: dict[str, Any] = {
                "cwd": cwd,
                "env": os.environ.copy(),
                "close_fds": False,
            }
            if sys.platform == "win32":
                # New process group so the child survives parent teardown.
                # Avoid DETACHED_PROCESS for GUI EXE (can hide the new window).
                flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if not getattr(sys, "frozen", False):
                    flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                popen_kwargs["creationflags"] = flags
            subprocess.Popen(cmd, **popen_kwargs)
        except Exception as exc:
            tb = traceback.format_exc()
            _debug_log(f"Restart App spawn failed: {tb}")
            self._set_status(f"Restart failed: {exc}")
            self._show_error(f"Restart App: {exc}")
            return
        self.after(250, self._quit_after_restart)

    def _quit_after_restart(self) -> None:
        _dashboard_heartbeat("restart app: quitting old process")
        try:
            self.destroy()
        except Exception:
            pass
        try:
            self.quit()
        except Exception:
            pass
        os._exit(0)

    def _on_budget_live_change(self, state: dict[str, Any]) -> None:
        self._budget_state = state
        if self._updating_budget:
            return
        if self._budget_after_id is not None:
            try:
                self.after_cancel(self._budget_after_id)
            except Exception:
                pass
        self._budget_after_id = self.after(10, self._apply_budget_live_change)

    def _apply_budget_live_change(self) -> None:
        self._budget_after_id = None
        if self._updating_budget:
            return
        self._updating_budget = True
        try:
            self._update_mode_banner()
            self._refresh_top_recommendations()
        finally:
            self._updating_budget = False

    def _on_budget_saved(self, state: dict[str, Any]) -> None:
        self._budget_state = state
        config.apply_profile_overrides()
        self._update_mode_banner()
        self._set_status("Bankroll saved")
        if self._payload is not None:
            self._render_all_tabs(self._payload)

    def _on_auto_watch_toggle(self) -> None:
        # Auto Watch UI removed; keep hook no-op safe if called.
        self._auto_watch = bool(
            getattr(self, "auto_watch_var", None) and self.auto_watch_var.get()
        )
        now = time.time()
        if self._auto_watch:
            self._next_card_ts = now + config.DASHBOARD_CARD_CHECK_MINUTES * 60
            self._next_odds_ts = now + config.DASHBOARD_AUTO_ODDS_MINUTES * 60
            self._set_status("Auto watch enabled - monitoring card + odds.")
            self._auto_watch_tick()
        else:
            self._next_card_ts = None
            self._next_odds_ts = None
            self._set_status("Auto watch disabled.")
        self._update_status_bar()

    def _auto_watch_tick(self) -> None:
        if not self._auto_watch:
            return
        now = time.time()
        if self._next_card_ts and now >= self._next_card_ts and not self._busy:
            self._next_card_ts = now + config.DASHBOARD_CARD_CHECK_MINUTES * 60
            threading.Thread(target=self._check_card_change_worker, daemon=True).start()
        if (
            self._next_odds_ts
            and now >= self._next_odds_ts
            and not self._busy
            and self._payload is not None
            and not _df_is_empty(self._payload.combined)
        ):
            self._next_odds_ts = now + config.DASHBOARD_AUTO_ODDS_MINUTES * 60
            self._run_quick_odds_async(auto=True)
        self.after(30_000, self._auto_watch_tick)

    def _check_card_change_worker(self) -> None:
        try:
            changed, event_name, _ = detect_card_change(event_index=0)
            if changed:
                self.after(
                    0,
                    lambda: self._on_new_card_detected(event_name),
                )
        except Exception as exc:
            self.after(0, lambda: self._set_status(f"Card check failed: {exc}"))

    def _on_new_card_detected(self, event_name: str) -> None:
        self._set_status(f"New card detected: {event_name} - running analysis...")
        if event_name and event_name not in self.event_menu.cget("values"):
            vals = list(self.event_menu.cget("values"))
            if event_name not in vals:
                vals.insert(0, event_name)
                self.event_menu.configure(values=vals)
        self.event_var.set(event_name if event_name else "Next Card")
        self._run_new_card_analysis(event_name)

    def _on_progress(self, msg: str, pct: float | None = None) -> None:
        self._set_status(msg)
        self._show_progress(pct)
        if hasattr(self, "mode_banner_label"):
            self.mode_banner_label.configure(text=f" {msg}")
        try:
            self.update_idletasks()
        except Exception:
            pass

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        _debug_log(f"Busy={busy} - toolbar state={state}")

        if self._busy_watchdog_id is not None:
            try:
                self.after_cancel(self._busy_watchdog_id)
            except Exception:
                pass
            self._busy_watchdog_id = None
        if busy:
            self._busy_watchdog_id = self.after(600_000, self._busy_watchdog)

        self.refresh_btn.configure(state=state)
        if hasattr(self, "soft_update_btn"):
            self.soft_update_btn.configure(state=state)
        if not busy and hasattr(self, "soft_update_btn"):
            self.soft_update_btn.configure(fg_color="#1d4ed8", hover_color="#2563eb")
        # Restart stays usable even while busy (escape hatch)
        if hasattr(self, "restart_app_btn"):
            self.restart_app_btn.configure(state="normal")
        if hasattr(self, "refresh_btn") and not busy:
            self.refresh_btn.configure(fg_color="#2563eb", hover_color="#3b82f6")
        self.profile_menu.configure(state=state)
        self.event_menu.configure(state=state)

    def _busy_watchdog(self) -> None:
        self._busy_watchdog_id = None
        if not self._busy:
            return
        _debug_log("Busy watchdog fired - resetting stuck busy state")
        self._set_status("Operation timed out - controls re-enabled. Try Refresh again.")
        self._finish_busy()

    def _log_loaded_fights(self, source: str, payload: DashboardPayload) -> None:
        """Debug log for fight row counts after refresh or cache load."""
        combined = _as_dataframe(getattr(payload, "combined", None))
        n_combined = _df_row_count(combined)
        ov = _as_dataframe(payload.books.get("Overview", {}).get("predictions"))
        n_overview = _df_row_count(ov)
        card_counts = [
            _df_row_count(c.get("predictions"))
            for c in (payload.cards or [])
            if isinstance(c.get("predictions"), pd.DataFrame) and not c["predictions"].empty
        ]
        for idx, card in enumerate(payload.cards or []):
            ev = card.get("event_name", "?")
            cn = _df_row_count(card.get("predictions"))
            _debug_log(f"Loaded Card {idx}: {ev} - {cn} fights")
        _debug_log(
            f"{source}: fights loaded combined={n_combined} overview={n_overview} "
            f"cards={len(payload.cards or [])} per_card={card_counts} label={payload.event_label!r}"
        )

    def _on_refresh(self) -> None:
        """Load next two cards (predictions); odds reuse cache when ODDS_FETCH_ONCE."""
        self._run_refresh_next_two()

    def _run_refresh_next_two(self) -> None:
        """Refresh Next Two worker — UFC.com cards + cached Odds API merge."""
        if self._busy:
            _button_debug("Refresh Next Two ignored - already busy")
            self._set_status("Already running - wait for the current operation to finish.")
            return
        if run_full_analysis is None:
            self._set_status("Dashboard still loading - wait 5s and try Refresh again.")
            return
        self._set_busy(True)
        self.event_var.set("Next Two Cards")
        self._on_progress(
            "Refresh: next two upcoming cards (predictions + odds)...", 0.02
        )
        _button_debug(
            "Refresh Next Two -> run_dashboard_analysis(event_mode='Next Two Cards') "
            "-> run_full_analysis() -> load_next_two_cards()"
        )

        profile = self._profile_from_menu(self.profile_var.get())
        budget = self._current_budget_state()

        def worker() -> None:
            try:
                _ensure_props_config()
                prog = lambda m, p=None: self.after(
                    0, lambda msg=m, pct=p: self._on_progress(msg, pct)
                )
                try:
                    from src.data_loader import clear_stale_upcoming_card_caches

                    cleared = clear_stale_upcoming_card_caches(max_age_hours=24.0, force=True)
                    if cleared:
                        _button_debug(f"Refresh Next Two cleared card caches: {cleared}")
                except Exception as clear_exc:
                    _debug_log(f"Card cache clear skipped: {clear_exc}")

                payload = run_dashboard_analysis(
                    event_mode="Next Two Cards",
                    profile=profile,
                    # ODDS_FETCH_ONCE: reuse first odds download; do not re-pull
                    force_refresh_odds=False,
                    explain=False,
                    use_cache=True,
                    budget_state=budget,
                    progress=prog,
                )
                card_names = [
                    str(c.get("event_name") or "")
                    for c in (payload.cards or [])
                    if str(c.get("event_name") or "").strip()
                ]
                _button_debug(f"Loading Next Two Cards: {card_names}")
                if _df_is_empty(getattr(payload, "combined", None)):
                    from src.dashboard_service import _background_analysis_fallback

                    fb = _background_analysis_fallback(profile=profile)
                    if fb:
                        payload = _result_to_payload({**fb, "profile": profile})
                        _button_debug("Refresh Next Two: applied background cache fallback (live empty)")
                    else:
                        _button_debug(
                            "Refresh Next Two: live empty and no matching background cache "
                            "(refusing stale Freedom 250)"
                        )

                def _done() -> None:
                    n_fights = _df_row_count(getattr(payload, "combined", None))
                    n_cards = _df_row_count(getattr(payload, "cards", None))
                    _button_debug(
                        f"Refresh Next Two complete -> _apply_payload "
                        f"({n_fights} fights, {n_cards} cards, "
                        f"events={card_names})"
                    )
                    self._apply_payload(payload, full_refresh=True, odds_refresh=True)
                    self._finish_busy()
                    self._update_mode_banner()

                self.after(0, _done)
            except Exception as exc:
                tb = traceback.format_exc()
                _debug_log(tb)
                try:
                    from src.dashboard_service import _background_analysis_fallback

                    fb = _background_analysis_fallback(profile=profile)
                    if fb:
                        payload = _result_to_payload(fb)

                        def _fallback_done() -> None:
                            self._apply_payload(payload, full_refresh=True, odds_refresh=True)
                            self._set_status(
                                f"Loaded cached fights - live refresh failed ({exc})"
                            )
                            self._finish_busy()

                        self.after(0, _fallback_done)
                        return
                except Exception:
                    pass
                self.after(0, lambda: self._show_error(f"{exc}\n{tb}"))
                self.after(0, self._finish_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _on_process_new_card(self) -> None:
        if self._busy:
            _button_debug("Process New Card ignored - already busy")
            self._set_status("Already running - wait for the current operation to finish.")
            return
        self._set_busy(True)
        self._on_progress("Checking for new UFC card...", 0.05)
        _button_debug("Process New Card -> detect_card_change(event_index=0)")

        def worker() -> None:
            delegated = False
            try:
                changed, event_name, _ = detect_card_change(event_index=0)
                if not changed:
                    _button_debug("Process New Card: no card change detected")
                    self.after(
                        0,
                        lambda: self._set_status("No new card detected - cache is current."),
                    )
                    return
                delegated = True
                _button_debug(f"Process New Card: change detected -> _run_new_card_analysis({event_name!r})")
                self.after(0, lambda: self._run_new_card_analysis(event_name))
            except Exception as exc:
                tb = traceback.format_exc()
                _debug_log(tb)
                self.after(0, lambda: self._show_error(str(exc)))
            finally:
                if not delegated:
                    self.after(0, self._finish_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _run_new_card_analysis(self, event_name: str) -> None:
        self._set_busy(True)
        self._on_progress(f"New card: {event_name} - full analysis...", 0.05)
        if event_name:
            vals = list(self.event_menu.cget("values"))
            if event_name not in vals:
                vals.insert(0, event_name)
                self.event_menu.configure(values=vals)
            self.event_var.set(event_name)

        event_mode = self.event_var.get()
        profile = self._profile_from_menu(self.profile_var.get())
        budget = self._current_budget_state()
        _button_debug(
            f"Process New Card -> run_dashboard_analysis(event_mode={event_mode!r}) "
            f"-> run_full_analysis()"
        )

        def worker() -> None:
            try:
                payload = run_dashboard_analysis(
                    event_mode=event_mode,
                    profile=profile,
                    # ODDS_FETCH_ONCE: reuse first odds download; do not re-pull
                    force_refresh_odds=False,
                    explain=True,
                    use_cache=True,
                    budget_state=budget,
                    progress=lambda m, p=None: self.after(0, lambda msg=m, pct=p: self._on_progress(msg, pct)),
                )

                def _done() -> None:
                    _button_debug(
                        f"Process New Card complete -> _apply_payload "
                        f"({_df_row_count(payload.combined)} fights)"
                    )
                    self._apply_payload(payload, full_refresh=True, odds_refresh=True)
                    self._finish_busy()

                self.after(0, _done)
            except Exception as exc:
                tb = traceback.format_exc()
                _debug_log(tb)
                self.after(0, lambda: self._show_error(f"{exc}\n{tb}"))
            finally:
                self.after(0, self._finish_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _on_quick_odds(self) -> None:
        if self._busy:
            _button_debug("Quick Odds + Props ignored - already busy")
            self._set_status("Already running - wait for the current operation to finish.")
            return
        if self._payload is None or _df_is_empty(self._payload.combined):
            self._set_status("Run Full Refresh first - need cached predictions.")
            return
        _button_debug(
            f"Quick Odds + Props -> _run_quick_odds_async() -> run_quick_odds_refresh() "
            f"({_df_row_count(self._payload.combined)} base fights)"
        )
        self._run_quick_odds_async(auto=False)

    def _on_refresh_props(self) -> None:
        """Fast props-only refresh (~30-90s) using cached predictions."""
        if self._busy:
            _button_debug("Refresh Props ignored - already busy")
            self._set_status("Already running - wait for the current operation to finish.")
            return
        if self._payload is None or _df_is_empty(self._payload.combined):
            self._set_status("Load fights first - Refresh Next Two or wait for nightly cache.")
            return
        if not _ensure_props_config():
            self._set_status("Props disabled - set ENABLE_PROPS=true in .env")
            return
        _button_debug("Refresh Props -> _run_quick_props_async() -> run_quick_props_refresh()")
        self._run_quick_props_async()

    def _run_quick_props_async(self) -> None:
        if self._busy or self._payload is None:
            return
        self._set_busy(True)
        self._on_progress("Props refresh (parallel, no ML re-run)...", 0.05)
        books_in = dict(self._payload.books)
        budget = self._current_budget_state()

        def worker() -> None:
            try:
                _button_debug(f"run_quick_props_refresh(books={list(books_in.keys())})")
                result = run_quick_props_refresh(
                    books_in,
                    budget_state=budget,
                    progress=lambda m, p=None: self.after(0, lambda msg=m, pct=p: self._on_progress(msg, pct)),
                )
                self.after(0, lambda: self._apply_quick_props(result))
            except Exception as exc:
                tb = traceback.format_exc()
                _debug_log(tb)
                self.after(0, lambda: self._show_error(str(exc)))
            finally:
                self.after(0, self._finish_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_quick_props(self, result: dict[str, Any]) -> None:
        if self._payload is None:
            return
        for name, data in (result.get("books") or {}).items():
            if name not in self._payload.books:
                self._payload.books[name] = data
            else:
                self._payload.books[name]["props"] = data.get("props") or {}
        ts = result.get("props_updated_at") or "now"
        counts: list[str] = []
        for book in ("Odds API", "BetNow.eu", "DraftKings", "MyBookie"):
            props = self._payload.books.get(book, {}).get("props") or {}
            n = len(props.get("singles") or [])
            counts.append(f"{book}={n}")
            _button_debug(f"Refresh Props: {book} -> {n} ranked singles")
        self._set_status(
            f"Props updated at {ts} - {', '.join(counts)}. Open Props tabs to review."
        )
        self._render_props_section(self._payload)
        self._rendered_sections.add("props")
        current = self.tabs.get() if hasattr(self, "tabs") else ""
        if current == "Odds API Props" or current.startswith("Props - "):
            book_key = self._props_tab_book_key(current)
            if book_key:
                self._render_single_props_tab(book_key)
        _button_debug(
            f"Refresh Props complete -> _render_props_section (books={list(self._payload.books.keys())})"
        )

    def _run_quick_odds_async(self, *, auto: bool) -> None:
        if self._busy or self._payload is None:
            return
        self._set_busy(True)
        books_q = "Odds API"
        if getattr(config, "BETNOW_ENABLED", False):
            books_q += " + BetNow"
        if getattr(config, "DRAFTKINGS_ENABLED", False):
            books_q += " + DraftKings"
        if config.MYBOOKIE_ENABLED:
            books_q += " + MyBookie"
        label = "Auto quick odds..." if auto else f"Quick odds refresh ({books_q})..."
        self._on_progress(label, 0.1)
        base = self._payload.combined.copy()
        event_label = self._payload.event_label
        budget = self._current_budget_state()

        def worker() -> None:
            try:
                _button_debug(
                    f"run_quick_odds_refresh({len(base)} fights, event_label={event_label!r})"
                )
                result = run_quick_odds_refresh(
                    base,
                    event_label=event_label,
                    budget_state=budget,
                    progress=lambda m, p=None: self.after(0, lambda msg=m, pct=p: self._on_progress(msg, pct)),
                )
                self.after(0, lambda: self._apply_quick_odds(result))
            except Exception as exc:
                self.after(0, lambda: self._show_error(str(exc)))
            finally:
                self.after(0, self._finish_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_quick_odds(self, result: dict[str, Any]) -> None:
        if self._payload is None:
            return
        books = result.get("books", {})
        for name, data in books.items():
            self._payload.books[name] = data
        if result.get("threshold_ctx"):
            self._payload.threshold_ctx = result["threshold_ctx"]
        if result.get("arb_scan"):
            self._payload.arb_scan = result["arb_scan"]
        self._payload.odds_updated_at = result.get("odds_updated_at", self._payload.odds_updated_at)
        odds_status = self._odds_api_status_from_books(books)
        _button_debug(
            f"Quick Odds + Props complete -> _apply_payload(odds_refresh=True) "
            f"books={list(books.keys())} | {odds_status}"
        )
        self._apply_payload(
            self._payload,
            odds_refresh=True,
            quick=True,
            odds_status=odds_status,
        )
        # Force Odds API tab re-render immediately (lazy tab may be stale)
        try:
            if getattr(self, "odds_api_tab", None) is not None:
                self._render_single_book_tab("Odds API")
        except Exception as exc:
            _debug_log(f"Odds API tab re-render after Soft Update failed: {exc}")

    @staticmethod
    def _odds_api_status_from_books(books: dict[str, Any]) -> str:
        data = books.get("Odds API") or {}
        meta = data.get("odds_match_meta") or {}
        if meta.get("status_line"):
            line = str(meta["status_line"])
            rem = meta.get("requests_remaining")
            last4 = meta.get("key_last4")
            extra = []
            if last4:
                extra.append(f"last4={last4}")
            if rem is not None and str(rem) != "":
                extra.append(f"remaining={rem}")
            if extra:
                line = f"{line} ({', '.join(extra)})"
            warn = str(data.get("warning") or "")
            if int(meta.get("matched") or data.get("odds_matched") or 0) == 0 and warn:
                # Prefer concrete mismatch reason over generic quick-odds text
                return warn if len(warn) < 280 else line + " — " + warn[:180]
            return line
        matched = int(data.get("odds_matched") or 0)
        total = int(data.get("odds_total") or 0)
        warn = str(data.get("warning") or data.get("error") or "").strip()
        if warn and matched == 0:
            return warn[:280]
        return f"Odds API: {matched}/{total} lines matched"

    def _finish_busy(self) -> None:
        self._set_busy(False)
        self._show_progress(None)
        if not self._grok_busy:
            try:
                if hasattr(self, "grok_btn"):
                    self.grok_btn.configure(state="normal")
            except Exception:
                pass
        pending = self._pending_tab_after_busy
        if pending:
            self._pending_tab_after_busy = None
            self.after(100, lambda tn=pending: self._handle_tab_selected(tn))

    def _show_error(self, msg: str) -> None:
        self._set_status(f"Error: {msg[:240]}")

    def _apply_payload(
        self,
        payload: DashboardPayload,
        *,
        full_refresh: bool = False,
        odds_refresh: bool = False,
        quick: bool = False,
        odds_status: str | None = None,
    ) -> None:
        self._log_loaded_fights("_apply_payload", payload)
        self._payload = payload
        now = time.time()
        if full_refresh:
            self._last_full_refresh_ts = now
            if hasattr(self, "arb_tab"):
                self.arb_tab.reset_alerts()
        if odds_refresh or full_refresh:
            if hasattr(self, "arb_tab") and bool(self.arb_tab.watch_var.get()):
                self.arb_tab.start_watch()
        if odds_refresh:
            self._last_odds_ts = now
            if self._auto_watch:
                self._next_odds_ts = now + config.DASHBOARD_AUTO_ODDS_MINUTES * 60
        self._update_status_bar()
        cache_note = " (cached features)" if payload.from_cache else ""
        quick_note = " - quick odds" if quick else ""
        self.meta_label.configure(
            text=f"{payload.generated_at or '-'}  |  {payload.event_label}  |  {config.normalize_profile(payload.profile)}{cache_note}"
        )
        if payload.errors:
            self._set_status("Done with warnings: " + "; ".join(payload.errors[:2]) + quick_note)
        elif odds_status:
            self._set_status(odds_status)
        elif quick:
            self._set_status(f"Quick odds + props updated at {payload.odds_updated_at or 'now'}.")
        else:
            self._set_status(f"Refresh complete - {payload.event_label}")

        try:
            self._schedule_render_all_tabs(payload)
            self.after(50, self._refresh_top_recommendations)
            book_keys = [k for k in payload.books if k != "Overview"]
            _button_debug(
                f"_apply_payload -> UI tabs: overview+books+props "
                f"(combined={_df_row_count(payload.combined)}, cards={_df_row_count(payload.cards)}, "
                f"books={book_keys})"
            )
        except Exception as exc:
            tb = traceback.format_exc()
            _debug_log(f"Tab render error: {tb}")
            self._show_error(f"Tab render failed: {exc}")

    def _on_tab_changed(self) -> None:
        """CTkTabview command callback - runs after the tab has switched."""
        tab_name = self.tabs.get()
        print(f"Button [Tab {tab_name}] clicked", flush=True)
        _button_debug(f"Tab [{tab_name}] clicked")
        self._tab_render_seq += 1
        seq = self._tab_render_seq
        self.after_idle(lambda tn=tab_name, s=seq: self._handle_tab_selected(tn, seq))

    def _handle_tab_selected(self, tab_name: str, seq: int | None = None) -> None:
        """Lazy-render heavy tabs on first visit; re-render book/props tabs on each switch."""
        if seq is not None and seq != self._tab_render_seq:
            return
        if self._busy:
            self._pending_tab_after_busy = tab_name
            self._set_status(
                f"Still loading - {tab_name} will open when refresh finishes..."
            )
            _button_debug(f"Tab [{tab_name}] deferred - refresh in progress")
            return
        try:
            self._handle_tab_selected_inner(tab_name)
        except Exception as exc:
            tb = traceback.format_exc()
            _debug_log(f"Tab [{tab_name}] render error: {tb}")
            try:
                import logging

                logging.getLogger("ufc_dashboard").error(
                    "Tab [%s] render failed: %s\n%s", tab_name, exc, tb
                )
            except Exception:
                pass
            self._show_error(f"{tab_name} tab failed: {exc}")

    def _handle_tab_selected_inner(self, tab_name: str) -> None:
        if tab_name in ("Odds API", "BetNow.eu", "DraftKings", "MyBookie"):
            if self._payload is None:
                _button_debug(f"Tab [{tab_name}]: no payload yet - run Refresh Next Two")
                return
            _button_debug(f"Switching to {tab_name} tab")
            self._render_single_book_tab(tab_name)
            return
        if tab_name == "Arb Scanner":
            if self._payload is None:
                _button_debug("Tab [Arb Scanner]: no payload yet - run Refresh Next Two")
                return
            self._render_arb_section(self._payload)
            return
        if tab_name in ("Odds API Props",) or tab_name.startswith("Props - "):
            book_key = self._props_tab_book_key(tab_name)
            if book_key:
                if self._payload is None:
                    _button_debug(f"Tab [{tab_name}]: no payload yet - run Refresh Next Two")
                    return
                self._render_single_props_tab(book_key)
            return
        self._render_tab_lazy(tab_name)

    @staticmethod
    def _props_tab_book_key(tab_name: str) -> str | None:
        return {
            "Odds API Props": "Odds API",
            "Props - BetNow": "BetNow.eu",
            "Props - DraftKings": "DraftKings",
            "Props - MyBookie": "MyBookie",
        }.get(tab_name)

    def _render_single_book_tab(self, book_key: str) -> None:
        payload = self._payload
        if payload is None:
            return
        tab_widgets = {
            "Odds API": getattr(self, "odds_api_tab", None),
            "BetNow.eu": getattr(self, "betnow_tab", None),
            "DraftKings": getattr(self, "dk_tab", None),
            "MyBookie": getattr(self, "mybookie_tab", None),
        }
        tab = tab_widgets.get(book_key)
        if tab is None:
            return
        data = self._book_tab_data(payload, book_key)
        data["cards"] = payload.cards
        data["_payload"] = payload
        preds = data.get("predictions", pd.DataFrame())
        matched = int(data.get("odds_matched") or 0)
        if isinstance(preds, pd.DataFrame) and not preds.empty and "odds_matched" in preds.columns:
            matched = int(preds["odds_matched"].sum())
        fight_n = len(preds) if isinstance(preds, pd.DataFrame) else 0
        _button_debug(f"Rendering {book_key}: {fight_n} fights, {matched} with {book_key} odds")
        ctx = payload.threshold_ctx or {}
        bs = self._budget_state
        profile = self._profile_from_menu(self.profile_var.get())
        tab.render(data, ctx, budget_state=bs, profile=profile)

    def _render_single_props_tab(self, book_key: str) -> None:
        payload = self._payload
        if payload is None:
            return
        tab_widgets = {
            "Odds API": getattr(self, "props_odds_api_tab", None),
            "BetNow.eu": getattr(self, "props_betnow_tab", None),
            "DraftKings": getattr(self, "props_dk_tab", None),
            "MyBookie": getattr(self, "props_mybookie_tab", None),
        }
        tab = tab_widgets.get(book_key)
        if tab is None:
            return
        props = payload.books.get(book_key, {}).get("props") or {}
        n = len(props.get("singles") or [])
        _button_debug(f"Props tab {book_key}: {n} singles in payload")
        bs = self._budget_state
        profile = self._profile_from_menu(self.profile_var.get())
        tab.render(payload, budget_state=bs, profile=profile)
        self._rendered_sections.add("props")

    def _render_tab_lazy(self, tab_name: str) -> None:
        payload = self._payload
        if payload is None:
            return
        if tab_name == "Next Two Cards" and "next_two" not in self._rendered_sections:
            self._render_next_two_section(payload)
            self._rendered_sections.add("next_two")
        elif tab_name == "Risk Analysis" and "risk" not in self._rendered_sections:
            self._render_risk_section(payload)
            self._rendered_sections.add("risk")
        elif tab_name == "Ollama Analysis" and "grok" not in self._rendered_sections:
            self._render_grok_section()
            self._rendered_sections.add("grok")
        elif tab_name == "Arb Scanner" and "arb" not in self._rendered_sections:
            self._render_arb_section(payload)
            self._rendered_sections.add("arb")

    def _render_arb_section(self, payload: "DashboardPayload") -> None:
        self.arb_tab.render(payload)

    def _render_grok_section(self) -> None:
        from src.grok_analysis import grok_available
        from src.ollama_client import check_ollama_health

        available = grok_available()
        result = self._grok_result
        # If we have a prior HA slip result, still render it while offline
        if result is None and not available:
            try:
                health = check_ollama_health()
                result = {
                    "ok": False,
                    "error": health.get("error"),
                    "error_class": health.get("error_class") or "offline",
                    "health_banner": health.get("banner")
                    or "Ollama offline — showing model tickets only",
                    "bet_slip": [],
                    "picks": [],
                    "summary": "",
                }
            except Exception:
                pass
        self.grok_panel.render(result, available=available)

    def _on_grok_analysis(self) -> None:
        if self._grok_busy:
            self._set_status("Ollama analysis already running...")
            return
        if self._payload is None or not self._payload.books:
            self._set_status("Run Refresh Next Two first - need card data for Ollama.")
            return
        _button_debug(
            "Run Ollama Analysis -> _run_grok_analysis_async() -> analyze_card_with_ollama()"
        )
        self._run_grok_analysis_async()

    def _run_grok_analysis_async(self) -> None:
        from src.grok_analysis import analyze_card_with_ollama

        self._grok_busy = True
        self.grok_panel.set_busy(True, "Ollama is thinking...")
        if hasattr(self, "grok_btn"):
            self.grok_btn.configure(state="disabled")
        model = self.grok_panel.sync_active_model()
        timeout_s = int(getattr(config, "OLLAMA_TIMEOUT_SEC", 60) or 60)
        self._set_status(f"Ollama is thinking... ({model}, timeout {timeout_s}s)")
        payload = self._payload
        budget = self._current_budget_state()
        if hasattr(self, "budget_bar") and self.budget_bar is not None:
            try:
                budget = {
                    **budget,
                    **self.budget_bar.get_state(),
                    "card_budget_overridden": bool(
                        getattr(self.budget_bar, "_card_budget_overridden", False)
                    ),
                }
            except Exception:
                pass
        event_label = payload.event_label if payload else ""
        profile = self._profile_from_menu(self.profile_var.get())
        allowed = self._allowed_fight_keys(payload.combined) if payload else set()

        def worker() -> None:
            try:
                _button_debug(
                    f"analyze_card_with_ollama(event_label={event_label!r}) "
                    f"profile={profile!r} "
                    f"model={model!r} "
                    f"timeout={getattr(config, 'OLLAMA_TIMEOUT_SEC', 60)}"
                )
                result = analyze_card_with_ollama(
                    payload.books if payload else {},
                    budget,
                    event_label=event_label,
                    profile=profile,
                    allowed_fights=allowed or None,
                )
                self.after(0, lambda: self._apply_grok_result(result))
            except Exception as exc:
                # Last-resort: still try to surface HA tickets instead of empty fail.
                err = _format_grok_user_error(exc)
                fallback: dict[str, Any] = {
                    "ok": True,
                    "warning": err,
                    "error_class": "ok",
                    "health_banner": "Ollama narrative skipped — showing HA tickets",
                    "narrative_degraded": True,
                    "summary": "HA bet slip ready — Ollama narrative skipped.",
                    "picks": [],
                    "bet_slip": [],
                    "source": "ha_slip",
                }
                try:
                    from src.grok_analysis import collect_card_analysis_inputs, merge_ollama_reasons_into_slip

                    inputs = collect_card_analysis_inputs(
                        payload.books if payload else {},
                        budget,
                        event_label=event_label,
                        profile=profile,
                        allowed_fights=allowed or None,
                    )
                    tickets = list(inputs.get("tickets") or [])
                    fallback["bet_slip"] = merge_ollama_reasons_into_slip(tickets, [])
                    fallback["fun_tiers"] = inputs.get("fun_tiers") or {}
                    fallback["n_actionable"] = inputs.get("n_actionable")
                    fallback["n_advisory"] = inputs.get("n_advisory")
                    fallback["total_stake_pct"] = inputs.get("total_stake_pct")
                    fallback["total_stake_usd"] = inputs.get("total_stake_usd")
                    fallback["card_budget"] = inputs.get("card_budget")
                    fallback["bankroll"] = inputs.get("bankroll")
                    fallback["event"] = event_label
                    n = len(fallback["bet_slip"])
                    fallback["summary"] = (
                        f"Top {n} HA tickets (Ollama skipped: {err[:80]})"
                        if n
                        else "NO BET — nothing cleared HA gates for this card."
                    )
                    fallback["no_bet"] = n == 0 or bool(inputs.get("no_bet"))
                except Exception:
                    fallback["ok"] = False
                    fallback["error"] = err
                    fallback["error_class"] = "other"
                self.after(0, lambda fb=fallback: self._apply_grok_result(fb))
            finally:
                self.after(0, self._finish_grok_busy)

        threading.Thread(target=worker, daemon=True).start()

    def _apply_grok_result(self, result: dict[str, Any]) -> None:
        self._grok_result = result
        self._render_grok_section()
        err_class = str(result.get("error_class") or "")
        latency = result.get("latency_ms")
        banner = str(result.get("health_banner") or "").strip()
        if result.get("ok") or result.get("bet_slip") or result.get("no_bet"):
            n = len(result.get("bet_slip") or result.get("picks") or [])
            cache = " (cached)" if result.get("from_cache") else ""
            model = result.get("model") or getattr(config, "OLLAMA_MODEL", "ollama")
            latency_note = f" · {int(latency)}ms" if latency is not None else ""
            class_note = f" · {err_class}" if err_class and err_class != "ok" else ""
            if result.get("no_bet") or result.get("no_usable_odds"):
                msg = str(result.get("summary") or "NO BET")
                self._set_status(f"Ollama: {msg} ({model}){cache}{latency_note}{class_note}")
            elif banner and not result.get("ok"):
                self._set_status(
                    f"{banner} — {n} ticket(s) ({model}){cache}{latency_note}{class_note}"
                )
            else:
                total_pct = result.get("total_stake_pct")
                total_usd = result.get("total_stake_usd")
                sizing = ""
                if total_pct is not None and total_usd is not None:
                    sizing = f" · {float(total_pct):.1f}% / ${float(total_usd):.2f}"
                self._set_status(
                    f"Ollama bet slip — {n} ticket(s){sizing} ({model})"
                    f"{cache}{latency_note}{class_note}"
                )
            if self._payload is not None:
                self._render_overview_section(self._payload)
            try:
                from src.grok_analysis import build_best_bets_briefing

                preds = None
                cleared: list[dict[str, Any]] = []
                if self._payload is not None:
                    preds = self._payload.combined
                    for book_data in (self._payload.books or {}).values():
                        if isinstance(book_data, dict):
                            cleared.extend(
                                (book_data.get("alerts") or {}).get("singles") or []
                            )
                briefing = build_best_bets_briefing(
                    result, predictions=preds, cleared_singles=cleared
                )
                self.grok_panel.append_chat("Stats", briefing)
            except Exception as exc:
                _debug_log(f"Auto best-bets briefing failed: {exc}")
        else:
            err = _format_grok_user_error(result.get("error"))
            prefix = banner or "Ollama failed"
            self._set_status(f"{prefix}: {err[:160]}")

    def _finish_grok_busy(self) -> None:
        self._grok_busy = False
        self.grok_panel.set_busy(False)
        try:
            if hasattr(self, "grok_btn"):
                self.grok_btn.configure(state="normal")
        except Exception:
            pass

    def _on_ollama_chat(self, question: str) -> None:
        """Answer Ask / Best bets from the Ollama tab communication bar."""
        q = str(question or "").strip()
        if not q:
            return
        if self._grok_busy:
            self.grok_panel.append_chat(
                "System", "Wait for the current Ollama analysis to finish, then ask again."
            )
            return

        result = self._grok_result or getattr(self.grok_panel, "_last_result", None)
        event_label = ""
        if self._payload is not None:
            event_label = str(getattr(self._payload, "event_label", "") or "")

        # Attach live preds so best-bet asks can rank GREEN/YELLOW fun picks.
        if isinstance(result, dict) and self._payload is not None:
            result = dict(result)
            if self._payload.combined is not None:
                result["predictions"] = self._payload.combined
            if not result.get("fun_tiers"):
                try:
                    from src.bet_tiers import (
                        collect_props_from_books,
                        merge_bet_tier_dicts,
                        rank_card_bet_tiers,
                        rank_prop_bet_tiers,
                    )

                    cleared: list[dict[str, Any]] = []
                    for book_data in (self._payload.books or {}).values():
                        if isinstance(book_data, dict):
                            cleared.extend(
                                (book_data.get("alerts") or {}).get("singles") or []
                            )
                    ml_tiers = rank_card_bet_tiers(
                        self._payload.combined,
                        cleared_singles=cleared,
                        limit_per_tier=6,
                    )
                    prop_rows = collect_props_from_books(
                        self._payload.books or {}, limit=6
                    )
                    prop_tiers = rank_prop_bet_tiers(prop_rows, limit_per_tier=6)
                    result["fun_tiers"] = merge_bet_tier_dicts(
                        ml_tiers, prop_tiers, limit_per_tier=8
                    )
                    result["prop_tiers"] = prop_tiers
                except Exception:
                    pass

        model = self.grok_panel.sync_active_model()
        self.grok_panel._chat_ask_busy = True
        self.grok_panel.set_chat_busy(True)
        try:
            self.grok_panel.status_label.configure(text=f"Answering... ({model})")
        except Exception:
            pass

        def worker() -> None:
            try:
                from src.grok_analysis import answer_ollama_chat

                out = answer_ollama_chat(
                    q,
                    analysis_result=result if isinstance(result, dict) else None,
                    event_label=event_label,
                )
                answer = str(out.get("answer") or out.get("error") or "No answer.")
                source = str(out.get("source") or "")
                role = "Stats" if source == "ha_briefing" else "Ollama"
                self.after(0, lambda a=answer, r=role: self._finish_ollama_chat(r, a))
            except Exception as exc:
                msg = f"Chat failed: {exc}"
                self.after(0, lambda m=msg: self._finish_ollama_chat("System", m))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_ollama_chat(self, role: str, answer: str) -> None:
        self.grok_panel._chat_ask_busy = False
        if not self._grok_busy:
            self.grok_panel.set_chat_busy(False)
        self.grok_panel.append_chat(role, answer)
        try:
            self.grok_panel.status_label.configure(text="Ready")
        except Exception:
            pass

    def _schedule_render_all_tabs(self, payload: DashboardPayload) -> None:
        """Render overview + books immediately; defer heavy tabs until visited."""
        self._render_token += 1
        token = self._render_token
        self._render_payload = payload
        self._rendered_sections.clear()
        _debug_log("Rendering overview + books (lazy tabs deferred)")
        props_enabled = _ensure_props_config()
        _debug_log(f"Rendering Props tabs: {'enabled' if props_enabled else 'disabled'}")

        try:
            self._render_overview_section(payload)
            self._rendered_sections.add("overview")
        except Exception as exc:
            _debug_log(f"Overview render error: {traceback.format_exc()}")
            self._set_status(f"Overview render warning: {exc}")

        try:
            self._render_books_section(payload)
            self._rendered_sections.add("books")
        except Exception as exc:
            _debug_log(f"Books render error: {traceback.format_exc()}")
            self._set_status(f"Book tabs render warning: {exc}")

        if props_enabled:
            try:
                self._render_props_section(payload)
                self._rendered_sections.add("props")
            except Exception as exc:
                _debug_log(f"Props render error: {traceback.format_exc()}")

        current = self.tabs.get() if hasattr(self, "tabs") else "Overview"
        self.after(1, lambda: self._render_tab_lazy(current))
        self.after(800, lambda t=token: self._render_idle_deferred(t))

    def _render_idle_deferred(self, token: int) -> None:
        """Warm props/risk in background if user has not opened them yet."""
        if token != self._render_token or self._payload is None:
            return
        if "props" not in self._rendered_sections:
            self._render_props_section(self._payload)
            self._rendered_sections.add("props")
        if "risk" not in self._rendered_sections:
            self._render_risk_section(self._payload)
            self._rendered_sections.add("risk")
        if "arb" not in self._rendered_sections:
            self._render_arb_section(self._payload)
            self._rendered_sections.add("arb")

    def _render_overview_section(self, payload: DashboardPayload) -> None:
        overview = payload.books.get("Overview", {})
        alerts = overview.get("alerts") or {}
        bankroll = float(
            self._budget_state.get("total_bankroll") or alerts.get("bankroll") or config.INITIAL_BANKROLL
        )
        strategy = strategy_from_profile(bankroll=bankroll)

        # Overview
        singles_n = int(alerts.get("singles_count") or len(alerts.get("singles") or []))
        parlays_n = int(alerts.get("parlays_count") or len(alerts.get("parlays") or []))
        self.overview_summary.configure(
            text=(
                f"{payload.event_label}  |  "
                f"{singles_n} qualifying singles  |  {parlays_n} parlays  |  "
                f"Profile: {config.normalize_profile(payload.profile)}"
            )
        )
        preds = overview.get("predictions", pd.DataFrame())
        if _df_is_empty(preds) and not _df_is_empty(payload.combined):
            preds = payload.combined
        preds = _dedupe_fight_rows(_as_dataframe(preds))
        _render_grouped_fight_tables(
            self.overview_fights_scroll,
            payload.cards or [],
            preds,
            bankroll=bankroll,
            strategy=strategy,
            compact=True,
            table_height=8,
            payload=payload,
        )
        n_fights = _df_row_count(preds)
        _debug_log(f"Overview fight table: {n_fights} rows across {len(payload.cards or [])} card(s)")

        bs = self._budget_state
        from src.strategy import collect_dashboard_risk_warnings, format_risk_warnings

        recs = self._overview_recommendations()
        self.top_bets_panel.render(recs)
        if not recs and n_fights > 0:
            self.top_bets_panel.empty_label.configure(
                text=(
                    f"No BLUE/GREEN edges on loaded card ({n_fights} fights). "
                    "Red = don't bet · try Soft Update if odds look missing."
                )
            )
            self.top_bets_panel.empty_label.pack(fill="x", padx=16, pady=(0, 14))

        overview_risks = collect_dashboard_risk_warnings(alerts, bs, bankroll=bankroll)
        risk_txt, risk_color = format_risk_warnings(overview_risks, max_lines=3)
        if risk_txt:
            self.overview_risk_box.configure(text=risk_txt, text_color=risk_color)
            self.overview_risk_box.pack(fill="x", padx=12, pady=(0, 4), after=self.overview_summary)
        else:
            self.overview_risk_box.pack_forget()

        try:
            from src.gane_foul_scenario import build_gane_foul_scenario

            foul_scenario = build_gane_foul_scenario(
                overview_predictions=preds,
                books=payload.books,
            )
            if foul_scenario.get("found"):
                self.gane_foul_panel.pack(fill="x", padx=8, pady=(0, 8), after=self.top_bets_panel)
                self.gane_foul_panel.render(foul_scenario)
            else:
                self.gane_foul_panel.pack_forget()
        except Exception as exc:
            _debug_log(f"Gane foul scenario render failed: {exc}")
            self.gane_foul_panel.pack_forget()

    def _book_tab_data(self, payload: DashboardPayload, book_key: str) -> dict[str, Any]:
        """Book payload for a tab - never substitute Overview/consensus odds on book tabs."""
        data = dict(payload.books.get(book_key) or {})
        preds = data.get("predictions", pd.DataFrame())
        if isinstance(preds, pd.DataFrame) and not preds.empty:
            if "odds_matched" in preds.columns:
                data["odds_matched"] = int(preds["odds_matched"].sum())
            data["odds_total"] = data.get("odds_total") or len(preds)
            return data

        disabled_hint = ""
        if book_key == "BetNow.eu" and not getattr(config, "BETNOW_ENABLED", False):
            disabled_hint = (
                "BetNow disabled — set BETNOW_ENABLED=true and BETNOW_SESSION "
                "(or a real BETNOW_COOKIE) in .env"
            )
        elif book_key == "DraftKings" and not getattr(config, "DRAFTKINGS_ENABLED", False):
            disabled_hint = (
                "DraftKings disabled — set DRAFTKINGS_ENABLED=true "
                "(uses THE_ODDS_API_KEY)"
            )
        elif book_key == "MyBookie" and not getattr(config, "MYBOOKIE_ENABLED", False):
            disabled_hint = "MyBookie disabled — set MYBOOKIE_ENABLED=true in .env"

        model_only = _model_fights_from_payload(payload)
        if not model_only.empty:
            data = {
                **data,
                "predictions": model_only,
                "odds_total": len(model_only),
                "odds_matched": 0,
                "warning": data.get("warning")
                or data.get("error")
                or disabled_hint
                or f"No {book_key} odds loaded - click Quick Odds + Props or Refresh Props.",
            }
        elif disabled_hint and not data.get("warning"):
            data["warning"] = disabled_hint
            data["odds_matched"] = 0
            data["odds_total"] = 0
        return data

    def _render_books_section(self, payload: DashboardPayload) -> None:
        ctx = payload.threshold_ctx or {}
        bs = self._budget_state
        profile = self._profile_from_menu(self.profile_var.get())
        for tab, key in (
            (getattr(self, "odds_api_tab", None), "Odds API"),
            (getattr(self, "betnow_tab", None), "BetNow.eu"),
            (getattr(self, "dk_tab", None), "DraftKings"),
            (getattr(self, "mybookie_tab", None), "MyBookie"),
        ):
            if tab is None:
                continue
            data = self._book_tab_data(payload, key)
            data["cards"] = payload.cards
            data["_payload"] = payload
            tab.render(data, ctx, budget_state=bs, profile=profile)

    def _render_next_two_section(self, payload: DashboardPayload) -> None:
        ctx = payload.threshold_ctx or {}
        bankroll = float(
            self._budget_state.get("total_bankroll") or config.INITIAL_BANKROLL
        )
        strategy = strategy_from_profile(bankroll=bankroll)

        for w in self.next_two_scroll.winfo_children():
            w.destroy()
        if not payload.cards and _df_is_empty(payload.combined):
            ctk.CTkLabel(
                self.next_two_scroll,
                text="No upcoming cards loaded - click Refresh Next Two.",
                anchor="w",
            ).pack(fill="x", padx=8, pady=8)
            return
        odds_api = payload.books.get("Odds API", {}).get("predictions", pd.DataFrame())
        dk = payload.books.get("DraftKings", {}).get("predictions", pd.DataFrame())
        # Prefer combined model card so both upcoming events populate when present;
        # book frames still supply odds via _preds_for_card merge.
        combined = _as_dataframe(payload.combined)
        if not combined.empty:
            book_preds = combined
        elif isinstance(odds_api, pd.DataFrame) and not odds_api.empty:
            book_preds = odds_api
        elif isinstance(dk, pd.DataFrame) and not dk.empty:
            book_preds = dk
        else:
            book_preds = combined
        # Overlay Odds API odds onto combined rows when available (display only).
        if (
            isinstance(odds_api, pd.DataFrame)
            and not odds_api.empty
            and not combined.empty
            and book_preds is combined
        ):
            try:
                book_preds = _merge_fights_with_book_odds(combined, odds_api)
            except Exception:
                book_preds = combined
        display_cards = _display_cards(payload, book_preds)
        if len(display_cards) < 2:
            _debug_log(f"Next Two tab: only {len(display_cards)} card group(s) - run Refresh Next Two")
        _render_grouped_fight_tables(
            self.next_two_scroll,
            payload.cards,
            book_preds,
            bankroll=bankroll,
            strategy=strategy,
            compact=False,
            table_height=8,
            payload=payload,
        )
        for card in display_cards:
            ev = card.get("event_name", "Card")
            cp = card.get("predictions", pd.DataFrame())
            sub_df = _preds_for_card(book_preds, cp, ev)
            try:
                from src.parlay_builder import ranked_parlays_for_card

                card_parlays = ranked_parlays_for_card(
                    sub_df,
                    bankroll=bankroll,
                    use_dynamic=ctx.get("use_dynamic"),
                    event_name=ev,
                )
                if card_parlays:
                    parlay_box = ctk.CTkFrame(self.next_two_scroll, fg_color="transparent")
                    parlay_box.pack(fill="x", padx=4, pady=(0, 8))
                    _render_ranked_parlays(
                        parlay_box,
                        card_parlays,
                        title=f"Same-card parlays - {_format_card_header(ev)}",
                        preds=sub_df,
                    )
            except Exception as exc:
                _debug_log(f"Parlay render skipped for {ev}: {exc}")

    def _render_props_section(self, payload: DashboardPayload) -> None:
        props_enabled = _ensure_props_config()
        _debug_log(f"Rendering Props tabs: {'enabled' if props_enabled else 'disabled'}")
        if not props_enabled:
            return
        bs = self._budget_state
        profile = self._profile_from_menu(self.profile_var.get())
        for book_key, tab in (
            ("Odds API", getattr(self, "props_odds_api_tab", None)),
            ("BetNow.eu", getattr(self, "props_betnow_tab", None)),
            ("DraftKings", getattr(self, "props_dk_tab", None)),
            ("MyBookie", getattr(self, "props_mybookie_tab", None)),
        ):
            if tab is None:
                continue
            props = payload.books.get(book_key, {}).get("props") or {}
            n = len(props.get("singles") or [])
            _button_debug(f"Loading props for {book_key} ({n} singles)")
            tab.render(payload, budget_state=bs, profile=profile)

    def _render_risk_section(self, payload: DashboardPayload) -> None:
        self._render_risk_tab(payload, budget_state=self._budget_state)

    def _render_all_tabs(self, payload: DashboardPayload) -> None:
        """Re-render all tabs (profile/budget changes) without blocking one long paint."""
        self._schedule_render_all_tabs(payload)

    def _render_model_insights_panel(self) -> None:
        """Show top discovered interaction features from the latest train run."""
        lines: list[str] = []
        try:
            import config

            paths = [
                config.DISCOVERED_INTERACTIONS_PATH,
                config.FEATURE_IMPORTANCE_PATH,
            ]
            discovered: dict[str, Any] = {}
            for path in paths:
                if path.is_file():
                    discovered = json.loads(path.read_text(encoding="utf-8"))
                    if discovered.get("top_interaction_importance") or discovered.get("insights"):
                        break

            top_ix = discovered.get("top_interaction_importance") or []
            if not top_ix and discovered.get("importance"):
                imp = discovered.get("importance") or {}
                top_ix = [
                    {"feature": k, "importance": v, "label": k}
                    for k, v in sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
                    if str(k).startswith("ix_")
                ][:10]

            insights = discovered.get("insights") or discovered.get("interaction_insights") or []
            if top_ix:
                lines.append("Top interaction features (model importance):")
                for i, row in enumerate(top_ix[:10], start=1):
                    feat = row.get("feature", "")
                    label = row.get("label", feat)
                    imp = float(row.get("importance", 0.0)) * 100
                    lines.append(f"  {i:2}. {label} ({feat}) - {imp:.1f}% importance")
            if insights:
                lines.append("")
                lines.append("Discovered patterns (train split):")
                for row in insights[:8]:
                    lines.append(f"  * {row.get('message', row.get('label', ''))}")
            if not lines:
                lines.append(
                    "No interaction discoveries yet. Run: python main.py --train --backtest-2025"
                )
        except Exception as exc:
            lines = [f"Model insights unavailable: {exc}"]
        self.model_insights_box.configure(text="\n".join(lines))

    def _render_risk_tab(self, payload: DashboardPayload, *, budget_state: dict[str, Any] | None = None) -> None:
        try:
            from src.strategy import (
                available_card_budget_text,
                collect_dashboard_risk_warnings,
                format_risk_warnings,
            )

            rm = payload.risk_metrics or {}
            ctx = payload.threshold_ctx or {}
            overview = payload.books.get("Overview", {})
            alerts = overview.get("alerts") or {}
            bankroll = float(
                (budget_state or {}).get("total_bankroll")
                or alerts.get("bankroll")
                or config.INITIAL_BANKROLL
            )
            cp = rm.get("card_pnl") or {}
            lines = ["Monte Carlo card risk:"]
            if rm.get("available"):
                lines.append(
                    f"Mean PnL ${cp.get('mean_pnl', 0):+,.0f}  |  "
                    f"P(loss) {cp.get('prob_loss', 0):.0%}  |  "
                    f"P5 ${cp.get('p5_pnl', 0):+,.0f}  |  P95 ${cp.get('p95_pnl', 0):+,.0f}"
                )
                lines.append(
                    f"Suggested card cap {rm.get('suggested_max_risk_pct', 0):.1f}%  |  "
                    f"{rm.get('n_bets', 0)} value bets"
                )
            else:
                lines.append(rm.get("reason", "Run Refresh with odds-matched fights."))
            if budget_state:
                lines.append(available_card_budget_text(budget_state))
            if ctx.get("thresholds"):
                t = ctx["thresholds"]
                lines.append(
                    f"Active thresholds - edge {t.get('alert_min_edge', 0):.1%}, "
                    f"parlay leg {t.get('parlay_min_edge', 0):.1%}, "
                    f"combined {t.get('parlay_min_combined_prob', 0):.0%}"
                )
            try:
                from src.high_accuracy_strategy import format_strategy_rules_block

                lines.append("")
                lines.append(format_strategy_rules_block())
            except Exception:
                pass
            dl = alerts.get("decision_layer") if isinstance(alerts, dict) else None
            if isinstance(dl, dict) and dl.get("strategy_line"):
                lines.append(str(dl["strategy_line"]))
            if config.is_live_profile():
                cap = config.max_card_stake_cap(bankroll)
                live_cap = config.live_card_budget_cap_usd(bankroll)
                lines.append(
                    f"LIVE GUARDRAILS - Max ${cap:,.0f} total stake this card "
                    f"(${bankroll:,.0f} bankroll, ${live_cap:,.0f} hard cap). "
                    "Fewer bets; higher edge required."
                )
            elif config.is_paper_profile():
                lines.append(
                    "PAPER - High-accuracy / low-volume simulation (same rules, slightly looser floors)."
                )
            risk_warnings = collect_dashboard_risk_warnings(alerts, budget_state, bankroll=bankroll)
            warn_txt, _ = format_risk_warnings(risk_warnings, max_lines=5, separator="\nWARNING: ")
            if warn_txt:
                lines.append(warn_txt)
            self.risk_summary.configure(text="\n".join(lines))
            # Skip scorecard panel (noise vs edge left on table)
            try:
                for child in self.skip_scorecard_frame.winfo_children():
                    child.destroy()
                sc = alerts.get("skip_scorecard") if isinstance(alerts, dict) else None
                _render_skip_scorecard_panel(self.skip_scorecard_frame, sc)
            except Exception as exc:
                _debug_log(f"Skip scorecard panel error: {exc}")
            try:
                if hasattr(self, "sleeve_stats_frame"):
                    for child in self.sleeve_stats_frame.winfo_children():
                        child.destroy()
                    _render_sleeve_stats_panel(self.sleeve_stats_frame)
            except Exception as exc:
                _debug_log(f"Sleeve stats panel error: {exc}")
            self._render_model_insights_panel()

            self._risk_fig.clear()
            ax = self._risk_fig.add_subplot(111)
            ax.set_facecolor("#1a1a1a")
            staking = rm.get("staking_modes") or {}
            if staking:
                names = list(staking.keys())
                dds = [staking[m].get("expected_max_drawdown_pct", 0) for m in names]
                ax.bar(names, dds, color="#6366f1")
                ax.set_title("Expected Max Drawdown by Staking Mode", color="white")
                ax.set_ylabel("Drawdown %", color="#ccc")
            else:
                ax.text(0.5, 0.5, "Run Refresh for MC data", ha="center", color="#888", transform=ax.transAxes)
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values():
                spine.set_color("#444")
            self._risk_canvas.draw()

            self._thresh_fig.clear()
            ax2 = self._thresh_fig.add_subplot(111)
            ax2.set_facecolor("#1a1a1a")
            examples = example_threshold_table(profile=payload.profile)
            ax2.plot(examples["bankroll"], examples["min_edge"] * 100, marker="o", label="Min edge %", color="#3dd68c")
            ax2.plot(
                examples["bankroll"],
                examples["parlay_leg_edge"] * 100,
                marker="s",
                label="Parlay leg %",
                color="#60a5fa",
            )
            ax2.set_xscale("log")
            ax2.set_title("Dynamic Thresholds vs Bankroll", color="white")
            ax2.set_xlabel("Bankroll ($)", color="#ccc")
            ax2.set_ylabel("Threshold (%)", color="#ccc")
            ax2.legend(facecolor="#2b2b2b", labelcolor="white", fontsize=8)
            ax2.tick_params(colors="#aaa")
            for spine in ax2.spines.values():
                spine.set_color("#444")
            self._thresh_canvas.draw()
        except Exception as exc:
            _debug_log(f"Risk tab error: {exc}")
            self.risk_summary.configure(text=f"Risk tab error: {exc}")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFC Predictor Dashboard")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print startup diagnostics to the console",
    )
    # PyInstaller passes through unknown args; ignore extras when frozen
    args, _unknown = parser.parse_known_args()
    return args


def main(argv: list[str] | None = None) -> int:
    global _DEBUG_MODE, _HEARTBEAT_LOGGER_READY

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]

    if "--debug" in sys.argv:
        _DEBUG_MODE = True
        _enable_debug_console()

    _parse_cli_args()
    _dashboard_heartbeat("dashboard bootstrap start")

    if _DEBUG_MODE:
        _debug_log(
            "Python started. The ~280 MB onefile EXE unpacks before this line - "
            "first launch can take up to ~1 minute with no further console output."
        )

    if _STARTUP_ERROR or ctk is None:
        _write_crash_log("startup_aborted", _STARTUP_ERROR or "ctk is None")
        return 1

    splash = SplashScreen()
    splash.set_status("Starting UFC Predictor Dashboard...")
    splash.pump()

    try:
        _load_dependencies(progress=splash.set_status)
        splash.pump()
    except Exception as exc:
        splash.close()
        tb = traceback.format_exc()
        _write_crash_log("dependency_load_failed", f"{exc}\n\n{tb}")
        try:
            print(tb, file=sys.stderr, flush=True)
        except Exception:
            pass
        _show_fatal_error("UFC Dashboard - startup error", f"{exc}\n\n{tb}")
        return 1

    try:
        if config is not None:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            from src.logging_utils import setup_logging

            setup_logging(
                verbose=_DEBUG_MODE,
                log_dir=config.LOG_DIR,
                log_name="dashboard.log",
                console=_DEBUG_MODE,
            )
            _HEARTBEAT_LOGGER_READY = True
            import logging as _logging

            _logging.getLogger("matplotlib").setLevel(_logging.WARNING)
            _logging.getLogger("matplotlib.font_manager").setLevel(_logging.WARNING)
            _dashboard_heartbeat("file logging ready (dashboard.log)")

        splash.set_status("Building main window...")
        splash.pump()
        splash.close()

        app = UFCDashboardApp()
        _dashboard_heartbeat("dashboard UI built")
        _debug_log("Main window ready - entering event loop")
        _dashboard_heartbeat("dashboard mainloop starting")
        app.mainloop()
        _dashboard_heartbeat("dashboard mainloop exited cleanly")
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        _write_crash_log("runtime_error", f"{exc}\n\n{tb}")
        try:
            print(tb, file=sys.stderr, flush=True)
        except Exception:
            pass
        _debug_log(tb)
        try:
            splash.close()
        except Exception:
            pass
        _show_fatal_error("UFC Dashboard - runtime error", f"{exc}\n\n{tb}")
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        _write_crash_log("unhandled_top_level", f"{exc}\n\n{tb}")
        try:
            print(tb, file=sys.stderr, flush=True)
        except Exception:
            pass
        _show_fatal_error("UFC Dashboard - unhandled error", f"{exc}\n\n{tb}")
        raise SystemExit(1)