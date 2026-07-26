"""Standalone project root — all runtime paths relative to C:\\UFC-Predictor."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path


def bundle_root() -> Path | None:
    """PyInstaller onefile extraction dir (_MEIPASS), if running frozen."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else None


def resolve_root(entry_file: Path | None = None) -> Path:
    """Return writable project root (EXE directory when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    if entry_file is not None:
        entry = entry_file.resolve()
        if entry.parent.name == "src":
            return entry.parents[1]
        return entry.parent
    return Path(__file__).resolve().parents[1]


def ensure_runtime_assets(root: Path) -> None:
    """
    Ensure data/ and models/ exist beside the EXE.

    PyInstaller --add-data bundles copies under _MEIPASS; copy to dist/ on first run.
    """
    bundle = bundle_root()
    if bundle is None:
        return

    for folder in ("data", "models"):
        src = bundle / folder
        dest = root / folder
        if not src.is_dir():
            continue
        if not dest.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
            continue
        # Fill in missing files without overwriting user cache
        for path in src.rglob("*"):
            if path.is_file():
                rel = path.relative_to(src)
                target = dest / rel
                if not target.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)


def setup_frozen_env(root: Path) -> None:
    """Matplotlib / DLL / GUI asset dirs for frozen onefile EXE."""
    cache = root / "data" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    mpl_dir = cache / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    (root / "data" / "logs").mkdir(parents=True, exist_ok=True)

    if not getattr(sys, "frozen", False):
        return

    bundle = bundle_root()
    if bundle is None:
        return

    # XGBoost / LightGBM native DLLs live under _MEIPASS after collect_dynamic_libs
    dll_dirs: list[Path] = []
    for rel in (
        "xgboost/lib",
        "xgboost",
        "lightgbm/bin",
        "lightgbm",
    ):
        candidate = bundle / rel.replace("/", os.sep)
        if candidate.is_dir():
            dll_dirs.append(candidate)

    path_parts = [str(d) for d in dll_dirs]
    if path_parts:
        os.environ["PATH"] = os.pathsep.join(path_parts + [os.environ.get("PATH", "")])
        if hasattr(os, "add_dll_directory"):
            for dll_dir in dll_dirs:
                try:
                    os.add_dll_directory(str(dll_dir))
                except OSError:
                    pass

    # customtkinter assets (themes, fonts) from collect-all
    for rel in ("customtkinter", "customtkinter/assets"):
        asset_dir = bundle / rel.replace("/", os.sep)
        if asset_dir.is_dir():
            os.environ.setdefault("CUSTOMTKINTER_ASSETS", str(asset_dir))
            break


def setup_sys_path(root: Path) -> None:
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    bundle = bundle_root()
    if bundle is not None:
        bundle_s = str(bundle)
        if bundle_s not in sys.path:
            sys.path.insert(0, bundle_s)


def patch_config(root: Path) -> None:
    """Point config.* at data/models/cache/logs under project root."""
    import config

    config.ROOT_DIR = root
    config.DATA_DIR = root / "data"
    config.RAW_DIR = config.DATA_DIR / "raw"
    config.PROCESSED_DIR = config.DATA_DIR / "processed"
    config.CACHE_DIR = config.DATA_DIR / "cache"
    config.MODELS_DIR = root / "models"
    config.RAW_FIGHTS_CSV = config.RAW_DIR / "fights.csv"
    config.PROCESSED_FEATURES_CSV = config.PROCESSED_DIR / "fight_features.csv"
    config.DEFAULT_MODEL_PATH = config.MODELS_DIR / "ensemble_winner.joblib"
    config.LEGACY_MODEL_PATH = config.MODELS_DIR / "lgbm_winner.joblib"
    config.METRICS_PATH = config.MODELS_DIR / "training_metrics.json"
    config.FEATURE_IMPORTANCE_PATH = config.MODELS_DIR / "feature_importance.json"
    config.BACKTEST_DIR = config.MODELS_DIR / "backtest"
    config.BACKTEST_SUMMARY_CSV = config.BACKTEST_DIR / "backtest_summary.csv"
    config.BACKTEST_PREDICTIONS_CSV = config.BACKTEST_DIR / "walk_forward_predictions.csv"
    config.BACKTEST_THRESHOLD_CSV = config.BACKTEST_DIR / "threshold_roi.csv"
    config.BACKTEST_IMPORTANCE_CSV = config.BACKTEST_DIR / "importance_timeline.csv"
    config.BACKTEST_METRICS_BY_YEAR_CSV = config.BACKTEST_DIR / "metrics_by_year.csv"
    config.BACKTEST_CALIBRATION_PNG = config.BACKTEST_DIR / "calibration_plot.png"
    config.BACKTEST_ROI_PNG = config.BACKTEST_DIR / "roi_threshold_plot.png"
    config.PLOTS_DIR = config.DATA_DIR / "plots"
    config.BACKTEST_2025_CSV = config.DATA_DIR / "backtest_2025_results.csv"
    config.GYMS_CSV = config.DATA_DIR / "gyms.csv"
    config.HISTORICAL_META_PATH = config.CACHE_DIR / "historical_meta.json"
    config.UPCOMING_CARD_CACHE = config.CACHE_DIR / "upcoming_card.csv"
    config.HISTORICAL_ODDS_CACHE = config.CACHE_DIR / "historical_odds_unified.csv"
    config.ODDS_API_CACHE_PATH = config.CACHE_DIR / "ufc_odds_api.csv"
    config.ODDS_CACHE_PATH = config.CACHE_DIR / "ufc_odds_api.csv"
    config.UFCSTATS_GRECO_CACHE_DIR = config.CACHE_DIR / "ufcstats_greco"
    config.UFCSTATS_ENRICH_META_PATH = config.CACHE_DIR / "ufcstats_enrich_meta.json"
    config.ALERT_STATE_PATH = config.CACHE_DIR / "alert_state.json"
    config.LOG_DIR = config.DATA_DIR / "logs"
    config.BET_JOURNAL_CSV = config.DATA_DIR / "bet_journal.csv"
    config.PREDICTION_BANK_CSV = config.DATA_DIR / "prediction_bank.csv"
    config.PREDICTION_LESSONS_JSON = config.DATA_DIR / "prediction_lessons.json"
    config.SKIP_SCORECARD_JSONL = config.LOG_DIR / "skip_scorecard.jsonl"
    config.SKIP_SCORECARD_JSON = config.DATA_DIR / "skip_scorecard.json"
    config.HEARTBEAT_PATH = config.CACHE_DIR / "heartbeat.json"
    config.CIRCUIT_BREAKER_STATE_PATH = config.CACHE_DIR / "circuit_breaker_state.json"
    config.DRAWDOWN_STATE_PATH = config.CACHE_DIR / "drawdown_state.json"
    config.RISK_EVENTS_LOG = config.LOG_DIR / "risk_events.log"
    config.BUDGET_JSON_PATH = config.DATA_DIR / "budget.json"
    config.BET_JOURNAL_CSV = config.DATA_DIR / "bet_journal.csv"
    config.HEARTBEAT_PATH = config.CACHE_DIR / "heartbeat.json"
    config.CIRCUIT_BREAKER_STATE_PATH = config.CACHE_DIR / "circuit_breaker_state.json"
    config.DRAWDOWN_STATE_PATH = config.CACHE_DIR / "drawdown_state.json"
    config.ALERT_STATE_PATH = config.CACHE_DIR / "alert_state.json"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.is_absolute() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


# Hard-coded Grok key sources (highest priority first). stock-bot always force-overrides.
GROK_ENV_LOAD_ORDER: list[tuple[Path, bool]] = [
    (Path(r"C:\Users\Owner\PythonTrading\stock-bot\.env"), True),
    (Path(r"C:\Users\Owner\PythonTrading\.env"), True),
    (Path(r"C:\UFC-Bot\.env"), False),
]

_GROK_KEY_PLACEHOLDERS = frozenset(
    {
        "",
        "your_key",
        "your_api_key",
        "changeme",
        "none",
        "null",
        "xai-...",
        "sk-...",
    }
)


def _normalize_grok_api_key(raw: str | None) -> str:
    key = str(raw or "").strip().strip('"').strip("'")
    if not key or key.lower() in _GROK_KEY_PLACEHOLDERS:
        return ""
    return key


def shared_trading_env_candidates(root: Path) -> list[Path]:
    """
    Trading-bot compatible .env locations (same pattern as stock-bot/config.py).

    Walks ancestors for repo-root .env and stock-bot/.env; optional SHARED_ENV_PATH.
    """
    candidates: list[Path] = []
    current = root.resolve()
    for _ in range(5):
        parent = current.parent
        if parent != current:
            candidates.append(parent / ".env")
            candidates.append(parent / "stock-bot" / ".env")
        if current == parent:
            break
        current = parent

    extra = os.getenv("SHARED_ENV_PATH", "").strip()
    if extra:
        candidates.append(Path(extra))

    return _dedupe_paths(candidates)


def env_file_load_order(root: Path) -> list[tuple[Path, bool]]:
    """
    (.env path, override) pairs in load order (lowest priority first).

    Shared trading-bot / repo-root files use override=False (fill missing keys only).
    UFC-local files use override=True so project settings win.
    """
    root = root.resolve()
    shared = [(path, False) for path in shared_trading_env_candidates(root)]

    local: list[tuple[Path, bool]] = [
        (root / "ufc_betting_bot" / ".env", True),
        (Path.cwd() / ".env", True),
        (root / ".env", True),
    ]
    if getattr(sys, "frozen", False):
        local.append((root.parent / ".env", True))

    seen: set[str] = set()
    ordered: list[tuple[Path, bool]] = []
    for path, override in shared + local:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((path, override))
    return ordered


def _grok_key_from_env_file(env_path: Path) -> str:
    from dotenv import dotenv_values

    vals = dotenv_values(env_path)
    return _normalize_grok_api_key(vals.get("GROK_API_KEY") or vals.get("XAI_API_KEY"))


def load_grok_api_key(
    *,
    log: Callable[[str], None] | None = None,
    force: bool = True,
) -> Path | None:
    """
    Load GROK_API_KEY / XAI_API_KEY from trading-bot .env paths (see GROK_ENV_LOAD_ORDER).

    When force=True (default), stock-bot / trading-bot keys always override ambient
    process env so an empty UFC .env cannot blank the key after startup.
    Logs source path + key length only (never the key itself).
    """
    from dotenv import load_dotenv

    import config

    # Always re-read trading-bot files first so dashboard startup picks up fresh keys.
    for env_path, override in GROK_ENV_LOAD_ORDER:
        if log:
            log(f"GROK .env check: {env_path} (override={override or force})")
        if not env_path.is_file():
            if log:
                log(f"GROK .env missing: {env_path}")
            continue
        load_dotenv(env_path, override=True if force else override)

    source: Path | None = None
    key = ""
    for env_path, _ in GROK_ENV_LOAD_ORDER:
        if not env_path.is_file():
            continue
        candidate = _grok_key_from_env_file(env_path)
        if not candidate:
            continue
        key = candidate
        source = env_path.resolve()
        break

    if not key:
        # Fall back to UFC-local .env after trading-bot paths.
        for env_path in (
            Path.cwd() / ".env",
            getattr(config, "ROOT_DIR", Path.cwd()) / ".env",
        ):
            if not env_path.is_file():
                continue
            if force:
                load_dotenv(env_path, override=False)
            candidate = _grok_key_from_env_file(env_path)
            if candidate:
                key = candidate
                source = env_path.resolve()
                break

    if not key:
        key = _normalize_grok_api_key(
            os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        )

    if key:
        os.environ["GROK_API_KEY"] = key
        os.environ["XAI_API_KEY"] = key
        config.GROK_API_KEY = key
        config.GROK_API_KEY_SOURCE = str(source) if source else "environment"
        if log:
            src_label = str(source) if source else "process environment"
            log(f"GROK_API_KEY loaded from {src_label} (len={len(key)})")
    else:
        config.GROK_API_KEY = ""
        config.GROK_API_KEY_SOURCE = ""
        if log:
            tried = ", ".join(str(p) for p, _ in GROK_ENV_LOAD_ORDER)
            log(f"GROK_API_KEY not found (checked: {tried})")

    return source


def resolve_grok_credentials(
    root: Path,
    loaded_files: list[Path] | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> Path | None:
    """Backward-compatible alias — Grok keys use load_grok_api_key hard paths only."""
    _ = root, loaded_files
    return load_grok_api_key(log=log, force=True)


def load_env_files(root: Path, *, log: Callable[[str], None] | None = None) -> list[Path]:
    """Load .env files; UFC-local files override shared trading-bot keys."""
    from dotenv import load_dotenv

    loaded: list[Path] = []
    for env_path, override in env_file_load_order(root):
        if log:
            log(f"Looking for .env at: {env_path} (override={override})")
        if not env_path.is_file():
            continue
        load_dotenv(env_path, override=override)
        loaded.append(env_path.resolve())
        if log:
            log(f"Loaded .env from: {env_path}")
    load_grok_api_key(log=log, force=True)
    return loaded


def reload_runtime_env(
    root: Path | None = None,
    *,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Re-load .env from disk and refresh config flags (call before odds/props refresh)."""
    root = root or resolve_root()
    load_env_files(root, log=log)
    import config

    config.refresh_runtime_env()
    # Force trading-bot Grok key again after UFC .env may have blanked ambient env.
    load_grok_api_key(log=log, force=True)
    config.refresh_runtime_env()
    return root


def bootstrap(*, entry_file: Path | None = None, env_log: Callable[[str], None] | None = None) -> Path:
    """
    Initialize cwd, sys.path, config paths, and .env for dev or frozen EXE.

    Layout:
        C:\\UFC-Predictor\\
        ├── src/
        ├── data/          (raw, processed, cache, logs)
        ├── models/
        ├── dist/          (ufc-predict.exe, ufc-dashboard.exe)
        └── ufc_betting_bot/
            └── .env
    """
    root = resolve_root(entry_file)
    if getattr(sys, "frozen", False):
        os.chdir(root)
    setup_sys_path(root)
    ensure_runtime_assets(root)
    setup_frozen_env(root)
    patch_config(root)
    load_env_files(root, log=env_log)

    import config

    # Force-reload Grok from trading-bot .env on every dashboard startup.
    load_grok_api_key(log=env_log, force=True)
    config.refresh_runtime_env()
    try:
        from src.odds_providers.odds_api_client import refresh_odds_api_runtime

        odds_meta = refresh_odds_api_runtime(root=root)
        if env_log:
            env_log(
                f"THE_ODDS_API_KEY source={odds_meta.get('key_source')} "
                f"len={odds_meta.get('key_length')} "
                f"last4={odds_meta.get('key_last4') or '-'}"
            )
    except Exception as exc:
        if env_log:
            env_log(f"Odds API key reload failed: {exc}")
    if env_log:
        env_log(f"ENABLE_PROPS loaded as: {config.ENABLE_PROPS}")
        env_log(f"Loaded MYBOOKIE_ENABLED = {config.MYBOOKIE_ENABLED}")
        _bn = (config.BETNOW_SESSION_TOKEN or "").strip()
        if _bn:
            env_log(
                f"BETNOW_SESSION loaded (len={len(_bn)}, "
                f"prefix={_bn[:12]}...)"
            )
        else:
            env_log("BETNOW_SESSION not set (cookie optional; public pages only)")
        _gk = (config.GROK_API_KEY or "").strip()
        if bool(getattr(config, "OLLAMA_ENABLED", True)):
            try:
                from src.ollama_client import ollama_available, resolve_model_chain

                if ollama_available():
                    chain = resolve_model_chain()
                    env_log(
                        f"OLLAMA ready host={getattr(config, 'OLLAMA_HOST', '')} "
                        f"model={chain[0] if chain else config.OLLAMA_MODEL}"
                    )
                else:
                    env_log(
                        f"OLLAMA enabled but not running at "
                        f"{getattr(config, 'OLLAMA_HOST', 'http://localhost:11434')}"
                    )
            except Exception as exc:
                env_log(f"OLLAMA status check failed: {exc}")
        if _gk and config.GROK_ENABLED:
            env_log(
                f"GROK_API_KEY present (len={len(_gk)}) but xAI is secondary; "
                "dashboard uses Ollama by default"
            )
        elif not bool(getattr(config, "OLLAMA_ENABLED", True)):
            env_log("OLLAMA_ENABLED=false - local analysis off")
    return root
