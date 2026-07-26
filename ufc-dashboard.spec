# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ufc-dashboard.exe — bundles XGBoost DLLs + customtkinter assets."""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

block_cipher = None

# XGBoost / LightGBM native libraries (critical on Windows onefile)
xgboost_datas = collect_data_files("xgboost")
xgboost_binaries = collect_dynamic_libs("xgboost")
lightgbm_binaries = collect_dynamic_libs("lightgbm")

ctk_datas, ctk_binaries, ctk_hidden = collect_all("customtkinter")
shap_datas, shap_binaries, shap_hidden = collect_all("shap")
mpl_datas, mpl_binaries, mpl_hidden = collect_all("matplotlib")

extra_datas = list(xgboost_datas) + list(ctk_datas) + list(shap_datas) + list(mpl_datas)
extra_datas += [("data", "data"), ("models", "models"), ("src", "src")]

extra_binaries = list(xgboost_binaries) + list(lightgbm_binaries) + list(ctk_binaries)
extra_binaries += list(shap_binaries) + list(mpl_binaries)

hiddenimports = [
    "main",
    "config",
    "xgboost",
    "xgboost.sklearn",
    "xgboost.core",
    "xgboost.compat",
    "lightgbm",
    "sklearn",
    "sklearn.utils._typedefs",
    "sklearn.utils._weight_vector",
    "sklearn.neighbors._partition_nodes",
    "customtkinter",
    "PIL",
    "PIL._tkinter_finder",
    "matplotlib",
    "matplotlib.backends.backend_tkagg",
    "bs4",
    "lxml",
    "dotenv",
    "joblib",
    "scipy",
    "scipy.special",
    "scipy.sparse",
    "ufc_betting_bot",
    "ufc_betting_bot.modules.edge",
    "ufc_betting_bot.modules.dynamic_thresholds",
    "ufc_betting_bot.modules.bankroll",
    "ufc_betting_bot.modules.odds",
    "src.grok_analysis",
    "src.ollama_client",
    "src.model_cache",
    "src.strategy",
    "src.dashboard_service",
    "src.arb_scanner",
    "src.background_runner",
    "src.props",
    "src.card_cache",
    "src.predictor",
    "src.gym_data",
    "src.bet_slip",
    "src.prediction_bank",
    "src.gane_foul_scenario",
    "src.safe_io",
    "src.project_paths",
    "src.data_loader",
    "src.alerts",
    "src.parlay_builder",
    "src.risk_manager",
    "src.explainability",
    "src.fight_brief",
    "src.backtester",
    "src.logging_utils",
    "src.odds_providers.betnow_scraper",
    "src.odds_providers.draftkings",
    "src.odds_providers.mybookie_scraper",
    "src.odds_providers.prop_odds_common",
    "src.odds_providers.cookie_capture",
    "src.odds_providers.odds_fallback",
    "requests",
]
hiddenimports += list(ctk_hidden) + list(shap_hidden) + list(mpl_hidden)

# build_dashboard.bat sets UFC_DASHBOARD_CONSOLE=1 for debug builds (console EXE).
# debug=False keeps PyInstaller bootloader quiet; app --debug still prints [dashboard] logs.
use_console = os.environ.get("UFC_DASHBOARD_CONSOLE", "").strip() in ("1", "true", "yes")

a = Analysis(
    ["src\\ufc_dashboard.py"],
    pathex=["."],
    binaries=extra_binaries,
    datas=extra_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["scripts\\rthook_ufc_dashboard.py"],
    excludes=["pytest", "tensorflow", "torch", "plotly"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ufc-dashboard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=use_console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
