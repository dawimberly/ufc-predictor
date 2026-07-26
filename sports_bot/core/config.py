"""Central settings — env-backed, no stock-bot coupling."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]  # .../SportsBettingBot
load_dotenv(ROOT / ".env", override=False)

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
BANK_DIR = DATA_DIR / "bank"

PREDICTION_BANK_CSV = BANK_DIR / "prediction_bank.csv"
PREDICTION_LESSONS_JSON = BANK_DIR / "prediction_lessons.json"

PROFILE = os.getenv("SPORTS_BOT_PROFILE", "paper").strip().lower()
INITIAL_BANKROLL = float(os.getenv("INITIAL_BANKROLL", "100"))
CARD_BUDGET = float(os.getenv("CARD_BUDGET", "15"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.03"))
MAX_BET_FRACTION = float(os.getenv("MAX_BET_FRACTION", "0.02"))

ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY", "")
ENABLE_ODDS_SCRAPE = os.getenv("ENABLE_ODDS_SCRAPE", "true").lower() in ("1", "true", "yes")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() in ("1", "true", "yes")

OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() in ("1", "true", "yes")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")
OLLAMA_THINK_MODEL = os.getenv("OLLAMA_THINK_MODEL", "deepseek-r1:8b")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "600"))

PREDICTION_BANK_AUTO_LOG = os.getenv("PREDICTION_BANK_AUTO_LOG", "true").lower() in (
    "1",
    "true",
    "yes",
)


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, BANK_DIR):
        path.mkdir(parents=True, exist_ok=True)
