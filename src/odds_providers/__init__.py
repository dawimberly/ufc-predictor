"""Book-specific odds providers for the UFC dashboard."""

from src.odds_providers.action_network import fetch_action_network_odds
from src.odds_providers.betnow_scraper import fetch_betnow_odds, fetch_betnow_prop_odds
from src.odds_providers.draftkings import fetch_draftkings_odds
from src.odds_providers.draftkings_props import fetch_draftkings_prop_odds
from src.odds_providers.mybookie_scraper import fetch_mybookie_odds, fetch_mybookie_prop_odds
from src.odds_providers.the_odds_api import fetch_the_odds_api_odds, fetch_the_odds_api_prop_odds
from src.odds_providers.odds_fallback import (
    fetch_best_available_odds,
    fetch_book_scraper_odds,
    odds_frame_usable,
)
from src.odds_providers.cookie_capture import (
    capture_all_book_cookies,
    ensure_cookies_before_refresh,
)

__all__ = [
    "fetch_action_network_odds",
    "fetch_betnow_odds",
    "fetch_betnow_prop_odds",
    "fetch_draftkings_odds",
    "fetch_draftkings_prop_odds",
    "fetch_mybookie_odds",
    "fetch_mybookie_prop_odds",
    "fetch_the_odds_api_odds",
    "fetch_the_odds_api_prop_odds",
    "fetch_best_available_odds",
    "fetch_book_scraper_odds",
    "odds_frame_usable",
    "capture_all_book_cookies",
    "ensure_cookies_before_refresh",
]
