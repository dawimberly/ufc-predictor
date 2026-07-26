"""Odds providers — scrape / API adapters (skeleton)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests

from sports_bot.core import config


@dataclass
class OddsLine:
    event: str
    market: str  # moneyline | total | prop
    selection: str
    book: str
    decimal_odds: float
    meta: dict[str, Any] | None = None


class OddsProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_event_odds(self, event_query: str) -> list[OddsLine]:
        raise NotImplementedError


class TheOddsApiProvider(OddsProvider):
    """https://the-odds-api.com — requires THE_ODDS_API_KEY."""

    name = "the_odds_api"
    sport_key = "mma_mixed_martial_arts"

    def fetch_event_odds(self, event_query: str) -> list[OddsLine]:
        if not config.ODDS_API_KEY:
            return []
        url = (
            f"https://api.the-odds-api.com/v4/sports/{self.sport_key}/odds"
            f"?apiKey={config.ODDS_API_KEY}&regions=us&markets=h2h&oddsFormat=decimal"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines: list[OddsLine] = []
        q = event_query.lower().strip()
        for event in resp.json():
            label = f"{event.get('home_team')} vs {event.get('away_team')}"
            if q and q not in label.lower():
                continue
            for book in event.get("bookmakers") or []:
                for market in book.get("markets") or []:
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes") or []:
                        lines.append(
                            OddsLine(
                                event=label,
                                market="moneyline",
                                selection=str(outcome.get("name") or ""),
                                book=str(book.get("title") or self.name),
                                decimal_odds=float(outcome.get("price") or 0),
                            )
                        )
        return lines


class ScraperProvider(OddsProvider):
    """
    Placeholder for DraftKings / BetNow / MyBookie HTML or API scrapers.

    Port battle-tested scrapers from C:\\UFC-Predictor\\src\\odds_providers when ready.
    """

    name = "scraper"

    def __init__(self, book: str) -> None:
        self.book = book
        self.name = book

    def fetch_event_odds(self, event_query: str) -> list[OddsLine]:
        # Skeleton: return empty until scrapers are ported.
        _ = event_query
        return []


def active_providers() -> list[OddsProvider]:
    providers: list[OddsProvider] = []
    if config.ODDS_API_KEY:
        providers.append(TheOddsApiProvider())
    if config.ENABLE_ODDS_SCRAPE:
        for book in ("DraftKings", "BetNow.eu", "MyBookie"):
            providers.append(ScraperProvider(book))
    return providers
