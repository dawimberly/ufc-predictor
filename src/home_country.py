"""Home-country / regional-home flags for research A/B (display-safe too).

True Sherdog/Wikipedia nationality caches are empty or noisy, so fighter
\"country\" is inferred leakage-safely from:
  1) gym location country when present in gyms.csv
  2) else modal country of the fighter's *prior* UFC events (as-of)

Event country comes from the fight ``location`` string.

Not added to production FEATURE_COLUMNS unless A/B keep rule passes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

import config
from src.data_loader import clean_fighter_name
from src.gym_data import _location_tokens, load_gym_profiles

logger = logging.getLogger(__name__)

HOME_COUNTRY_FEATURE_COLUMNS = (
    "home_country_diff",
    "home_country_rate_diff",
)

# City / region tokens → canonical country key (lowercase).
_CITY_COUNTRY: dict[str, str] = {
    # Australia / NZ
    "sydney": "australia",
    "melbourne": "australia",
    "perth": "australia",
    "brisbane": "australia",
    "adelaide": "australia",
    "queensland": "australia",
    "nsw": "australia",
    "qld": "australia",
    "australia": "australia",
    "sydney": "australia",
    "melbourne": "australia",
    "perth": "australia",
    "brisbane": "australia",
    "adelaide": "australia",
    "auckland": "new_zealand",
    "wellington": "new_zealand",
    "zealand": "new_zealand",
    # UK / Ireland
    "london": "uk",
    "manchester": "uk",
    "birmingham": "uk",
    "liverpool": "uk",
    "cardiff": "uk",
    "belfast": "uk",
    "newcastle": "uk",
    "glasgow": "uk",
    "edinburgh": "uk",
    "england": "uk",
    "scotland": "uk",
    "wales": "uk",
    "dublin": "ireland",
    "ireland": "ireland",
    # Brazil
    "brasilia": "brazil",
    "curitiba": "brazil",
    "fortaleza": "brazil",
    "goiania": "brazil",
    "natal": "brazil",
    "barueri": "brazil",
    "brazil": "brazil",
    "brasil": "brazil",
    # Canada
    "toronto": "canada",
    "vancouver": "canada",
    "montreal": "canada",
    "calgary": "canada",
    "ottawa": "canada",
    "winnipeg": "canada",
    "halifax": "canada",
    "canada": "canada",
    # UAE
    "dubai": "uae",
    "uae": "uae",
    "emirates": "uae",
    # Asia
    "tokyo": "japan",
    "saitama": "japan",
    "yokohama": "japan",
    "chiba": "japan",
    "osaka": "japan",
    "japan": "japan",
    "singapore": "singapore",
    "macau": "macau",
    "cotai": "macau",
    "shanghai": "china",
    "beijing": "china",
    "china": "china",
    "seoul": "south_korea",
    "manila": "philippines",
    "philippines": "philippines",
    "bangkok": "thailand",
    "phuket": "thailand",
    "thailand": "thailand",
    # Europe
    "stockholm": "sweden",
    "sweden": "sweden",
    "berlin": "germany",
    "cologne": "germany",
    "oberhausen": "germany",
    "germany": "germany",
    "paris": "france",
    "france": "france",
    "amsterdam": "netherlands",
    "rotterdam": "netherlands",
    "haarlem": "netherlands",
    "netherlands": "netherlands",
    "holland": "netherlands",
    "prague": "czech_republic",
    "brno": "czech_republic",
    "moscow": "russia",
    "dagestan": "russia",
    "russia": "russia",
    "poland": "poland",
    "gdansk": "poland",
    "krakow": "poland",
    "mexico": "mexico",
    "monterrey": "mexico",
    "guadalajara": "mexico",
    "johannesburg": "south_africa",
    "spain": "spain",
    "alicante": "spain",
    "madrid": "spain",
    "barcelona": "spain",
    # US states / cities
    "usa": "usa",
    "nevada": "usa",
    "california": "usa",
    "florida": "usa",
    "texas": "usa",
    "arizona": "usa",
    "colorado": "usa",
    "georgia": "usa",
    "ohio": "usa",
    "illinois": "usa",
    "pennsylvania": "usa",
    "washington": "usa",
    "oregon": "usa",
    "hawaii": "usa",
    "minnesota": "usa",
    "wisconsin": "usa",
    "connecticut": "usa",
    "louisiana": "usa",
    "tennessee": "usa",
    "massachusetts": "usa",
    "michigan": "usa",
    "missouri": "usa",
    "indiana": "usa",
    "oklahoma": "usa",
    "utah": "usa",
    "virginia": "usa",
    "maryland": "usa",
    "alabama": "usa",
    "mississippi": "usa",
    "arkansas": "usa",
    "iowa": "usa",
    "kansas": "usa",
    "nebraska": "usa",
    "idaho": "usa",
    "montana": "usa",
    "wyoming": "usa",
    "alaska": "usa",
    "denver": "usa",
    "phoenix": "usa",
    "houston": "usa",
    "dallas": "usa",
    "miami": "usa",
    "orlando": "usa",
    "atlanta": "usa",
    "chicago": "usa",
    "boston": "usa",
    "seattle": "usa",
    "portland": "usa",
    "sacramento": "usa",
    "anaheim": "usa",
    "brooklyn": "usa",
    "newark": "usa",
    "philadelphia": "usa",
    "pittsburgh": "usa",
    "cleveland": "usa",
    "columbus": "usa",
    "detroit": "usa",
    "milwaukee": "usa",
    "minneapolis": "usa",
    "nashville": "usa",
    "memphis": "usa",
    "albuquerque": "usa",
    "honolulu": "usa",
    "waianae": "usa",
    "englewood": "usa",
    "enumclaw": "usa",
    "vegas": "usa",
}

# USPS state abbrevs → usa (event strings like "Denver, CO")
_US_STATE = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
    "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
}


def location_to_country(location: Any) -> str:
    """Map a free-text event/gym location to a canonical country key (or \"\")."""
    text = str(location or "").strip()
    if not text:
        return ""
    low = text.lower()
    # Explicit country words first
    for needle, country in (
        ("gold coast", "australia"),
        ("australia", "australia"),
        ("new zealand", "new_zealand"),
        ("united kingdom", "uk"),
        ("united states", "usa"),
        ("sao paulo", "brazil"),
        ("rio de janeiro", "brazil"),
        ("belo horizonte", "brazil"),
        ("abu dhabi", "uae"),
        ("marina bay", "singapore"),
        ("new orleans", "usa"),
        ("las vegas", "usa"),
        ("los angeles", "usa"),
        ("san diego", "usa"),
        ("kansas city", "usa"),
        ("oklahoma city", "usa"),
        ("long island", "usa"),
        ("new mexico", "usa"),
        ("north carolina", "usa"),
        ("south carolina", "usa"),
        ("south africa", "south_africa"),
        ("united arab", "uae"),
        ("mexico city", "mexico"),
        ("brazil", "brazil"),
        ("brasil", "brazil"),
        ("canada", "canada"),
        ("japan", "japan"),
        ("china", "china"),
        ("mexico", "mexico"),
        ("ireland", "ireland"),
        ("france", "france"),
        ("germany", "germany"),
        ("sweden", "sweden"),
        ("netherlands", "netherlands"),
        ("russia", "russia"),
        ("spain", "spain"),
        ("thailand", "thailand"),
        ("singapore", "singapore"),
        ("philippines", "philippines"),
        ("czech", "czech_republic"),
    ):
        if needle in low:
            return country

    # "City, ST" US pattern
    m = re.search(r",\s*([A-Za-z]{2})\s*$", text)
    if m and m.group(1).lower() in _US_STATE:
        return "usa"
    # Canadian provinces
    if re.search(r",\s*(ON|BC|AB|QC|PQ|MB|NS|SK|NB|NL|PE|YT|NT|NU)\s*$", text, re.I):
        return "canada"

    toks = _location_tokens(text)
    votes: dict[str, int] = {}
    for t in toks:
        c = _CITY_COUNTRY.get(t)
        if c:
            votes[c] = votes.get(c, 0) + 1
    if not votes:
        return ""
    best = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0] == "usa", kv[0]))[0][0]
    return best


def _gym_country_lookup() -> dict[str, str]:
    profiles = load_gym_profiles()
    out: dict[str, str] = {}
    if profiles is None or profiles.empty:
        return out
    for _, row in profiles.iterrows():
        key = str(row.get("fighter_key") or clean_fighter_name(row.get("fighter_name")))
        if not key:
            continue
        c = location_to_country(row.get("location"))
        if c:
            out[key] = c
    return out


def _fighter_prior_country_mode(
    fights: pd.DataFrame,
    *,
    fid_col: str,
    date_col: str,
    f1_col: str,
    f2_col: str,
    loc_col: str,
) -> pd.DataFrame:
    """
    For each fight row, leakage-safe modal prior event-country per fighter.

    Returns fight_id, f1/f2_country_mode, f1/f2_home_rate.
    """
    work = fights[[fid_col, date_col, f1_col, f2_col, loc_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work["event_country"] = work[loc_col].map(location_to_country)
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    # Chronological appearances
    appearances: list[tuple[Any, Any, str, str]] = []  # date, fight_id, fighter, country
    for _, r in work.iterrows():
        ec = str(r["event_country"] or "")
        if not ec:
            continue
        d = r[date_col]
        fid = r[fid_col]
        for col in (f1_col, f2_col):
            name = clean_fighter_name(r[col])
            if name:
                appearances.append((d, fid, name, ec))

    # As-of stats keyed by (fight_id, fighter)
    from collections import Counter, defaultdict

    prior_counts: dict[str, Counter] = defaultdict(Counter)
    asof: dict[tuple[Any, str], tuple[str, Counter]] = {}
    for _d, fid, name, ec in sorted(appearances, key=lambda x: x[0]):
        counts = prior_counts[name]
        mode = counts.most_common(1)[0][0] if counts else ""
        asof[(fid, name)] = (mode, Counter(counts))
        prior_counts[name][ec] += 1

    rows = []
    for _, r in work.iterrows():
        fid = r[fid_col]
        ec = str(r["event_country"] or "")
        f1n = clean_fighter_name(r[f1_col])
        f2n = clean_fighter_name(r[f2_col])
        m1, c1 = asof.get((fid, f1n), ("", Counter()))
        m2, c2 = asof.get((fid, f2n), ("", Counter()))
        n1 = sum(c1.values())
        n2 = sum(c2.values())
        rows.append(
            {
                fid_col: fid,
                "f1_country_mode": m1,
                "f2_country_mode": m2,
                "f1_home_rate": (c1.get(ec, 0) / n1) if n1 and ec else 0.0,
                "f2_home_rate": (c2.get(ec, 0) / n2) if n2 and ec else 0.0,
            }
        )
    return pd.DataFrame(rows).drop_duplicates(fid_col, keep="last")


def attach_home_country_features(
    df: pd.DataFrame,
    *,
    fights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Attach event_country, fighter country proxies, and home_country_* diffs.

    Prefer gym country when known; else prior-event modal country.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df

    out = df.copy()
    fid = config.FIGHT_ID_COLUMN
    date_col = config.DATE_COLUMN
    f1 = "fighter_1" if "fighter_1" in out.columns else "fighter1"
    f2 = "fighter_2" if "fighter_2" in out.columns else "fighter2"
    if f1 not in out.columns or f2 not in out.columns:
        return out

    src = fights if isinstance(fights, pd.DataFrame) and not fights.empty else out
    if "location" not in out.columns and isinstance(fights, pd.DataFrame) and "location" in fights.columns:
        loc = fights[[fid, "location"]].drop_duplicates(fid, keep="last")
        out = out.merge(loc, on=fid, how="left")

    if "location" in out.columns:
        out["event_country"] = out["location"].map(location_to_country)
    else:
        out["event_country"] = ""

    gym_c = _gym_country_lookup()
    out["f1_gym_country"] = out[f1].map(lambda n: gym_c.get(clean_fighter_name(n), ""))
    out["f2_gym_country"] = out[f2].map(lambda n: gym_c.get(clean_fighter_name(n), ""))

    # Prior-mode from full fight history when available
    prior = pd.DataFrame()
    if (
        isinstance(src, pd.DataFrame)
        and fid in src.columns
        and date_col in src.columns
        and "location" in src.columns
    ):
        sf1 = "fighter_1" if "fighter_1" in src.columns else "fighter1"
        sf2 = "fighter_2" if "fighter_2" in src.columns else "fighter2"
        try:
            prior = _fighter_prior_country_mode(
                src,
                fid_col=fid,
                date_col=date_col,
                f1_col=sf1,
                f2_col=sf2,
                loc_col="location",
            )
        except Exception as exc:
            logger.warning("prior country mode failed: %s", exc)
            prior = pd.DataFrame()

    if not prior.empty and fid in prior.columns:
        keep = [
            c
            for c in (
                fid,
                "f1_country_mode",
                "f2_country_mode",
                "f1_home_rate",
                "f2_home_rate",
            )
            if c in prior.columns
        ]
        out = out.drop(columns=[c for c in keep if c != fid and c in out.columns], errors="ignore")
        out = out.merge(prior[keep], on=fid, how="left")
    else:
        out["f1_country_mode"] = ""
        out["f2_country_mode"] = ""
        out["f1_home_rate"] = 0.0
        out["f2_home_rate"] = 0.0

    def _fighter_country(gym: Any, mode: Any) -> str:
        g = str(gym or "").strip()
        if g:
            return g
        return str(mode or "").strip()

    out["f1_country"] = [
        _fighter_country(g, m)
        for g, m in zip(out["f1_gym_country"], out.get("f1_country_mode", ""))
    ]
    out["f2_country"] = [
        _fighter_country(g, m)
        for g, m in zip(out["f2_gym_country"], out.get("f2_country_mode", ""))
    ]

    ec = out["event_country"].fillna("").astype(str)
    out["f1_home_country"] = [
        int(bool(c) and bool(e) and c == e) for c, e in zip(out["f1_country"], ec)
    ]
    out["f2_home_country"] = [
        int(bool(c) and bool(e) and c == e) for c, e in zip(out["f2_country"], ec)
    ]
    out["home_country_diff"] = (
        pd.to_numeric(out["f1_home_country"], errors="coerce").fillna(0).astype(int)
        - pd.to_numeric(out["f2_home_country"], errors="coerce").fillna(0).astype(int)
    )
    out["home_country_rate_diff"] = (
        pd.to_numeric(out.get("f1_home_rate"), errors="coerce").fillna(0.0)
        - pd.to_numeric(out.get("f2_home_rate"), errors="coerce").fillna(0.0)
    )
    return out


def log_home_country_coverage(df: pd.DataFrame, *, year: int | None = None, label: str = "") -> None:
    work = df
    if year is not None and config.DATE_COLUMN in df.columns:
        dts = pd.to_datetime(df[config.DATE_COLUMN], errors="coerce")
        work = df.loc[dts.dt.year == year]
    n = len(work)
    if n == 0:
        logger.info("home_country coverage %s: empty", label or year)
        return
    ec = work["event_country"].fillna("").astype(str).str.len().gt(0).mean() if "event_country" in work.columns else 0
    nz = (pd.to_numeric(work.get("home_country_diff"), errors="coerce").fillna(0) != 0).mean()
    any_home = (
        (
            pd.to_numeric(work.get("f1_home_country"), errors="coerce").fillna(0)
            + pd.to_numeric(work.get("f2_home_country"), errors="coerce").fillna(0)
        )
        > 0
    ).mean()
    logger.info(
        "home_country coverage %s n=%s event_country=%.1f%% any_home=%.1f%% nonzero_diff=%.1f%%",
        label or year,
        n,
        100 * ec,
        100 * any_home,
        100 * nz,
    )
