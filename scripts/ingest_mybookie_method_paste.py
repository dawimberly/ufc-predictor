"""Merge pasted MyBookie method-of-victory lines into mybookie_prop_odds.csv."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.project_paths import bootstrap

bootstrap()

import pandas as pd

import config
from src.odds_providers.mybookie_scraper import MYBOOKIE_PROP_CACHE_PATH
from src.odds_providers.prop_odds_common import american_to_decimal, prop_row, remap_totals_prop_keys
from src.predictor import _names_match

# (fighter_1, fighter_2) as on MyBookie board, then method lines:
# fighter_last_first_or_first_last, method, american
# Fights from Aug 15 UFC card paste (KO / submission / decision). Draws omitted.
_PASTE: list[tuple[str, str, list[tuple[str, str, int]]]] = [
    (
        "Jeremiah Wells",
        "Myktybek Orolbay Uulu",
        [
            ("Myktybek Orolbay Uulu", "ko", 165),
            ("Myktybek Orolbay Uulu", "sub", 185),
            ("Myktybek Orolbay Uulu", "dec", 226),
            ("Jeremiah Wells", "ko", 900),
            ("Jeremiah Wells", "dec", 900),
            ("Jeremiah Wells", "sub", 1800),
        ],
    ),
    (
        "Neil Magny",
        "Ramiz Brahimaj",
        [
            ("Ramiz Brahimaj", "sub", 158),
            ("Neil Magny", "dec", 280),
            ("Ramiz Brahimaj", "ko", 470),
            ("Ramiz Brahimaj", "dec", 480),
            ("Neil Magny", "ko", 600),
            ("Neil Magny", "sub", 610),
        ],
    ),
    (
        "Rafael Tobias",
        "Lucas Fernando",
        [
            ("Lucas Fernando", "ko", 125),
            ("Lucas Fernando", "dec", 191),
            ("Rafael Tobias", "dec", 580),
            ("Rafael Tobias", "sub", 600),
            ("Lucas Fernando", "sub", 790),
            ("Rafael Tobias", "ko", 1000),
        ],
    ),
    (
        "Vicente Luque",
        "Tresean Gore",
        [
            ("Vicente Luque", "dec", 285),
            ("Tresean Gore", "ko", 340),
            ("Tresean Gore", "sub", 340),
            ("Tresean Gore", "dec", 390),
            ("Vicente Luque", "sub", 440),
            ("Vicente Luque", "ko", 450),
        ],
    ),
    (
        "Donte Johnson",
        "Eric McConico",
        [
            ("Donte Johnson", "ko", -109),
            ("Donte Johnson", "dec", 280),
            ("Eric McConico", "dec", 500),
            ("Eric McConico", "ko", 630),
            ("Donte Johnson", "sub", 680),
            ("Eric McConico", "sub", 1400),
        ],
    ),
    (
        "Charles Johnson",
        "Eduardo Henrique Da Silva Dos Santos",
        [
            ("Eduardo Henrique Da Silva Dos Santos", "dec", 171),
            ("Charles Johnson", "dec", 180),
            ("Charles Johnson", "ko", 310),
            ("Eduardo Henrique Da Silva Dos Santos", "ko", 450),
            ("Charles Johnson", "sub", 1400),
            ("Eduardo Henrique Da Silva Dos Santos", "sub", 1475),
        ],
    ),
    (
        "Chidi Njokuani",
        "Joel Alvarez",
        [
            ("Joel Alvarez", "sub", 155),
            ("Joel Alvarez", "ko", 224),
            ("Joel Alvarez", "dec", 320),
            ("Chidi Njokuani", "ko", 510),
            ("Chidi Njokuani", "dec", 590),
            ("Chidi Njokuani", "sub", 2900),
        ],
    ),
    (
        "Edson Barboza",
        "Esteban Ribovics",
        [
            ("Esteban Ribovics", "ko", -242),
            ("Esteban Ribovics", "dec", 310),
            ("Edson Barboza", "dec", 800),
            ("Edson Barboza", "ko", 850),
            ("Esteban Ribovics", "sub", 1125),
            ("Edson Barboza", "sub", 2800),
        ],
    ),
    (
        "Mansur Abdul-Malik",
        "Dustin Stoltzfus",
        [
            ("Mansur Abdul-Malik", "ko", -106),
            ("Mansur Abdul-Malik", "dec", 189),
            ("Mansur Abdul-Malik", "sub", 590),
            ("Dustin Stoltzfus", "dec", 710),
            ("Dustin Stoltzfus", "sub", 950),
            ("Dustin Stoltzfus", "ko", 1525),
        ],
    ),
    (
        "Jalin Turner",
        "Kaue Fernandes",
        [
            ("Jalin Turner", "ko", 210),
            ("Kaue Fernandes", "ko", 244),
            ("Jalin Turner", "sub", 430),
            ("Jalin Turner", "dec", 440),
            ("Kaue Fernandes", "dec", 450),
            ("Kaue Fernandes", "sub", 840),
        ],
    ),
    (
        "Mackenzie Dern",
        "Gillian Robertson",
        [
            ("Mackenzie Dern", "dec", 157),
            ("Mackenzie Dern", "sub", 206),
            ("Gillian Robertson", "dec", 228),
            ("Mackenzie Dern", "ko", 1100),
            ("Gillian Robertson", "ko", 1150),
            ("Gillian Robertson", "sub", 1150),
        ],
    ),
    (
        "Islam Makhachev",
        "Ian Machado Garry",
        [
            ("Islam Makhachev", "dec", 115),
            ("Islam Makhachev", "sub", 191),
            ("Ian Machado Garry", "dec", 430),
            ("Islam Makhachev", "ko", 720),
            ("Ian Machado Garry", "ko", 720),
            ("Ian Machado Garry", "sub", 2400),
        ],
    ),
]

_METHOD_TO_KEY = {
    "ko": "fighter_ko",
    "sub": "fighter_sub",
    "dec": "fighter_decision",
}


def _canon_fighter(name: str, pool: list[str]) -> str:
    for cand in pool:
        if _names_match(name, cand):
            return cand
    # Unique first-token fallback (Orolbay ↔ Orolbai)
    first = name.split()[0].lower() if name.split() else ""
    if len(first) >= 4:
        hits = [c for c in pool if c.lower().startswith(first) or first in c.lower()]
        if len(hits) == 1:
            return hits[0]
    last = name.split()[-1].lower() if name.split() else ""
    if len(last) >= 5:
        hits = []
        for c in pool:
            cl = c.split()[-1].lower() if c.split() else ""
            if cl.startswith(last[:4]) or last.startswith(cl[:4]):
                hits.append(c)
        if len(hits) == 1:
            return hits[0]
    return name


def _resolve_fight(f1: str, f2: str, card: list[tuple[str, str]]) -> tuple[str, str]:
    for a, b in card:
        if (_names_match(f1, a) and _names_match(f2, b)) or (
            _names_match(f1, b) and _names_match(f2, a)
        ):
            return a, b
        # Soft first-name match both sides
        pool = [a, b]
        c1, c2 = _canon_fighter(f1, pool), _canon_fighter(f2, pool)
        if {c1, c2} == {a, b}:
            return a, b
    return f1, f2


def main() -> None:
    path = Path(MYBOOKIE_PROP_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = remap_totals_prop_keys(pd.read_csv(path))
    else:
        existing = pd.DataFrame()

    card: list[tuple[str, str]] = []
    if not existing.empty:
        for _, r in existing.iterrows():
            pair = (str(r["fighter_1"]).strip(), str(r["fighter_2"]).strip())
            if pair not in card and pair[0] and pair[1]:
                card.append(pair)
    # Prefer prediction card spellings when available
    feat = ROOT / "data" / "cache" / "event_analysis" / "ufc_330_6babbe8859d6" / "features.parquet"
    if feat.is_file():
        pdf = pd.read_parquet(feat)
        for _, r in pdf.iterrows():
            pair = (str(r["fighter_1"]).strip(), str(r["fighter_2"]).strip())
            # Prefer card spelling: replace soft-matching totals names
            replaced = False
            for i, (a, b) in enumerate(card):
                if (
                    (_names_match(a, pair[0]) and _names_match(b, pair[1]))
                    or (_names_match(a, pair[1]) and _names_match(b, pair[0]))
                    or ({_canon_fighter(a, list(pair)), _canon_fighter(b, list(pair))} == set(pair))
                ):
                    card[i] = pair
                    replaced = True
                    break
            if not replaced and pair not in card:
                card.append(pair)

    new_rows: list[dict] = []
    for f1, f2, lines in _PASTE:
        cf1, cf2 = _resolve_fight(f1, f2, card)
        pool = [cf1, cf2]
        for fighter, method, american in lines:
            prop_key = _METHOD_TO_KEY[method]
            named = _canon_fighter(fighter, pool)
            decimal = american_to_decimal(float(american))
            new_rows.append(
                prop_row(
                    fighter_1=cf1,
                    fighter_2=cf2,
                    prop_key=prop_key,
                    selection=f"{named} Yes",
                    decimal_odds=decimal,
                    bookmaker="MyBookie",
                    odds_source="live",
                    market_key="prop",
                    american_odds=float(american),
                )
            )

    method_df = pd.DataFrame(new_rows)
    if existing.empty:
        out = method_df
    else:
        keep = existing[~existing["prop_key"].astype(str).isin(_METHOD_TO_KEY.values())]
        out = pd.concat([keep, method_df], ignore_index=True)

    out = remap_totals_prop_keys(out)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} prop rows -> {path}")
    print(out["prop_key"].value_counts().to_string())


if __name__ == "__main__":
    main()
