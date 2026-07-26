"""Shared fighter name aliases for external source matching.

Handles accents, hyphens, middle names, and known Sherdog/Wiki spelling variants.
Fail-soft: unknown names pass through unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Canonical (pipeline) name -> preferred external lookup spellings (ordered).
# First entry is typically the most common Wikipedia/Sherdog form.
FIGHTER_NAME_ALIASES: dict[str, list[str]] = {
    "benoit saint denis": [
        "Benoit St. Denis",
        "Benoît St. Denis",
        "Benoit St Denis",
        "Benoît Saint-Denis",
        "Benoit Saint-Denis",
        "Benoît Saint Denis",
        "Benoit Saint Denis",
    ],
    "benoit st denis": [
        "Benoit St. Denis",
        "Benoît Saint-Denis",
        "Benoit Saint Denis",
    ],
    "benoit st. denis": [
        "Benoit St. Denis",
        "Benoît Saint-Denis",
        "Benoit Saint Denis",
    ],
    "ian machado garry": [
        "Ian Garry",
        "Ian Machado Garry",
    ],
    "ian garry": [
        "Ian Garry",
        "Ian Machado Garry",
    ],
    "alexandr rakic": [
        "Aleksandar Rakić",
        "Aleksandar Rakic",
        "Alexandr Rakic",
        "Alexander Rakic",
    ],
    "aleksandar rakic": [
        "Aleksandar Rakić",
        "Aleksandar Rakic",
        "Alexandr Rakic",
    ],
    "ramazan temirov": [
        "Ramazonbek Temirov",
        "Ramazan Temirov",
        "Razonbek Temirov",
        "Rozanbek Temirov",
        "Ramazonbek Temurlan Temirov",
    ],
    "ramazonbek temirov": [
        "Ramazonbek Temirov",
        "Ramazan Temirov",
        "Razonbek Temirov",
    ],
    # Common accent / spelling variants for other roster names
    "jiri prochazka": ["Jiří Procházka", "Jiri Prochazka"],
    "ciryl gane": ["Ciryl Gane", "Cyril Gane"],
    "dricus du plessis": ["Dricus du Plessis", "Dricus Du Plessis"],
    "zhang weili": ["Zhang Weili", "Weili Zhang"],
    "ilia topuria": ["Ilia Topuria"],
    "khamzat chimaev": ["Khamzat Chimaev"],
    "magomed ankalaev": ["Magomed Ankalaev"],
    "umer nurmagomedov": ["Umar Nurmagomedov"],
    "umar nurmagomedov": ["Umar Nurmagomedov"],
    "movsar evloev": ["Movsar Evloev"],
    "yair rodriguez": ["Yair Rodríguez", "Yair Rodriguez"],
    "volkan oezdemir": ["Volkan Oezdemir", "Volkan Özdemir"],
    "jiří procházka": ["Jiří Procházka", "Jiri Prochazka"],
    "benoît saint-denis": [
        "Benoit St. Denis",
        "Benoît Saint-Denis",
        "Benoit Saint-Denis",
        "Benoit Saint Denis",
    ],
    "aleksandar rakić": [
        "Aleksandar Rakić",
        "Aleksandar Rakic",
        "Alexandr Rakic",
    ],
}

# Known Sherdog IDs when search HTML is unreliable (fail-soft overrides).
SHERDOG_ID_OVERRIDES: dict[str, str] = {
    "benoit saint denis": "317103",
    "benoit st denis": "317103",
    "ramazan temirov": "215649",
    "ramazonbek temirov": "215649",
}

# Lightweight accent / punctuation variants applied to any name (no network).
_GENERAL_ACCENT_SWAPS: tuple[tuple[str, str], ...] = (
    ("é", "e"),
    ("è", "e"),
    ("ê", "e"),
    ("á", "a"),
    ("à", "a"),
    ("í", "i"),
    ("ó", "o"),
    ("ú", "u"),
    ("ñ", "n"),
    ("ć", "c"),
    ("č", "c"),
    ("š", "s"),
    ("ž", "z"),
    ("ř", "r"),
    ("ý", "y"),
    ("ö", "o"),
    ("ü", "u"),
    ("ä", "a"),
)


def strip_accents(text: str) -> str:
    """Remove diacritics: Benoît → Benoit, Rakić → Rakic."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_alias_key(name: str) -> str:
    """Lowercase, strip accents, collapse punctuation/whitespace for alias lookup."""
    text = strip_accents(name).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, list[str]]:
    """Build bidirectional alias index keyed by normalized forms."""
    index: dict[str, list[str]] = {}
    for canon_key, variants in FIGHTER_NAME_ALIASES.items():
        variants_list = list(variants)
        # Ensure canonical spelling is included
        if canon_key not in {normalize_alias_key(v) for v in variants_list}:
            variants_list = [canon_key.title()] + variants_list
        keys = {normalize_alias_key(canon_key)}
        keys.update(normalize_alias_key(v) for v in variants_list)
        for k in keys:
            existing = index.get(k, [])
            # Prefer longer / accented variants first for external search
            merged = list(dict.fromkeys(variants_list + existing))
            index[k] = merged
    return index


def alias_lookup_names(name: str) -> list[str]:
    """Ordered lookup spellings for external sources (aliases first, then original)."""
    from src.data_loader import clean_fighter_name

    clean = clean_fighter_name(name)
    if not clean:
        return []
    key = normalize_alias_key(clean)
    aliases = list(_alias_index().get(key, []))
    stripped = strip_accents(clean)
    dehyphen = re.sub(r"\s+", " ", stripped.replace("-", " ")).strip()
    ordered: list[str] = []
    for candidate in aliases + [clean, stripped, dehyphen]:
        c = candidate.strip()
        if c and c not in ordered:
            ordered.append(c)
        if any(src in c.lower() for src, _ in _GENERAL_ACCENT_SWAPS):
            swapped = c
            for src, dst in _GENERAL_ACCENT_SWAPS:
                swapped = swapped.replace(src, dst).replace(src.upper(), dst.upper())
            if swapped.strip() and swapped.strip() not in ordered:
                ordered.append(swapped.strip())
    return ordered


def apply_alias_for_search(name: str) -> str:
    """Single best external-search spelling (first alias if any)."""
    names = alias_lookup_names(name)
    return names[0] if names else str(name or "")


def names_match_aliased(a: str, b: str) -> bool:
    """True if a/b are the same person under aliases + accent-normalized tokens."""
    from src.data_loader import _fighters_same_person, clean_fighter_name

    ca, cb = clean_fighter_name(a), clean_fighter_name(b)
    if not ca or not cb:
        return False
    if _fighters_same_person(ca, cb):
        return True
    if normalize_alias_key(ca) == normalize_alias_key(cb):
        return True
    a_names = alias_lookup_names(ca)
    b_names = alias_lookup_names(cb)
    if {normalize_alias_key(x) for x in a_names} & {normalize_alias_key(x) for x in b_names}:
        return True
    for an in a_names:
        for bn in b_names:
            if _fighters_same_person(an, bn):
                return True
    return False