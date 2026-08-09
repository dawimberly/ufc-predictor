"""Tests for weigh-in context helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.weigh_in import (
    athlete_slug,
    format_weigh_in_line,
    lookup_weigh_in_note,
    reload_weigh_in_notes,
)


def test_athlete_slug_strips_accents():
    assert athlete_slug("José Montanha") == "jose-montanha"
    assert athlete_slug("Mateusz Gamrot") == "mateusz-gamrot"


def test_weigh_in_note_lookup(tmp_path: Path, monkeypatch):
    notes = {
        "version": 1,
        "notes": [
            {
                "fighter": "Diego Ferreira",
                "event": "UFC Fight Night",
                "date": "2026-08-08",
                "missed_weight": True,
                "weighed_lb": 157.5,
                "limit_lb": 156.0,
                "note": "test row",
            }
        ],
    }
    path = tmp_path / "weigh_in_notes.json"
    path.write_text(json.dumps(notes), encoding="utf-8")
    monkeypatch.setattr("src.weigh_in.NOTES_PATH", path)
    reload_weigh_in_notes()
    hit = lookup_weigh_in_note("Diego Ferreira", date="2026-08-08")
    assert hit is not None
    assert hit.missed_weight is True
    line = format_weigh_in_line("Diego Ferreira", "Billy Quarantillo", date="2026-08-08")
    assert line and "MISSED WEIGHT" in line
