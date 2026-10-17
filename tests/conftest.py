"""Pytest path setup for monorepo (ufc-predictor + ufc_betting_bot)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREDICTOR = Path(__file__).resolve().parents[1]
if str(PREDICTOR) not in sys.path:
    sys.path.insert(0, str(PREDICTOR))

TEST_LIVE_BANKROLL = 1000.0


@pytest.fixture(autouse=True)
def _isolated_safety_state(tmp_path, monkeypatch):
    """Every test gets its own daily-loss file. The real cache file is never read or written.

    Dashboard bootstrap calls ``project_paths.patch_config``, which would otherwise
    reset ``CIRCUIT_BREAKER_STATE_PATH`` to the real cache. Keep the temp path.
    Production ``patch_config`` is unchanged; only the test monkeypatch wraps it.
    """
    import config
    from src import circuit_breaker
    from src import project_paths

    path = tmp_path / "circuit_breaker_state.json"
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_STATE_PATH", path)

    original_patch = project_paths.patch_config

    def _patch_config_keep_test_safety(root):
        original_patch(root)
        config.CIRCUIT_BREAKER_STATE_PATH = path

    monkeypatch.setattr(project_paths, "patch_config", _patch_config_keep_test_safety)
    circuit_breaker.set_entry_block(None)
    yield path
    circuit_breaker.set_entry_block(None)
    # If dashboard was imported and bootstrap already ran under a prior patch,
    # force the temp path back for any teardown that still reads config.
    config.CIRCUIT_BREAKER_STATE_PATH = path


@pytest.fixture
def live_session(_isolated_safety_state):
    """Today's Live session, started the way an operator would start it."""
    from src.circuit_breaker import initialize_live_session

    initialize_live_session(TEST_LIVE_BANKROLL)
    return _isolated_safety_state
