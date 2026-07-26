"""Tests for hard-coded Grok API key loading paths."""

from __future__ import annotations

import os
from pathlib import Path

import src.project_paths as project_paths
from src.project_paths import load_env_files, load_grok_api_key, shared_trading_env_candidates


def test_shared_trading_env_candidates_includes_stock_bot(tmp_path, monkeypatch):
    root = tmp_path / "ufc-predictor"
    root.mkdir()
    repo = tmp_path
    stock_env = repo / "stock-bot" / ".env"
    stock_env.parent.mkdir(parents=True)
    stock_env.write_text("GROK_API_KEY=from-stock-bot\n", encoding="utf-8")

    paths = shared_trading_env_candidates(root)
    assert stock_env.resolve() in [p.resolve() for p in paths]


def test_load_grok_api_key_from_second_path(tmp_path, monkeypatch):
    root = tmp_path / "ufc-predictor"
    root.mkdir()
    root_env = tmp_path / ".env"
    root_env.write_text("XAI_API_KEY=shared-xai-key\n", encoding="utf-8")

    monkeypatch.setattr(
        project_paths,
        "GROK_ENV_LOAD_ORDER",
        [
            (tmp_path / "stock-bot" / ".env", True),
            (root_env, False),
            (tmp_path / "ufc-bot" / ".env", False),
        ],
    )
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    source = load_grok_api_key()
    import config

    config.refresh_runtime_env()

    assert config.GROK_API_KEY == "shared-xai-key"
    assert source == root_env.resolve()
    assert config.GROK_API_KEY_SOURCE == str(root_env.resolve())


def test_load_grok_api_key_stock_bot_wins_with_override(tmp_path, monkeypatch):
    stock_env = tmp_path / "stock-bot" / ".env"
    stock_env.parent.mkdir(parents=True)
    stock_env.write_text("GROK_API_KEY=from-stock-bot\n", encoding="utf-8")
    root_env = tmp_path / ".env"
    root_env.write_text("GROK_API_KEY=from-root\n", encoding="utf-8")

    monkeypatch.setattr(
        project_paths,
        "GROK_ENV_LOAD_ORDER",
        [
            (stock_env, True),
            (root_env, False),
            (tmp_path / "ufc-bot" / ".env", False),
        ],
    )
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    source = load_grok_api_key()
    assert source == stock_env.resolve()
    assert os.getenv("GROK_API_KEY") == "from-stock-bot"


def test_load_grok_api_key_logs_success(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("GROK_API_KEY=local-key\n", encoding="utf-8")

    monkeypatch.setattr(
        project_paths,
        "GROK_ENV_LOAD_ORDER",
        [
            (tmp_path / "missing" / ".env", True),
            (env_path, False),
            (tmp_path / "ufc-bot" / ".env", False),
        ],
    )
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    logs: list[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)

    load_grok_api_key(log=_log)
    assert any("GROK_API_KEY loaded from" in line and "len=" in line for line in logs)
    assert os.getenv("GROK_API_KEY") == "local-key"


def test_load_grok_api_key_logs_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        project_paths,
        "GROK_ENV_LOAD_ORDER",
        [
            (tmp_path / "a" / ".env", True),
            (tmp_path / "b" / ".env", False),
            (tmp_path / "c" / ".env", False),
        ],
    )
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    logs: list[str] = []
    load_grok_api_key(log=lambda msg: logs.append(msg))
    assert any("GROK_API_KEY not found" in line for line in logs)


def test_load_env_files_calls_grok_loader(tmp_path, monkeypatch):
    env_path = tmp_path / "repo" / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("GROK_API_KEY=via-load-env\n", encoding="utf-8")

    monkeypatch.setattr(
        project_paths,
        "GROK_ENV_LOAD_ORDER",
        [
            (tmp_path / "stock-bot" / ".env", True),
            (env_path, False),
            (tmp_path / "ufc-bot" / ".env", False),
        ],
    )
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    load_env_files(tmp_path / "ufc-predictor")
    import config

    config.refresh_runtime_env()
    assert config.GROK_API_KEY == "via-load-env"
