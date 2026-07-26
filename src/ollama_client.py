"""Local Ollama client for UFC card analysis (mirrors stock-bot ollama_client)."""

from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.request
from typing import Any

import config

logger = logging.getLogger(__name__)

# Prefer these after a timeout on a heavy/reasoning model.
_FAST_MODEL_PREFS: tuple[str, ...] = (
    "llama3.2:3b",
    "qwen2.5:7b",
    "qwen2.5-coder:7b",
)
_HEAVY_MODEL_MARKERS: tuple[str, ...] = (
    "deepseek-r1",
    "deepseek",
    ":14b",
    ":32b",
    ":70b",
    ":72b",
)


def _host() -> str:
    return str(getattr(config, "OLLAMA_HOST", "http://localhost:11434") or "").rstrip("/")


def ollama_timeout_sec() -> int:
    return max(5, int(getattr(config, "OLLAMA_TIMEOUT_SEC", 600) or 600))


def ollama_available() -> bool:
    """True when the local Ollama daemon responds to /api/tags."""
    try:
        req = urllib.request.Request(f"{_host()}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return int(getattr(resp, "status", 200) or 200) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ollama_installed_models() -> set[str]:
    try:
        req = urllib.request.Request(f"{_host()}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return set()
    names: set[str] = set()
    for item in body.get("models") or []:
        name = str(item.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def resolve_model(model: str, installed: set[str] | None = None) -> str | None:
    installed = installed if installed is not None else ollama_installed_models()
    if not installed:
        return model
    if model in installed:
        return model
    if ":" not in model:
        for name in sorted(installed):
            if name.startswith(f"{model}:"):
                return name
    base = model.split(":")[0]
    for name in sorted(installed):
        if name.startswith(f"{base}:") or name == base:
            return name
    return None


def _is_heavy_model(name: str) -> bool:
    low = str(name or "").lower()
    return any(marker in low for marker in _HEAVY_MODEL_MARKERS)


def _model_param_b(name: str) -> int:
    """Rough parameter size in billions for ranking speed (higher = slower)."""
    low = str(name or "").lower()
    if "deepseek-r1" in low or ("deepseek" in low and "r1" in low):
        return 64  # reasoning models are much slower than param count implies
    m = re.search(r":(\d+(?:\.\d+)?)\s*b\b", low)
    if m:
        try:
            return int(float(m.group(1)))
        except ValueError:
            pass
    if _is_heavy_model(name):
        return 32
    return 7


def _suggested_fast_models(installed: set[str] | None = None) -> list[str]:
    installed = installed if installed is not None else ollama_installed_models()
    out: list[str] = []
    for pref in _FAST_MODEL_PREFS:
        pick = resolve_model(pref, installed) if installed else pref
        if pick and pick not in out:
            if installed and pick not in installed:
                continue
            out.append(pick)
    if installed:
        for name in sorted(installed, key=_model_param_b):
            if not _is_heavy_model(name) and name not in out and _model_param_b(name) <= 8:
                out.append(name)
    return out or list(_FAST_MODEL_PREFS[:3])


def resolve_model_chain() -> list[str]:
    """Primary first; optional smaller fallbacks only (never auto-escalate to heavier)."""
    primary = str(
        getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:7b") or "qwen2.5-coder:7b"
    ).strip()
    fallbacks = [
        m.strip()
        for m in str(getattr(config, "OLLAMA_FALLBACK_MODELS", "") or "").split(",")
        if m.strip()
    ]
    candidates = [primary]
    for m in fallbacks:
        if m not in candidates:
            candidates.append(m)

    installed = ollama_installed_models()
    if not installed:
        return candidates

    resolved: list[str] = []
    for model in candidates:
        pick = resolve_model(model, installed)
        if pick and pick not in resolved:
            resolved.append(pick)
    if not resolved:
        return [primary]

    primary_resolved = resolve_model(primary, installed) or resolved[0]
    primary_size = _model_param_b(primary_resolved)
    # Keep primary first; only keep fallbacks that are strictly smaller/faster.
    out = [primary_resolved]
    for m in resolved:
        if m == primary_resolved:
            continue
        if _model_param_b(m) < primary_size and not _is_heavy_model(m):
            out.append(m)
    return out


def list_model_choices() -> list[str]:
    """Models for the dashboard picker: configured primary first, then installed."""
    installed = sorted(ollama_installed_models())
    primary = str(getattr(config, "OLLAMA_MODEL", "") or "").strip()
    choices: list[str] = []
    if primary:
        choices.append(primary)
    for name in _suggested_fast_models(set(installed)):
        if name not in choices:
            choices.append(name)
    for name in installed:
        if name not in choices:
            choices.append(name)
    for extra in (
        "llama3.2:3b",
        "qwen2.5:7b",
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b",
        "deepseek-r1:8b",
    ):
        if extra not in choices:
            choices.append(extra)
    return choices or ["qwen2.5-coder:7b"]


def set_active_model(model: str) -> str:
    """Update runtime config model (session); returns resolved name."""
    name = str(model or "").strip() or str(
        getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:7b")
    )
    config.OLLAMA_MODEL = name
    return name


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def _timeout_message(model: str, timeout: int, *, tried: list[str] | None = None) -> str:
    suggestions = _suggested_fast_models()
    suggest_txt = ", ".join(suggestions[:3]) if suggestions else "llama3.2:3b / qwen2.5:7b"
    tried_txt = ""
    if tried:
        tried_txt = f" Tried: {', '.join(tried)}. "
    return (
        f"Ollama timed out after {timeout}s using {model}.{tried_txt} "
        f"Keep OLLAMA_MODEL=qwen2.5-coder:7b (or pull llama3.2:3b), "
        f"raise OLLAMA_TIMEOUT_SEC (now {ollama_timeout_sec()}s), or lower GROK_MAX_FIGHTS. "
        f"Faster pulls: {suggest_txt}."
    )


def _http_post(path: str, body: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{_host()}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        if _is_timeout_error(exc):
            model = str(body.get("model") or getattr(config, "OLLAMA_MODEL", "ollama"))
            raise TimeoutError(_timeout_message(model, timeout)) from exc
        raise


def _merge_thinking_response(body: dict[str, Any]) -> str:
    """Prefer final answer; include deepseek thinking when needed for JSON parse."""
    answer = str(body.get("response") or body.get("message", {}).get("content") or "").strip()
    thinking = str(body.get("thinking") or "").strip()
    if answer:
        return answer
    if thinking:
        return thinking
    raise RuntimeError("Ollama returned empty response")


def _reorder_after_timeout(remaining: list[str], failed: str) -> list[str]:
    """After a timeout, only retry strictly smaller/faster installed models."""
    failed_size = _model_param_b(failed)
    rest = [m for m in remaining if m != failed and _model_param_b(m) < failed_size]
    extras: list[str] = []
    for pref in _suggested_fast_models():
        if pref == failed or pref in rest:
            continue
        if _model_param_b(pref) < failed_size and not _is_heavy_model(pref):
            extras.append(pref)
    ordered = [*extras, *rest]
    seen: set[str] = set()
    out: list[str] = []
    for m in ordered:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def ollama_complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    timeout_sec: int | None = None,
    json_mode: bool = True,
    temperature: float = 0.25,
) -> tuple[str, str]:
    """
    Call Ollama chat/generate.

    Returns (model_used, text). On timeout, falls back only to smaller/faster models.
    """
    timeout = int(timeout_sec if timeout_sec is not None else ollama_timeout_sec())
    chain = [model] if model else resolve_model_chain()
    if not chain:
        chain = [str(getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:14b"))]

    num_predict = max(256, int(getattr(config, "OLLAMA_NUM_PREDICT", 1600) or 1600))
    num_ctx = max(2048, int(getattr(config, "OLLAMA_NUM_CTX", 8192) or 8192))
    last_err: Exception | None = None
    tried: list[str] = []
    queue = list(chain)

    while queue:
        pick = queue.pop(0)
        if pick in tried:
            continue
        tried.append(pick)
        options = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }
        use_chat = bool(getattr(config, "OLLAMA_USE_CHAT_API", True))
        body: dict[str, Any] | None = None
        try:
            if use_chat:
                messages: list[dict[str, str]] = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                chat_body: dict[str, Any] = {
                    "model": pick,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "5m",
                    "options": options,
                }
                if json_mode and bool(getattr(config, "OLLAMA_JSON_FORMAT", True)):
                    chat_body["format"] = "json"
                try:
                    raw = _http_post("/api/chat", chat_body, timeout=timeout)
                    body = {
                        "response": (raw.get("message") or {}).get("content", ""),
                        "thinking": raw.get("thinking", ""),
                    }
                except TimeoutError:
                    raise
                except (
                    urllib.error.URLError,
                    urllib.error.HTTPError,
                    OSError,
                    json.JSONDecodeError,
                ) as exc:
                    logger.debug("Ollama chat failed for %s: %s - trying generate", pick, exc)

            if body is None:
                gen_body: dict[str, Any] = {
                    "model": pick,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "5m",
                    "options": options,
                }
                if system:
                    gen_body["system"] = system
                if json_mode and bool(getattr(config, "OLLAMA_JSON_FORMAT", True)):
                    gen_body["format"] = "json"
                body = _http_post("/api/generate", gen_body, timeout=timeout)

            text = _merge_thinking_response(body)
            return pick, text
        except TimeoutError as exc:
            last_err = TimeoutError(_timeout_message(pick, timeout, tried=tried))
            logger.warning("%s", last_err)
            queue = _reorder_after_timeout(queue, pick)
            # Keep full timeout on fallbacks — accuracy mode, do not rush weaker models.
            continue
        except Exception as exc:
            last_err = exc
            logger.warning("Ollama model %s failed: %s", pick, exc)
            continue

    if last_err and _is_timeout_error(last_err):
        raise TimeoutError(str(last_err)) from last_err
    if last_err:
        raise RuntimeError(f"Ollama analysis failed: {last_err}") from last_err
    raise RuntimeError("Ollama analysis failed: no models available")


def ollama_status_message() -> str:
    """Short status for UI when Ollama is down or ready."""
    host = _host()
    if not ollama_available():
        return (
            f"Ollama not running at {host}. Start it with `ollama serve`, "
            f"then pull a model (e.g. `ollama pull qwen2.5-coder:7b`)."
        )
    chain = resolve_model_chain()
    active = chain[0] if chain else getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:7b")
    installed = ollama_installed_models()
    if installed and active not in installed and resolve_model(active, installed) is None:
        return (
            f"Ollama model missing: {active}. "
            f"Run `ollama pull {active}` (host {host})."
        )
    return f"Ollama ready ({host}) - model {active} (timeout {ollama_timeout_sec()}s)"


def classify_ollama_error(exc_or_msg: Any) -> str:
    """Map errors to a stable class: offline | timeout | model_missing | disabled | other."""
    text = str(exc_or_msg or "").lower()
    if "disabled" in text and "ollama" in text:
        return "disabled"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if (
        "model missing" in text
        or "not found" in text
        or ("pull" in text and "model" in text)
        or "no such model" in text
    ):
        return "model_missing"
    if (
        "connection refused" in text
        or "not running" in text
        or "unreachable" in text
        or "failed to establish" in text
        or "name or service not known" in text
        or "nodename nor servname" in text
    ):
        return "offline"
    return "other"


_OLLAMA_HEALTH_CACHE: dict[str, Any] | None = None
_OLLAMA_HEALTH_AT: float = 0.0
_OLLAMA_HEALTH_TTL_SEC = 8.0


def check_ollama_health(*, force: bool = False) -> dict[str, Any]:
    """
    Trading-bot style Ollama probe: reachability, model present, latency.

    Safe to call often — cached briefly unless ``force=True``.
    """
    import time

    global _OLLAMA_HEALTH_CACHE, _OLLAMA_HEALTH_AT
    now = time.monotonic()
    if (
        not force
        and _OLLAMA_HEALTH_CACHE is not None
        and (now - _OLLAMA_HEALTH_AT) < _OLLAMA_HEALTH_TTL_SEC
    ):
        return dict(_OLLAMA_HEALTH_CACHE)

    host = _host()
    primary = str(getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:7b") or "qwen2.5-coder:7b")
    t0 = time.perf_counter()
    reachable = False
    installed: set[str] = set()
    error: str | None = None
    error_class = "offline"
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            reachable = int(getattr(resp, "status", 200) or 200) == 200
            if reachable:
                body = json.loads(resp.read().decode("utf-8"))
                for item in body.get("models") or []:
                    name = str(item.get("name") or "").strip()
                    if name:
                        installed.add(name)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        error_class = classify_ollama_error(exc)
        reachable = False

    latency_ms = int((time.perf_counter() - t0) * 1000)
    resolved = resolve_model(primary, installed) if installed else primary
    model_ok = bool(reachable and (not installed or resolved is not None))
    if reachable and installed and resolved is None:
        error_class = "model_missing"
        error = error or f"model missing: {primary}"
    elif reachable:
        error_class = "ok"
        error = None
    elif error_class == "other":
        error_class = "offline"

    if not reachable:
        banner = "Ollama offline — showing model tickets only"
    elif error_class == "model_missing":
        banner = f"Ollama model missing ({primary}) — showing model tickets only"
    else:
        banner = f"Ollama ready · {resolved or primary} · {latency_ms}ms"

    status: dict[str, Any] = {
        "reachable": reachable,
        "host": host,
        "primary_model": primary,
        "resolved_model": resolved,
        "model_ok": model_ok,
        "installed_count": len(installed),
        "latency_ms": latency_ms,
        "error": error,
        "error_class": error_class,
        "banner": banner,
        "timeout_sec": ollama_timeout_sec(),
    }
    _OLLAMA_HEALTH_CACHE = status
    _OLLAMA_HEALTH_AT = now
    logger.info(
        "Ollama health error_class=%s reachable=%s model=%s latency_ms=%s installed=%s%s",
        error_class,
        reachable,
        resolved or primary,
        latency_ms,
        len(installed),
        f" err={error}" if error else "",
    )
    return dict(status)


def format_ollama_status_line() -> str:
    """One-line Ollama status for banners (mirrors stock-bot)."""
    health = check_ollama_health()
    return str(health.get("banner") or ollama_status_message())
