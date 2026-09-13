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
    """Primary first; append smaller installed fallbacks (esp. after heavy primary)."""
    primary = str(
        getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:7b") or "qwen2.5-coder:7b"
    ).strip()
    fallbacks = [
        m.strip()
        for m in str(getattr(config, "OLLAMA_FALLBACK_MODELS", "") or "").split(",")
        if m.strip()
    ]
    # Always prefer a fast path when the primary is heavy and smaller models exist.
    if _is_heavy_model(primary) or _model_param_b(primary) >= 14:
        for pref in _FAST_MODEL_PREFS:
            if pref not in fallbacks and pref != primary:
                fallbacks.append(pref)

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
    # If primary missing but a fast model is installed, put fast first.
    if primary_resolved not in installed:
        fast = _suggested_fast_models(installed)
        if fast:
            return [fast[0]] + [m for m in out if m != fast[0]]
    return out


def _analysis_num_predict(model: str, *, json_mode: bool) -> int:
    """Cap generation length — short JSON reasons do not need 1600 tokens."""
    configured = max(128, int(getattr(config, "OLLAMA_NUM_PREDICT", 512) or 512))
    if not json_mode:
        return configured
    # Card narrate JSON is tiny; heavy models waste the whole timeout padding tokens.
    if _is_heavy_model(model) or _model_param_b(model) >= 14:
        return min(configured, 420)
    return min(configured, 560)


def _attempt_timeout_sec(model: str, full_timeout: int, *, has_fallback: bool) -> int:
    """Soft-cap heavy first attempts so we fail over (or return HA slip) sooner."""
    full_timeout = max(30, int(full_timeout))
    if not has_fallback:
        return full_timeout
    if _is_heavy_model(model) or _model_param_b(model) >= 14:
        # Don't burn the whole budget on a 14b that will never finish on CPU.
        return min(full_timeout, max(90, int(full_timeout * 0.35)))
    return full_timeout


def _fallback_installed_model(installed: set[str]) -> str:
    """Pick a usable installed model (prefer larger quality models)."""
    if not installed:
        return str(getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:14b") or "qwen2.5-coder:14b")
    return sorted(installed, key=lambda n: (_model_param_b(n), n), reverse=True)[0]


def list_model_choices() -> list[str]:
    """Models for the dashboard picker: active primary, then installed (no unpulled stubs)."""
    installed = ollama_installed_models()
    primary = str(getattr(config, "OLLAMA_MODEL", "") or "").strip()
    choices: list[str] = []
    # Only keep primary in the list when it is actually installed (or nothing is listed yet).
    if primary and (not installed or primary in installed or resolve_model(primary, installed)):
        resolved_primary = resolve_model(primary, installed) if installed else primary
        if resolved_primary:
            choices.append(resolved_primary)
    for name in _suggested_fast_models(set(installed) if installed else None):
        if name not in choices and (not installed or name in installed):
            choices.append(name)
    for name in sorted(installed, key=lambda n: (_model_param_b(n), n)):
        if name not in choices:
            choices.append(name)
    if not choices:
        choices = [primary or "qwen2.5-coder:7b"]
    return choices


def model_speed_hint(model: str) -> str:
    """Short UI hint for the selected model."""
    name = str(model or "").strip()
    if not name:
        return "pick a model"
    if _is_heavy_model(name) or _model_param_b(name) >= 14:
        return "heavier · better narrative, slower"
    if _model_param_b(name) <= 3:
        return "fast · lighter narrative"
    return "balanced speed / quality"


_LAST_MODEL_WARN = ""


def set_active_model(model: str) -> str:
    """Update runtime config model (session); fall back if the tag is not installed."""
    global _LAST_MODEL_WARN
    _LAST_MODEL_WARN = ""
    name = str(model or "").strip() or str(
        getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:7b")
    )
    installed = ollama_installed_models()
    if not installed:
        config.OLLAMA_MODEL = name
        return name

    resolved = resolve_model(name, installed)
    if resolved:
        config.OLLAMA_MODEL = resolved
        return resolved

    fallback = _fallback_installed_model(installed)
    _LAST_MODEL_WARN = (
        f"{name} is not installed. Using {fallback} instead. "
        f"To install: ollama pull {name}"
    )
    logger.warning("%s", _LAST_MODEL_WARN)
    config.OLLAMA_MODEL = fallback
    return fallback


def last_model_warn() -> str:
    return str(_LAST_MODEL_WARN or "")


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


def ollama_chat_images(
    prompt: str,
    images_b64: list[str],
    *,
    model: str,
    timeout_sec: int = 45,
    system: str | None = None,
    json_mode: bool = True,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """One-shot /api/chat with base64 images. No text-model fallback (they cannot see)."""
    pick = str(model or "").strip()
    if not pick:
        raise RuntimeError("Ollama vision model not set")
    if not images_b64:
        raise RuntimeError("Ollama vision call missing images")
    timeout = max(8, int(timeout_sec or 45))
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": prompt,
            "images": list(images_b64),
        }
    )
    body: dict[str, Any] = {
        "model": pick,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "num_predict": min(320, max(128, int(getattr(config, "OLLAMA_NUM_PREDICT", 512) or 512))),
            "num_ctx": min(4096, max(2048, int(getattr(config, "OLLAMA_NUM_CTX", 4096) or 4096))),
        },
    }
    if json_mode and bool(getattr(config, "OLLAMA_JSON_FORMAT", True)):
        body["format"] = "json"
    raw = _http_post("/api/chat", body, timeout=timeout)
    merged = {
        "response": (raw.get("message") or {}).get("content", ""),
        "thinking": raw.get("thinking", ""),
    }
    return pick, _merge_thinking_response(merged)


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
    Heavy models get a soft first-attempt timeout so CPU hosts fail over sooner.
    """
    timeout = int(timeout_sec if timeout_sec is not None else ollama_timeout_sec())
    chain = [model] if model else resolve_model_chain()
    if not chain:
        chain = [str(getattr(config, "OLLAMA_MODEL", "qwen2.5-coder:7b"))]

    # Prefetch installed fast models into the queue after a heavy primary.
    installed = ollama_installed_models()
    if chain and (_is_heavy_model(chain[0]) or _model_param_b(chain[0]) >= 14):
        for pref in _suggested_fast_models(installed):
            if pref not in chain:
                chain.append(pref)

    num_ctx = max(2048, int(getattr(config, "OLLAMA_NUM_CTX", 4096) or 4096))
    # Smaller context for heavy JSON narrate — less KV cache thrash on CPU.
    if chain and (_is_heavy_model(chain[0]) or _model_param_b(chain[0]) >= 14):
        num_ctx = min(num_ctx, 4096)

    last_err: Exception | None = None
    tried: list[str] = []
    queue = list(chain)

    while queue:
        pick = queue.pop(0)
        if pick in tried:
            continue
        tried.append(pick)
        num_predict = _analysis_num_predict(pick, json_mode=json_mode)
        attempt_timeout = _attempt_timeout_sec(
            pick, timeout, has_fallback=bool(queue) or bool(_suggested_fast_models(installed))
        )
        # If nothing smaller is installed, give the heavy model the full budget.
        if not any(_model_param_b(m) < _model_param_b(pick) for m in (*queue, *_suggested_fast_models(installed))):
            attempt_timeout = timeout

        options = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }
        use_chat = bool(getattr(config, "OLLAMA_USE_CHAT_API", True))
        body: dict[str, Any] | None = None
        try:
            logger.info(
                "Ollama attempt model=%s timeout=%ss num_predict=%s num_ctx=%s",
                pick,
                attempt_timeout,
                num_predict,
                num_ctx,
            )
            if use_chat:
                messages: list[dict[str, str]] = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                chat_body: dict[str, Any] = {
                    "model": pick,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": options,
                }
                if json_mode and bool(getattr(config, "OLLAMA_JSON_FORMAT", True)):
                    chat_body["format"] = "json"
                try:
                    raw = _http_post("/api/chat", chat_body, timeout=attempt_timeout)
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
                    "keep_alive": "10m",
                    "options": options,
                }
                if system:
                    gen_body["system"] = system
                if json_mode and bool(getattr(config, "OLLAMA_JSON_FORMAT", True)):
                    gen_body["format"] = "json"
                body = _http_post("/api/generate", gen_body, timeout=attempt_timeout)

            text = _merge_thinking_response(body)
            return pick, text
        except TimeoutError as exc:
            last_err = TimeoutError(_timeout_message(pick, attempt_timeout, tried=tried))
            logger.warning("%s", last_err)
            queue = _reorder_after_timeout(queue, pick)
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
    if reachable and installed and resolved is None:
        # Soft-recover: don't hard-fail analysis when another model is available.
        resolved = _fallback_installed_model(installed)
        config.OLLAMA_MODEL = resolved
        error_class = "ok"
        error = None
        banner = (
            f"Ollama ready · {resolved} · {latency_ms}ms "
            f"(requested {primary} missing — run: ollama pull {primary})"
        )
        model_ok = True
    else:
        model_ok = bool(reachable and (not installed or resolved is not None))
        if reachable:
            error_class = "ok"
            error = None
            banner = f"Ollama ready · {resolved or primary} · {latency_ms}ms"
        else:
            if error_class == "other":
                error_class = "offline"
            banner = "Ollama offline — showing model tickets only"

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
