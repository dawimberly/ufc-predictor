"""Digital analysis of cached fighter photos (display / research, not LightGBM).

Looks at the same UFC.com athlete stills the dashboard already shows. When a
local Ollama *vision* model is installed (llava, qwen2.5vl, llama3.2-vision,
…), both photos are scored for physique / finisher look / size mismatch.

UFC 330 lesson this exists for: two knockout-looking fighters made Over 1.5
look terrible once you *saw* them. That is a research caution — never a
reason to invent HA Blue, and never a LightGBM feature.

Fail-soft: missing photos, no vision model, timeout, or bad JSON → empty
analysis. Over 1.5 is only faded when a cached/vision read actually says both
look like early finishers.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

CACHE_DIR = Path(config.CACHE_DIR) / "weigh_in" / "photo_analysis"

# Tags / name fragments that can accept image bytes in Ollama /api/chat.
_VISION_MARKERS: tuple[str, ...] = (
    "llava",
    "bakllava",
    "moondream",
    "qwen2.5vl",
    "qwen2.5-vl",
    "qwen2-vl",
    "qwen-vl",
    "llama3.2-vision",
    "llama3.2vision",
    "minicpm-v",
    "minicpmv",
    "pixtral",
    "granite3.2-vision",
    "granite-vision",
    "gemma3",  # gemma3 tags are multimodal in Ollama
    "mistral-small3",
    "vision",
)

_FINISHER_TOKENS = frozenset(
    {
        "knockout_artist",
        "ko_artist",
        "finisher",
        "early_finisher",
        "power_puncher",
        "brawler",
        "striker_power",
    }
)
_PHYSQUE_OK = frozenset({"compact", "athletic", "bulky", "lean", "unknown"})
_LOOK_OK = frozenset(
    {
        "knockout_artist",
        "grappler",
        "durable",
        "mixed",
        "unknown",
        "finisher",
        "brawler",
    }
)

_VS_SPLIT = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)


@dataclass
class FighterPhotoRead:
    name: str
    physique: str = "unknown"
    finisher_look: str = "unknown"
    note: str = ""
    confidence: float = 0.0


@dataclass
class PairPhotoAnalysis:
    fighter_1: str
    fighter_2: str
    f1: FighterPhotoRead = field(default_factory=lambda: FighterPhotoRead(name=""))
    f2: FighterPhotoRead = field(default_factory=lambda: FighterPhotoRead(name=""))
    size_mismatch: bool = False
    both_early_finishers: bool = False
    over_15_caution: bool = False
    summary: str = ""
    source: str = "skipped"  # vision | cache | skipped
    model: str = ""
    reason: str = ""

    def line(self) -> str:
        """One ASCII strip line for the dashboard / Ollama."""
        if self.source == "skipped" and not self.summary:
            return ""
        bits: list[str] = []
        if self.over_15_caution or self.both_early_finishers:
            bits.append("both look like early finishers - fade Over 1.5")
        if self.size_mismatch:
            bits.append("size mismatch in stills")
        body = self.summary.strip()
        if body and body not in " | ".join(bits):
            bits.append(body)
        if not bits:
            return ""
        src = ""
        if self.source == "vision" and self.model:
            src = f" [{self.model}]"
        elif self.source == "cache":
            src = " [cached]"
        return "Photos: " + " | ".join(bits) + src


def photo_analysis_enabled() -> bool:
    return bool(getattr(config, "PHOTO_ANALYSIS_ENABLED", True))


def _is_vision_model_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    if not low:
        return False
    return any(marker in low for marker in _VISION_MARKERS)


def resolve_vision_model(installed: set[str] | None = None) -> str | None:
    """Configured OLLAMA_VISION_MODEL, else first installed vision tag."""
    try:
        from src.ollama_client import ollama_installed_models, resolve_model
    except Exception:
        return None

    configured = str(getattr(config, "OLLAMA_VISION_MODEL", "") or "").strip()
    tags = installed if installed is not None else ollama_installed_models()
    if configured:
        if not tags:
            return configured
        hit = resolve_model(configured, tags)
        if hit:
            return hit
        # Explicit config that is not installed → no silent text-model fallback.
        return None
    if not tags:
        return None
    vision = [n for n in sorted(tags) if _is_vision_model_name(n)]
    if not vision:
        return None
    # Prefer smaller/faster vision tags (3b/7b before 13b+).
    def _rank(name: str) -> tuple[int, str]:
        m = re.search(r":(\d+(?:\.\d+)?)\s*b\b", name.lower())
        size = int(float(m.group(1))) if m else 13
        return (size, name)

    return sorted(vision, key=_rank)[0]


def fighter_names_from_record(obj: Any) -> tuple[str, str]:
    """Best-effort (f1, f2) from a ticket, prop row, or fight label."""
    if obj is None:
        return "", ""
    get = obj.get if isinstance(obj, dict) or hasattr(obj, "get") else None
    f1 = f2 = ""
    if get is not None:
        f1 = str(get("fighter_1") or get("fighter1") or "").strip()
        f2 = str(get("fighter_2") or get("fighter2") or "").strip()
        if f1 and f2:
            return f1, f2
        fight = str(get("fight") or get("matchup") or get("label") or "").strip()
    else:
        fight = str(obj).strip()
    if fight:
        parts = _VS_SPLIT.split(fight, maxsplit=1)
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            # Drop trailing " — Over 1.5" style labels
            right = re.split(r"\s+[—\-:]+\s+", right, maxsplit=1)[0].strip()
            if left and right:
                return left, right
    return f1, f2


def _slug_pair(f1: str, f2: str) -> str:
    from src.weigh_in import athlete_slug

    a = athlete_slug(f1) or "unknown"
    b = athlete_slug(f2) or "unknown"
    return f"{a}__{b}"


def _cache_path(f1: str, f2: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_slug_pair(f1, f2)}.json"


def _file_fingerprint(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    try:
        st = path.stat()
        return f"{path.name}:{int(st.st_mtime)}:{st.st_size}"
    except OSError:
        return path.name


def _image_to_b64(path: Path, *, max_side: int = 512) -> str:
    raw = path.read_bytes()
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        raw = buf.getvalue()
    except Exception:
        pass
    return base64.b64encode(raw).decode("ascii")


def _clip_conf(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v > 1.5:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def _norm_physique(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace(" ", "_")
    if text in _PHYSQUE_OK:
        return text
    if "stocky" in text or "thick" in text or "muscular" in text:
        return "bulky"
    if "skinny" in text or "lanky" in text:
        return "lean"
    if "jacked" in text or "athletic" in text:
        return "athletic"
    return "unknown"


def _norm_look(value: Any) -> str:
    text = str(value or "unknown").strip().lower().replace(" ", "_")
    if text in _LOOK_OK:
        if text in {"finisher", "brawler"}:
            return "knockout_artist"
        return text
    if any(tok in text for tok in ("ko", "knockout", "power", "puncher", "finisher")):
        return "knockout_artist"
    if "grappl" in text or "wrestl" in text:
        return "grappler"
    if "durable" in text or "chin" in text:
        return "durable"
    return "unknown"


def _looks_like_finisher(read: FighterPhotoRead) -> bool:
    look = (read.finisher_look or "").lower()
    if look in _FINISHER_TOKENS or look == "knockout_artist":
        return True
    note = (read.note or "").lower()
    return any(tok in note for tok in ("knockout", "finisher", "power puncher", "early finish"))


def _read_from_blob(name: str, blob: Any) -> FighterPhotoRead:
    if not isinstance(blob, dict):
        return FighterPhotoRead(name=name)
    return FighterPhotoRead(
        name=name,
        physique=_norm_physique(blob.get("physique")),
        finisher_look=_norm_look(blob.get("finisher_look") or blob.get("look")),
        note=str(blob.get("note") or blob.get("comment") or "").strip()[:160],
        confidence=_clip_conf(blob.get("confidence")),
    )


def parse_vision_payload(
    raw: Any,
    *,
    fighter_1: str,
    fighter_2: str,
) -> PairPhotoAnalysis:
    """Parse vision JSON into a pair analysis (fail-soft)."""
    data: dict[str, Any] = {}
    if isinstance(raw, dict):
        data = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return PairPhotoAnalysis(
                fighter_1=fighter_1,
                fighter_2=fighter_2,
                source="skipped",
                reason="empty_vision",
            )
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, dict):
                        data = parsed
                except json.JSONDecodeError:
                    data = {}
        if not data:
            return PairPhotoAnalysis(
                fighter_1=fighter_1,
                fighter_2=fighter_2,
                source="skipped",
                reason="unparseable_vision",
            )

    f1 = _read_from_blob(fighter_1, data.get("fighter_1") or data.get("f1"))
    f2 = _read_from_blob(fighter_2, data.get("fighter_2") or data.get("f2"))
    both = bool(data.get("both_early_finishers"))
    if not both:
        both = _looks_like_finisher(f1) and _looks_like_finisher(f2)
    caution = bool(data.get("over_15_caution")) or both
    mismatch = bool(data.get("size_mismatch"))
    summary = str(data.get("summary") or data.get("note") or "").strip()
    summary = " ".join(summary.replace("\n", " ").split())[:220]
    if caution and "over 1.5" not in summary.lower() and "over1.5" not in summary.lower():
        extra = "both stills look like finishers - Over 1.5 is a fade"
        summary = f"{summary} {extra}".strip() if summary else extra
    return PairPhotoAnalysis(
        fighter_1=fighter_1,
        fighter_2=fighter_2,
        f1=f1,
        f2=f2,
        size_mismatch=mismatch,
        both_early_finishers=both,
        over_15_caution=caution,
        summary=summary,
        source="vision",
        reason="",
    )


def _analysis_from_cache_blob(blob: dict[str, Any]) -> PairPhotoAnalysis:
    f1_name = str(blob.get("fighter_1") or "")
    f2_name = str(blob.get("fighter_2") or "")
    f1 = blob.get("f1") if isinstance(blob.get("f1"), dict) else {}
    f2 = blob.get("f2") if isinstance(blob.get("f2"), dict) else {}
    return PairPhotoAnalysis(
        fighter_1=f1_name,
        fighter_2=f2_name,
        f1=_read_from_blob(f1_name, f1),
        f2=_read_from_blob(f2_name, f2),
        size_mismatch=bool(blob.get("size_mismatch")),
        both_early_finishers=bool(blob.get("both_early_finishers")),
        over_15_caution=bool(blob.get("over_15_caution")),
        summary=str(blob.get("summary") or ""),
        source="cache",
        model=str(blob.get("model") or ""),
        reason=str(blob.get("reason") or ""),
    )


def load_cached_analysis(
    fighter_1: Any,
    fighter_2: Any,
    *,
    image_fps: tuple[str, str] | None = None,
) -> PairPhotoAnalysis | None:
    f1, f2 = str(fighter_1 or "").strip(), str(fighter_2 or "").strip()
    if not f1 or not f2:
        return None
    path = _cache_path(f1, f2)
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(blob, dict):
        return None
    if image_fps is not None:
        cached_fp = str(blob.get("image_fp") or "")
        expect = "|".join(image_fps)
        if cached_fp and cached_fp != expect:
            return None
    return _analysis_from_cache_blob(blob)


def _save_cache(
    analysis: PairPhotoAnalysis,
    *,
    image_fp: str,
) -> None:
    path = _cache_path(analysis.fighter_1, analysis.fighter_2)
    payload = {
        "fighter_1": analysis.fighter_1,
        "fighter_2": analysis.fighter_2,
        "f1": asdict(analysis.f1),
        "f2": asdict(analysis.f2),
        "size_mismatch": analysis.size_mismatch,
        "both_early_finishers": analysis.both_early_finishers,
        "over_15_caution": analysis.over_15_caution,
        "summary": analysis.summary,
        "source": analysis.source,
        "model": analysis.model,
        "reason": analysis.reason,
        "image_fp": image_fp,
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("photo analysis cache write failed: %s", exc)


def _vision_prompt(f1: str, f2: str) -> str:
    return (
        f"You are an MMA photo desk. Image 1 is {f1}. Image 2 is {f2}. "
        "Judge only what is visible in the stills (physique, build, whether they "
        "look like early finishers / knockout artists vs grinders). Do not pick a "
        "winner and do not invent records or gyms.\n"
        "Return JSON only:\n"
        "{"
        f'"fighter_1":{{"physique":"compact|athletic|bulky|lean|unknown","finisher_look":"knockout_artist|grappler|durable|mixed|unknown","note":"...","confidence":0.0}},'
        f'"fighter_2":{{"physique":"compact|athletic|bulky|lean|unknown","finisher_look":"knockout_artist|grappler|durable|mixed|unknown","note":"...","confidence":0.0}},'
        '"size_mismatch":false,'
        '"both_early_finishers":false,'
        '"over_15_caution":false,'
        '"summary":"one line"}\n'
        "both_early_finishers=true when BOTH stills look like knockout artists / "
        "early finishers (power, stocky strikers). over_15_caution=true in that "
        "case — Over 1.5 rounds looks like a fade. size_mismatch=true when one "
        "fighter looks clearly bigger in the photos."
    )


def analyze_pair(
    fighter_1: Any,
    fighter_2: Any,
    *,
    fetch_images: bool = False,
    use_vision: bool = True,
    timeout_sec: int | None = None,
) -> PairPhotoAnalysis:
    """Analyze a fight pair. Cache-first. Vision is optional and fail-soft."""
    f1, f2 = str(fighter_1 or "").strip(), str(fighter_2 or "").strip()
    empty = PairPhotoAnalysis(fighter_1=f1, fighter_2=f2, source="skipped", reason="no_names")
    if not photo_analysis_enabled():
        empty.reason = "disabled"
        return empty
    if not f1 or not f2:
        return empty

    try:
        from src.weigh_in import pair_image_paths
    except Exception as exc:
        empty.reason = f"weigh_in:{exc}"
        return empty

    p1, p2 = pair_image_paths(f1, f2, fetch=fetch_images)
    fps = (_file_fingerprint(p1), _file_fingerprint(p2))
    cached = load_cached_analysis(f1, f2, image_fps=fps)
    if cached is not None and cached.source in {"vision", "cache"} and (
        cached.summary or cached.over_15_caution or cached.both_early_finishers
    ):
        return cached

    if p1 is None or p2 is None or not p1.is_file() or not p2.is_file():
        empty.reason = "missing_photos"
        return empty

    if not use_vision:
        if cached is not None:
            return cached
        empty.reason = "cache_only"
        empty.summary = "photos on file - vision not run"
        return empty

    model = resolve_vision_model()
    if not model:
        empty.reason = "no_vision_model"
        empty.summary = (
            "photos on file - pull a vision model (ollama pull llava) "
            "or set OLLAMA_VISION_MODEL"
        )
        return empty

    timeout = int(
        timeout_sec
        if timeout_sec is not None
        else getattr(config, "OLLAMA_VISION_TIMEOUT_SEC", 45) or 45
    )
    timeout = max(8, min(timeout, 120))
    try:
        from src.ollama_client import ollama_chat_images

        images = [_image_to_b64(p1), _image_to_b64(p2)]
        used, text = ollama_chat_images(
            _vision_prompt(f1, f2),
            images,
            model=model,
            timeout_sec=timeout,
            json_mode=True,
        )
    except Exception as exc:
        logger.info("photo vision failed for %s vs %s: %s", f1, f2, exc)
        empty.reason = "vision_error"
        empty.model = model
        empty.summary = f"vision failed ({type(exc).__name__})"
        return empty

    analysis = parse_vision_payload(text, fighter_1=f1, fighter_2=f2)
    analysis.model = used or model
    analysis.source = "vision"
    _save_cache(analysis, image_fp="|".join(fps))
    return analysis


def format_photo_analysis_line(
    fighter_1: Any,
    fighter_2: Any,
    *,
    fetch_vision: bool = False,
) -> str:
    """Cache-only line for the context strip (never blocks the UI on Ollama)."""
    analysis = analyze_pair(
        fighter_1,
        fighter_2,
        fetch_images=False,
        use_vision=fetch_vision,
    )
    return analysis.line()


def photo_over_15_blocks(prop: dict[str, Any] | None) -> bool:
    """True when cached vision says both fighters look like early finishers.

    Never calls Ollama. Missing cache → False (do not invent a fade).
    """
    if not isinstance(prop, dict):
        return False
    if not photo_analysis_enabled():
        return False
    if str(prop.get("prop_key") or "").strip().lower() != "over_1_5_rounds":
        label = str(prop.get("label") or prop.get("prop_short") or prop.get("market") or "")
        if "over 1.5" not in label.lower():
            return False
    if prop.get("photo_over_15_caution") is True:
        return True
    f1, f2 = fighter_names_from_record(prop)
    if not f1 or not f2:
        return False
    cached = load_cached_analysis(f1, f2)
    return bool(cached and (cached.over_15_caution or cached.both_early_finishers))


def attach_photo_notes(ticket: dict[str, Any]) -> dict[str, Any]:
    """Stamp cache-only photo fields onto a ticket (no vision call)."""
    if not isinstance(ticket, dict):
        return ticket
    f1, f2 = fighter_names_from_record(ticket)
    if not f1 or not f2:
        return ticket
    cached = load_cached_analysis(f1, f2)
    if cached is None:
        return ticket
    ticket["photo_note"] = cached.line()
    ticket["photo_over_15_caution"] = bool(
        cached.over_15_caution or cached.both_early_finishers
    )
    ticket["photo_size_mismatch"] = bool(cached.size_mismatch)
    return ticket


def card_photo_notes(tickets: list[dict[str, Any]] | None) -> str:
    """Compact photo block for the Ollama narrate prompt."""
    lines: list[str] = []
    seen: set[str] = set()
    for t in tickets or []:
        attach_photo_notes(t)
        note = str(t.get("photo_note") or "").strip()
        fight = str(t.get("fight") or t.get("side") or "").strip()
        key = hashlib.sha1(f"{fight}|{note}".encode("utf-8")).hexdigest()[:12]
        if note and key not in seen:
            seen.add(key)
            label = fight or "fight"
            lines.append(f"- {label}: {note}")
        elif t.get("photo_over_15_caution") and key not in seen:
            seen.add(key)
            lines.append(f"- {fight or 'fight'}: Photos: fade Over 1.5 (both finishers)")
    return "\n".join(lines[:8])
