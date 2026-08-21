"""Real Gemini adapters (google-genai) — imports are LAZY so the package is optional.

Two separate Gemini calls per plan, per PLAN.md §2:
1. Research WITH Google Search grounding tools enabled — returns candidate topics
   with source URLs/titles and access time.
2. A SECOND call WITHOUT tools producing structured JSON for VideoPlan. The
   grounding + structured-output combination is deliberately avoided (Preview).

All parsing lives in pure module functions so contract tests run offline. A
pre-built client can be injected for tests; nothing here opens the network at
import time.
"""

import base64
import json
import re
from datetime import UTC, datetime
from typing import Any

from app.ai.base import (
    OP_FAILED,
    OP_RUNNING,
    OP_SUCCEEDED,
    ImageResult,
    SourceInfo,
    TopicCandidate,
    VideoOperation,
)
from app.config import get_model_config, get_settings
from app.errors import UpstreamError

# ---------------------------------------------------------------------------
# Pure helpers (offline-testable)
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```$")


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response (code fences tolerated)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", cleaned))
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise UpstreamError("model response contained no JSON object", retryable=True)
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise UpstreamError(f"model returned invalid JSON: {exc}", retryable=True) from exc
    if not isinstance(data, dict):
        raise UpstreamError("model JSON must be an object", retryable=True)
    return data


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _maybe_score(value: Any) -> float | None:
    if isinstance(value, int | float):
        return min(1.0, max(0.0, float(value)))
    return None


def parse_topic_candidates(data: dict, accessed_at: datetime) -> list[TopicCandidate]:
    """Turn the research call's JSON payload into TopicCandidate objects."""
    out: list[TopicCandidate] = []
    for raw in data.get("topics") or []:
        if not isinstance(raw, dict) or not raw.get("title"):
            continue
        sources: list[SourceInfo] = []
        for src in raw.get("sources") or []:
            if not isinstance(src, dict) or not src.get("url"):
                continue
            sources.append(
                SourceInfo(
                    url=str(src["url"]),
                    title=str(src.get("title", "")),
                    publisher=str(src.get("publisher", "")),
                    is_official=bool(src.get("is_official", False)),
                    accessed_at=accessed_at,
                )
            )
        out.append(
            TopicCandidate(
                title=str(raw["title"]),
                summary=str(raw.get("summary", "")),
                category=str(raw.get("category", "")),
                published_at=_parse_datetime(raw.get("published_at")),
                sources=sources,
                audience_fit=_maybe_score(raw.get("audience_fit")),
                visual_potential=_maybe_score(raw.get("visual_potential")),
                freshness=_maybe_score(raw.get("freshness")),
                cross_verification=_maybe_score(raw.get("cross_verification")),
            )
        )
    return out


def build_research_prompt(brief: str, category: str, window_hours: int) -> str:
    return (
        "You are researching emerging AI/technology topics for a short-video team.\n"
        f"Brief: {brief}\nCategory: {category or 'ai/technology'}\n"
        f"Only include topics first reported within the last {window_hours} hours.\n"
        "Use Google Search to verify every topic across at least 2 independent "
        "publishers and prefer at least one official source (vendor blog, paper, "
        "regulator). Return ONLY a JSON object shaped as:\n"
        '{"topics": [{"title": str, "summary": str, "category": str, '
        '"published_at": "ISO-8601 UTC", "audience_fit": 0..1, '
        '"visual_potential": 0..1, "sources": [{"url": str, "title": str, '
        '"publisher": str, "is_official": bool}]}]}\n'
        "Return 3 to 6 topics, no prose outside the JSON."
    )


def build_plan_prompt(topic: str, sources: list[SourceInfo], brief: str) -> str:
    source_lines = "\n".join(
        f"- source_id=src{i + 1} url={s.url} title={s.title!r} official={s.is_official}"
        for i, s in enumerate(sources)
    )
    return (
        "Write a VideoPlan v1 JSON for a 38.8 s vertical (9:16) short video.\n"
        f"Topic: {topic}\nBrief: {brief}\nSources (cite by source_id):\n{source_lines}\n"
        "Hard requirements:\n"
        "- schema_version '1.0'; exactly 5 scenes, each duration_seconds 8.0, indices 0..4.\n"
        "- Each scene: keyframe_prompt_en, visual_prompt_en, negative_prompt_en, "
        "continuity_note, fact_ids.\n"
        "- facts: every factual claim gets a fact_id mapped to source_ids from the list above.\n"
        "- locales 'vi' and 'en': narration (5 strings, each speakable in <= 7.6 s), "
        "on_screen_text (5 strings <= 80 chars), title, description, hashtags.\n"
        "- disclosure.synthetic_media true; no text/logos/lip-sync in visual prompts.\n"
        "Return ONLY the JSON object."
    )


def build_shorten_prompt(plan: dict, issues: list[dict]) -> str:
    flagged = "\n".join(
        f"- locale={i.get('locale')} scene={i.get('scene_index')}: {i.get('message', '')}"
        for i in issues
    )
    return (
        "The following narrations exceed the 7.6 s per-scene voice-over budget:\n"
        f"{flagged}\n"
        "Shorten ONLY those narrations so each is comfortably speakable within "
        "7.6 seconds, preserving meaning and facts. Keep every other field of the "
        "plan EXACTLY unchanged. Return the FULL updated VideoPlan JSON only.\n"
        f"Current plan JSON:\n{json.dumps(plan, ensure_ascii=False)}"
    )


# ---------------------------------------------------------------------------
# Adapters (lazy google-genai)
# ---------------------------------------------------------------------------


class _GenAIAdapter:
    """Shared lazy client handling; a client can be injected for tests."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_settings().gemini_api_key
        self._model_id = model_id or self._default_model()
        self._client = client

    def _default_model(self) -> str:
        return get_model_config().gemini_text_model

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai  # noqa: PLC0415 — lazy: optional dependency
            except ImportError as exc:
                raise UpstreamError(
                    "google-genai is not installed; install the 'google' extra "
                    "or inject a client",
                    retryable=False,
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client


class GeminiResearchProvider(_GenAIAdapter):
    """Research call WITH Google Search grounding tools (call 1 of 2)."""

    def research(self, brief: str, category: str, window_hours: int) -> list[TopicCandidate]:
        client = self._get_client()
        accessed_at = datetime.now(UTC)
        response = client.models.generate_content(
            model=self._model_id,
            contents=build_research_prompt(brief, category, window_hours),
            config={"tools": [{"google_search": {}}], "temperature": 0.4},
        )
        data = extract_json(getattr(response, "text", "") or "")
        candidates = parse_topic_candidates(data, accessed_at)
        grounding = _grounding_sources(response, accessed_at)
        for candidate in candidates:
            if not candidate.sources and grounding:
                candidate.sources = list(grounding)
        return candidates


def _grounding_sources(response: Any, accessed_at: datetime) -> list[SourceInfo]:
    """Harvest grounding-metadata web chunks (URL + title) from a genai response."""
    sources: list[SourceInfo] = []
    for candidate in getattr(response, "candidates", None) or []:
        metadata = getattr(candidate, "grounding_metadata", None)
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            uri = getattr(web, "uri", None)
            if uri:
                sources.append(
                    SourceInfo(
                        url=str(uri),
                        title=str(getattr(web, "title", "") or ""),
                        accessed_at=accessed_at,
                    )
                )
    return sources


class GeminiScriptProvider(_GenAIAdapter):
    """Structured-output call WITHOUT tools (call 2 of 2) plus the shorten pass."""

    _JSON_CONFIG = {"response_mime_type": "application/json", "temperature": 0.6}

    def generate_plan(self, topic: str, sources: list[SourceInfo], brief: str) -> dict:
        client = self._get_client()
        response = client.models.generate_content(
            model=self._model_id,
            contents=build_plan_prompt(topic, sources, brief),
            config=dict(self._JSON_CONFIG),
        )
        return extract_json(getattr(response, "text", "") or "")

    def shorten_narrations(self, plan: dict, issues: list[dict]) -> dict:
        client = self._get_client()
        response = client.models.generate_content(
            model=self._model_id,
            contents=build_shorten_prompt(plan, issues),
            config=dict(self._JSON_CONFIG),
        )
        return extract_json(getattr(response, "text", "") or "")


class GeminiImageProvider(_GenAIAdapter):
    """Keyframe/style-board images via gemini-3.1-flash-image."""

    def _default_model(self) -> str:
        return get_model_config().gemini_image_model

    def generate_image(
        self, prompt: str, *, model_id: str | None = None, negative_prompt: str = ""
    ) -> ImageResult:
        client = self._get_client()
        model = model_id or self._model_id
        contents = prompt if not negative_prompt else f"{prompt}\n\nAvoid: {negative_prompt}"
        response = client.models.generate_content(model=model, contents=contents)
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None)
                if data:
                    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                    mime = str(getattr(inline, "mime_type", "") or "image/png")
                    return ImageResult(image_bytes=raw, model_id=model, mime_type=mime)
        raise UpstreamError("image model returned no image data", retryable=True)


class GenAIVideoProvider(_GenAIAdapter):
    """Veo long-running operations via google-genai; downloads bytes promptly on success."""

    def _default_model(self) -> str:
        return get_model_config().veo_model_lite

    def submit(
        self,
        *,
        prompt: str,
        model_id: str,
        duration_seconds: float,
        keyframe_bytes: bytes | None = None,
    ) -> str:
        client = self._get_client()
        from google.genai import types  # noqa: PLC0415 — lazy: optional dependency

        kwargs: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "config": types.GenerateVideosConfig(aspect_ratio="9:16", resolution="720p"),
        }
        if keyframe_bytes is not None:
            kwargs["image"] = types.Image(image_bytes=keyframe_bytes, mime_type="image/png")
        operation = client.models.generate_videos(**kwargs)
        name = getattr(operation, "name", None)
        if not name:
            raise UpstreamError("veo submit returned no operation name", retryable=True)
        return str(name)

    def poll(self, operation_name: str) -> VideoOperation:
        client = self._get_client()
        from google.genai import types  # noqa: PLC0415 — lazy: optional dependency

        handle = types.GenerateVideosOperation(name=operation_name)
        operation = client.operations.get(handle)
        if not getattr(operation, "done", False):
            return VideoOperation(operation_name, OP_RUNNING)
        error = getattr(operation, "error", None)
        if error:
            return VideoOperation(operation_name, OP_FAILED, error=str(error))
        response = getattr(operation, "response", None) or getattr(operation, "result", None)
        videos = getattr(response, "generated_videos", None) or []
        if not videos:
            return VideoOperation(
                operation_name, OP_FAILED, error="operation finished without video output"
            )
        video = videos[0].video
        data = getattr(video, "video_bytes", None)
        if data is None:
            # Google keeps generated files only ~2 days; download promptly.
            data = client.files.download(file=video)
        return VideoOperation(operation_name, OP_SUCCEEDED, video_bytes=bytes(data))
