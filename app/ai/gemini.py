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
import logging
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
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
from app.ai.topics import registrable_domain
from app.config import get_model_config, get_settings
from app.errors import UpstreamError, ValidationFailed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (offline-testable)
# ---------------------------------------------------------------------------

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```$")
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "ConnectError",
        "ConnectionClosed",
        "ConnectionResetError",
        "DeadlineExceeded",
        "NetworkError",
        "ReadError",
        "ReadTimeout",
        "ServerDisconnectedError",
        "TimeoutError",
        "TransportError",
    }
)


def _is_transient_transport_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if type(current).__name__ in _TRANSIENT_EXCEPTION_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def _call_genai(operation: str, call: Callable[[], Any]) -> Any:
    """Normalize SDK/network failures without leaking provider payloads or keys."""
    try:
        return call()
    except UpstreamError:
        raise
    except Exception as exc:
        status_code: int | None = None
        for attribute in ("status_code", "code"):
            value = getattr(exc, attribute, None)
            if isinstance(value, int):
                status_code = value
                break
        retryable = (
            status_code in {408, 429}
            or (status_code is not None and status_code >= 500)
            or _is_transient_transport_error(exc)
        )
        details: dict[str, Any] = {"provider": "gemini", "operation": operation}
        if status_code is not None:
            details["status_code"] = status_code
        raise UpstreamError(
            f"Gemini {operation} request failed",
            retryable=retryable,
            details=details,
        ) from exc


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
        "You are an elite viral short-form video creator and cinematic director creating a top-performing 38.8s vertical (9:16) video for TikTok, YouTube Shorts, and Reels.\n\n"
        f"Topic: {topic}\nBrief: {brief}\nSources (cite by source_id):\n{source_lines}\n\n"
        "CREATIVE STORYTELLING & SCRIPT STRUCTURE (5 SCENES, 8.0s each):\n"
        "- Scene 0 (The Irresistible Hook): Explosive, curious, or shocking opening statement/question that immediately stops scrolling in the first 2 seconds.\n"
        "- Scene 1 (Context & Pain Point): Vividly sets the stakes and explains why this development is shaking up the industry or everyday life.\n"
        "- Scene 2 (The Core Revelation / Deep Insight): Unpacks the essential breakthrough or fact with cited evidence and clear, compelling explanation.\n"
        "- Scene 3 (Real-world Impact & Transformation): Shows practical implications, future outlook, or direct consequences for people/developers/users.\n"
        "- Scene 4 (Memorable Climax & Call-to-Action): Delivers a powerful concluding takeaway and prompts high engagement/discussion in the comments.\n\n"
        "100% VISUAL-TO-CONTENT ALIGNMENT RULES FOR KEYFRAMES & VISUAL PROMPTS:\n"
        "- For EACH scene, `keyframe_prompt_en` and `visual_prompt_en` MUST DIRECTLY AND SPECIFICALLY depict the exact physical subject, action, tools, setting, and emotion discussed in that scene's narration.\n"
        "- Be highly concrete and descriptive: specify the exact person/subject, action taking place, environment (e.g. high-tech laboratory, modern bustling office, sunlit studio, outdoor datacenter), camera angle (e.g. macro close-up, dynamic low angle, cinematic eye-level medium shot), and lighting (e.g. volumetric neon, warm golden hour, dramatic rim light).\n"
        "- Never use generic or vague placeholders like 'abstract background' or 'concept art'. Every prompt must describe a realistic, vivid photographic scene.\n"
        "- START each keyframe_prompt_en with the concrete searchable subject and its action (nouns a stock-photo search would match), THEN add style details. Example good start: 'a robotic hand assembling a glowing microchip in a dark lab' — not 'the future of computation'.\n\n"
        "VISUAL IDENTITY — 5 SCENES MUST READ AS ONE FILM (fill style_guide):\n"
        "- style_guide.palette: 2-3 named colors that carry the topic's emotion; every scene's prompts must reuse this palette.\n"
        "- style_guide.mood + style_guide.camera: one cinematic treatment (e.g. 'dark futuristic tech, neon cyan accents, slow push-ins and subtle parallax') applied to ALL scenes.\n"
        "- style_guide.consistency_notes: what NEVER changes across scenes — the same recurring hero subject or location, same lighting grade, same color temperature — so cuts feel intentional, not like 5 unrelated stock clips.\n"
        "- Every scene composition: subject centered in the middle third, high contrast, vibrant but consistent colors, shallow depth of field, clean uncluttered space at the very top and bottom of the 9:16 frame (captions and platform UI live there).\n"
        "- on_screen_text: max 6 punchy words per scene — a keyword, number, or short phrase that COMPLEMENTS the narration, never a subtitle repeating it.\n\n"
        "HARD SCHEMA REQUIREMENTS:\n"
        "- schema_version '1.0'; exactly 5 scenes, each duration_seconds 8.0, indices 0..4.\n"
        "- Each scene object: keyframe_prompt_en, visual_prompt_en, negative_prompt_en, continuity_note, fact_ids (list of fact_ids used in this scene).\n"
        "- facts: array of objects with fact_id (e.g. 'fact1', 'fact2'), claim (truthful claim), and source_ids (list of source_ids cited from above, e.g. ['src1']).\n"
        "- locales 'vi' and 'en':\n"
        "  * narration: array of 5 strings (one per scene). Natural, conversational, rhythmic storyteller tone. Each string MUST be comfortably speakable within <= 7.6 seconds (~20-25 words in Vietnamese, ~18-24 words in English).\n"
        "  * on_screen_text: array of 5 punchy subtitle/caption highlight strings (<= 80 chars each).\n"
        "  * title: high-CTR engaging title (<= 90 chars).\n"
        "  * description: engaging summary with context.\n"
        "  * hashtags: 3-6 trending, relevant hashtags.\n"
        "- disclosure.synthetic_media: true; no text/logos/watermarks in visual prompts.\n\n"
        "Return ONLY the valid JSON object."
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


def build_repair_prompt(plan: dict, issues: list[dict]) -> str:
    """Ask the model to fix source/fact binding errors it made in the first pass.

    Only fact <-> source integrity errors belong here; narration length has its
    own shorten loop. The model must return the FULL plan with a complete
    ``facts`` array whose ``source_ids`` cite the sources already in the plan.
    """
    flagged = "\n".join(f"- {i.get('code')}: {i.get('message', '')}" for i in issues)
    return (
        "This VideoPlan has fact/source binding errors:\n"
        f"{flagged}\n"
        "Fix them by returning the FULL updated VideoPlan JSON with:\n"
        "- a complete 'facts' array; every fact_id referenced by any scene must "
        "appear there with a truthful claim and source_ids citing ONLY the "
        "source_id values already present in the plan's 'sources' array;\n"
        "- no new sources, no removed scenes, narrations and locales unchanged.\n"
        "Never invent a fact that the cited sources do not support. Return ONLY "
        "the corrected JSON object.\n"
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
        fallback_model_ids: Sequence[str] | None = None,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else get_settings().gemini_api_key
        self._model_id = model_id or self._default_model()
        fallbacks = (
            tuple(fallback_model_ids)
            if fallback_model_ids is not None
            else self._default_fallback_models()
        )
        self._model_ids = tuple(
            dict.fromkeys(
                candidate.strip()
                for candidate in (self._model_id, *fallbacks)
                if candidate.strip()
            )
        )
        self._active_model_index = 0
        self._client = client
        self._injected_client = client is not None

    def _default_model(self) -> str:
        return get_model_config().gemini_text_model

    def _default_fallback_models(self) -> tuple[str, ...]:
        return ()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai  # noqa: PLC0415 — lazy: optional dependency
            except ImportError as exc:
                raise UpstreamError(
                    "google-genai is not installed; install the 'google' extra or inject a client",
                    retryable=False,
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
        return self._client


class _GeminiTextAdapter(_GenAIAdapter):
    """Gemini text calls with ordered, quota-only model fallback."""

    def _default_fallback_models(self) -> tuple[str, ...]:
        return get_model_config().gemini_text_fallback_models

    def _call_text_model(
        self,
        operation: str,
        call: Callable[[str], Any],
    ) -> Any:
        """Try lower text models only after an HTTP 429 from the active model."""
        for index in range(self._active_model_index, len(self._model_ids)):
            model_id = self._model_ids[index]
            try:
                response = _call_genai(operation, partial(call, model_id))
            except UpstreamError as exc:
                is_quota_error = exc.details.get("status_code") == 429
                has_fallback = index + 1 < len(self._model_ids)
                if not is_quota_error or not has_fallback:
                    raise
                logger.warning(
                    "Gemini %s quota exhausted for %s; falling back to %s",
                    operation,
                    model_id,
                    self._model_ids[index + 1],
                )
                continue
            self._active_model_index = index
            return response
        raise RuntimeError("Gemini text model chain is empty")  # pragma: no cover


class GeminiResearchProvider(_GeminiTextAdapter):
    """Research call WITH Google Search grounding tools (call 1 of 2), with direct fallback."""

    def research(self, brief: str, category: str, window_hours: int) -> list[TopicCandidate]:
        client = self._get_client()
        accessed_at = datetime.now(UTC)
        try:
            response = self._call_text_model(
                "research",
                lambda model_id: client.models.generate_content(
                    model=model_id,
                    contents=build_research_prompt(brief, category, window_hours),
                    config={"tools": [{"google_search": {}}]},
                ),
            )
            data = extract_json(getattr(response, "text", "") or "")
            candidates = parse_topic_candidates(data, accessed_at)
            grounding = _grounding_sources(response, accessed_at)
            if grounding:
                for candidate in candidates:
                    candidate.sources = _verified_candidate_sources(candidate.sources, grounding)
            return candidates
        except UpstreamError as exc:
            if not self._injected_client and (exc.details.get("status_code") in {408, 429} or exc.retryable):
                try:
                    response = self._call_text_model(
                        "research_fallback",
                        lambda model_id: client.models.generate_content(
                            model=model_id,
                            contents=build_research_prompt(brief, category, window_hours),
                        ),
                    )
                    data = extract_json(getattr(response, "text", "") or "")
                    return parse_topic_candidates(data, accessed_at)
                except Exception:
                    settings = get_settings()
                    if settings.groq_api_key:
                        from app.ai.groq_providers import GroqScriptProvider

                        groq_p = GroqScriptProvider(
                            api_key=settings.groq_api_key,
                            model_config=get_model_config(),
                        )
                        data = groq_p._chat_json(
                            build_research_prompt(brief, category, window_hours),
                            note="research topics fallback",
                        )
                        return parse_topic_candidates(data, accessed_at)
                    raise
            raise


def _verified_candidate_sources(
    claimed: list[SourceInfo], grounded: list[SourceInfo]
) -> list[SourceInfo]:
    """Keep only URLs whose URL/domain actually appears in grounding metadata."""
    if not grounded:
        return []
    grounded_urls = {source.url.rstrip("/") for source in grounded}
    grounded_domains = {registrable_domain(source.url) for source in grounded}
    verified: list[SourceInfo] = []
    seen: set[str] = set()
    for source in claimed:
        url = source.url.rstrip("/")
        if url not in grounded_urls and registrable_domain(source.url) not in grounded_domains:
            continue
        if url in seen:
            continue
        seen.add(url)
        verified.append(source)
    return verified


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


class GeminiScriptProvider(_GeminiTextAdapter):
    """Structured-output call WITHOUT tools (call 2 of 2) plus the shorten pass."""

    # Gemini 3.6+ rejects the legacy sampling parameters (temperature/top_p/top_k).
    # JSON mode is sufficient here and keeps this adapter compatible with 3.7 Flash.
    _JSON_CONFIG = {"response_mime_type": "application/json"}

    def generate_plan(self, topic: str, sources: list[SourceInfo], brief: str) -> dict:
        try:
            client = self._get_client()
            response = self._call_text_model(
                "script generation",
                lambda model_id: client.models.generate_content(
                    model=model_id,
                    contents=build_plan_prompt(topic, sources, brief),
                    config=dict(self._JSON_CONFIG),
                ),
            )
            return extract_json(getattr(response, "text", "") or "")
        except UpstreamError as err:
            if self._injected_client:
                raise
            logger.warning("Gemini generate_plan failed with %s; falling back to Groq", err)
            from app.ai.groq_providers import GroqScriptProvider

            groq = GroqScriptProvider()
            return groq.generate_plan(topic, sources, brief)

    def shorten_narrations(self, plan: dict, issues: list[dict]) -> dict:
        try:
            client = self._get_client()
            response = self._call_text_model(
                "narration shortening",
                lambda model_id: client.models.generate_content(
                    model=model_id,
                    contents=build_shorten_prompt(plan, issues),
                    config=dict(self._JSON_CONFIG),
                ),
            )
            return extract_json(getattr(response, "text", "") or "")
        except UpstreamError as err:
            if self._injected_client:
                raise
            logger.warning("Gemini shorten_narrations failed with %s; falling back to Groq", err)
            from app.ai.groq_providers import GroqScriptProvider

            groq = GroqScriptProvider()
            return groq.shorten_narrations(plan, issues)

    def repair_plan(self, plan: dict, issues: list[dict]) -> dict:
        """Bounded second pass fixing fact/source binding errors (adds facts)."""
        try:
            client = self._get_client()
            response = self._call_text_model(
                "plan repair",
                lambda model_id: client.models.generate_content(
                    model=model_id,
                    contents=build_repair_prompt(plan, issues),
                    config=dict(self._JSON_CONFIG),
                ),
            )
            return extract_json(getattr(response, "text", "") or "")
        except UpstreamError as err:
            if self._injected_client:
                raise
            logger.warning("Gemini repair_plan failed with %s; falling back to Groq", err)
            from app.ai.groq_providers import GroqScriptProvider

            groq = GroqScriptProvider()
            return groq.repair_plan(plan, issues)


class GeminiImageProvider(_GenAIAdapter):
    """Keyframe/style-board images via gemini-3.1-flash-image."""

    _IMAGE_CONFIG = {
        "response_modalities": ["IMAGE"],
        "image_config": {"aspect_ratio": "9:16", "image_size": "1K"},
    }

    def _default_model(self) -> str:
        return get_model_config().gemini_image_model

    def generate_image(
        self, prompt: str, *, model_id: str | None = None, negative_prompt: str = ""
    ) -> ImageResult:
        model = model_id or self._model_id
        try:
            client = self._get_client()
            contents = prompt if not negative_prompt else f"{prompt}\n\nAvoid: {negative_prompt}"
            response = _call_genai(
                "image generation",
                lambda: client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=dict(self._IMAGE_CONFIG),
                ),
            )
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
        except UpstreamError as err:
            if self._injected_client:
                raise
            logger.warning("Gemini generate_image failed with %s; fetching HD photo / AI image", err)
            import subprocess
            import urllib.parse

            import httpx

            # Clean prompt of AI boilerplate for effective photo searches
            prompt_clean = prompt.split("Continuity:")[0].split("Consistency:")[0]
            prompt_clean = re.sub(r"(?i)\b(vertical\s*9:?16|shot\s*of|cinematic|photorealistic|no\s*text|no\s*logos?|realistic\s*lighting|high\s*quality|4k|hd|8k|soft\s*focus)\b", "", prompt_clean)
            prompt_clean = re.sub(r"[^a-zA-Z0-9 ]", " ", prompt_clean)
            words = [w for w in prompt_clean.split() if len(w) > 2]
            search_query = " ".join(words[:6]) if words else "nature landscape"
            # This value contains only ASCII letters, digits and spaces, so it
            # is safe to embed in FFmpeg's drawtext expression below.
            clean_q = search_query[:72]

            # 1. Search for real relevant HD photos matching the prompt
            try:
                r_ddg = httpx.get(
                    "https://duckduckgo.com/?q=" + urllib.parse.quote(search_query) + "&iax=images&ia=images",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
                    timeout=6.0,
                )
                vqd = re.search(r"vqd=([\d-]+)", r_ddg.text)
                if vqd:
                    r_json = httpx.get(
                        "https://duckduckgo.com/i.js?q=" + urllib.parse.quote(search_query) + f"&o=json&vqd={vqd.group(1)}",
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=6.0,
                    )
                    for item in r_json.json().get("results", [])[:8]:
                        img_url = item.get("image")
                        w, h = item.get("width") or 0, item.get("height") or 0
                        # Portrait candidates only: landscape/square photos get
                        # center-cropped to the 9:16 frame and lose both sides.
                        if w and h and w / h > 0.85:
                            continue
                        if img_url and img_url.startswith(("http://", "https://")):
                            try:
                                img_resp = httpx.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8.0, follow_redirects=True)
                                if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                                    return ImageResult(
                                        image_bytes=img_resp.content,
                                        model_id="hd_image_photo",
                                        mime_type="image/jpeg",
                                    )
                            except Exception:
                                continue
            except Exception as ddg_err:
                logger.warning("HD image search failed: %s", ddg_err)

            # 2. Try Pollinations AI image generator
            try:
                url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(search_query)}?width=720&height=1280&nologo=true&model=turbo"
                resp = httpx.get(url, timeout=10.0)
                if resp.status_code == 200 and resp.content and len(resp.content) > 1000:
                    return ImageResult(
                        image_bytes=resp.content,
                        model_id="pollinations_turbo",
                        mime_type=resp.headers.get("content-type") or "image/jpeg",
                    )
            except Exception as poll_err:
                logger.warning("Pollinations image generation failed: %s", poll_err)

            # 3. Try Lorem Picsum HD photo
            try:
                picsum_resp = httpx.get("https://picsum.photos/1080/1920", follow_redirects=True, timeout=6.0)
                if picsum_resp.status_code == 200 and len(picsum_resp.content) > 5000:
                    return ImageResult(
                        image_bytes=picsum_resp.content,
                        model_id="picsum_hd_photo",
                        mime_type="image/jpeg",
                    )
            except Exception:
                pass

            # 4. Final fallback visual card via FFmpeg
            lavfi = (
                "color=c=0x0f172a:s=1080x1920:d=1,format=rgba,"
                "drawbox=x=60:y=60:w=960:h=1800:color=0x38bdf8@0.7:t=8,"
                "drawbox=x=100:y=200:w=880:h=360:color=0x1e293b@0.9:t=fill,"
                f"drawtext=text='{clean_q}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=340"
            )
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    lavfi,
                    "-vframes",
                    "1",
                    "-f",
                    "image2",
                    "-c:v",
                    "png",
                    "pipe:1",
                ],
                capture_output=True,
                check=False,
            )
            if proc.stdout and len(proc.stdout) > 500:
                return ImageResult(
                    image_bytes=proc.stdout,
                    model_id="procedural_keyframe",
                    mime_type="image/png",
                )
            raw_fallback = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x02\xd0\x00\x00\x05\x00\x08\x02\x00\x00\x00\x06\x91\xd1\x8c"
                b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return ImageResult(
                image_bytes=raw_fallback,
                model_id="fallback_keyframe",
                mime_type="image/png",
            )


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

        duration = int(duration_seconds)
        if duration_seconds != duration or duration not in {4, 6, 8}:
            raise ValidationFailed(
                "Veo 3.1 duration must be exactly 4, 6, or 8 seconds",
                details={"duration_seconds": duration_seconds},
            )
        kwargs: dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "config": types.GenerateVideosConfig(
                aspect_ratio="9:16",
                resolution="720p",
                duration_seconds=duration,
            ),
        }
        if keyframe_bytes is not None:
            kwargs["image"] = types.Image(image_bytes=keyframe_bytes, mime_type="image/png")
        operation = _call_genai("video submission", lambda: client.models.generate_videos(**kwargs))
        name = getattr(operation, "name", None)
        if not name:
            raise UpstreamError("veo submit returned no operation name", retryable=True)
        return str(name)

    def poll(self, operation_name: str) -> VideoOperation:
        client = self._get_client()
        from google.genai import types  # noqa: PLC0415 — lazy: optional dependency

        # model_validate avoids an incomplete SDK type stub that rejects the
        # runtime-supported ``name=`` constructor argument under mypy.
        handle = types.GenerateVideosOperation.model_validate({"name": operation_name})
        operation = _call_genai("video polling", lambda: client.operations.get(handle))
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
