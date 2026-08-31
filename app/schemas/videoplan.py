"""VideoPlan v1 — the structured contract between scripting and generation.

Pydantic handles shape validation; `validate_semantics()` enforces the business
rules from PLAN.md §2 (duration, sources, narration length, metadata limits).
"""

import re
from contextlib import suppress
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.ai.topic_snapshots import validate_http_source_url
from app.ai.topics import registrable_domain

SCHEMA_VERSION = "1.0"
SCENE_COUNT = 5
SCENE_SECONDS = 8.0
CROSSFADE_SECONDS = 0.3
MASTER_DURATION = SCENE_COUNT * SCENE_SECONDS - (SCENE_COUNT - 1) * CROSSFADE_SECONDS  # 38.8

# Voice-over budget per scene; >5% over triggers a Gemini shorten pass, never a
# time-stretch outside 0.95–1.05.
NARRATION_MAX_SECONDS = 7.6
NARRATION_TOLERANCE = 0.05

# Speaking-rate heuristics for pre-TTS length estimation (chars/sec).
SPEECH_CHARS_PER_SECOND = {"vi": 15.0, "en": 14.0}

TITLE_MAX = 100  # YouTube hard limit
DESCRIPTION_MAX = 5000
HASHTAG_MAX_COUNT = 15
ON_SCREEN_TEXT_MAX_CHARS = 80


class SourceRef(BaseModel):
    source_id: str = Field(default="src_1")
    url: str
    title: str = ""
    publisher: str = ""
    is_official: bool = False

    _safe_http_url = field_validator("url")(validate_http_source_url)

    @model_validator(mode="before")
    @classmethod
    def parse_source_input(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {
                "source_id": "src_1",
                "url": data,
                "title": data,
            }
        if isinstance(data, dict):
            d = dict(data)
            if not d.get("source_id"):
                d["source_id"] = "src_1"
            return d
        return data


class Fact(BaseModel):
    fact_id: str = Field(default="fact_1")
    claim: str = ""
    source_ids: list[str] = Field(default_factory=lambda: ["src_1"])

    @model_validator(mode="before")
    @classmethod
    def parse_fact_input(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"fact_id": "fact_1", "claim": data, "source_ids": ["src_1"]}
        if isinstance(data, dict):
            d = dict(data)
            fact_id = d.get("fact_id") or "fact_1"
            claim = d.get("claim") or d.get("text") or d.get("fact") or str(d)
            source_ids = d.get("source_ids") or ["src_1"]
            if isinstance(source_ids, str):
                source_ids = [s.strip() for s in source_ids.split(",") if s.strip()] or ["src_1"]
            return {"fact_id": str(fact_id), "claim": str(claim), "source_ids": [str(s) for s in source_ids]}
        return data


class ScenePlan(BaseModel):
    index: int = Field(default=0, ge=0, le=SCENE_COUNT - 1)
    duration_seconds: float = SCENE_SECONDS
    keyframe_prompt_en: str = Field(default="Cinematic keyframe")
    visual_prompt_en: str = Field(default="Cinematic video shot")
    negative_prompt_en: str = ""
    continuity_note: str = ""
    fact_ids: list[str] = Field(default_factory=list)
    is_hero: bool = False

    @model_validator(mode="before")
    @classmethod
    def parse_scene_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            prompt = (
                d.get("keyframe_prompt_en")
                or d.get("visual_prompt_en")
                or d.get("prompt")
                or "Cinematic video shot"
            )
            d.setdefault("keyframe_prompt_en", str(prompt))
            d.setdefault("visual_prompt_en", str(prompt))
            for str_field in ["keyframe_prompt_en", "visual_prompt_en", "negative_prompt_en", "continuity_note"]:
                val = d.get(str_field)
                if isinstance(val, (list, tuple, set)):
                    d[str_field] = ", ".join(str(x) for x in val)
                elif val is not None and not isinstance(val, str):
                    d[str_field] = str(val)
            if isinstance(d.get("fact_ids"), str):
                d["fact_ids"] = [f.strip() for f in d["fact_ids"].split(",") if f.strip()]
            return d
        return data


class LocaleContent(BaseModel):
    narration: list[str] = Field(min_length=SCENE_COUNT, max_length=SCENE_COUNT)
    on_screen_text: list[str] = Field(min_length=SCENE_COUNT, max_length=SCENE_COUNT)
    title: str = Field(min_length=1)
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def parse_locale_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            if not d.get("title"):
                d["title"] = d.get("name") or d.get("video_title") or d.get("topic") or "Short Video"
            for str_field in ["title", "description"]:
                val = d.get(str_field)
                if isinstance(val, (list, tuple, set)):
                    d[str_field] = " ".join(str(x) for x in val)
                elif val is not None and not isinstance(val, str):
                    d[str_field] = str(val)
            return d
        return data

    @field_validator("narration", "on_screen_text", mode="before")
    @classmethod
    def parse_list_lines(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return [str(x) for x in v.values()]
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return v

    @field_validator("hashtags", mode="before")
    @classmethod
    def parse_and_normalize_hashtags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            items = [part.strip() for part in re.split(r"[\s,]+", v) if part.strip()]
            return [h if h.startswith("#") else f"#{h}" for h in items]
        if isinstance(v, list):
            return [str(h) if str(h).startswith("#") else f"#{h}" for h in v if str(h).strip()]
        return []


class Disclosure(BaseModel):
    synthetic_media: bool = True
    made_for_kids: bool = False
    content_risks: list[str] = Field(default_factory=list)


class StyleGuide(BaseModel):
    palette: str = ""
    mood: str = ""
    camera: str = ""
    consistency_notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def parse_style_guide_input(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {
                "palette": "",
                "mood": data,
                "camera": "",
                "consistency_notes": "",
            }
        if isinstance(data, dict):
            d = dict(data)
            for k in ["palette", "mood", "camera", "consistency_notes"]:
                v = d.get(k)
                if isinstance(v, (list, tuple, set)):
                    d[k] = ", ".join(str(x) for x in v)
                elif v is not None and not isinstance(v, str):
                    d[k] = str(v)
            return d
        return data


class VideoPlan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    topic: str = Field(min_length=1)
    angle: str = ""
    style_guide: StyleGuide = Field(default_factory=StyleGuide)
    sources: list[SourceRef] = Field(min_length=1)
    facts: list[Fact] = Field(default_factory=list)
    scenes: list[ScenePlan] = Field(min_length=SCENE_COUNT, max_length=SCENE_COUNT)
    locales: dict[str, LocaleContent]  # keys: "vi", "en"
    disclosure: Disclosure = Field(default_factory=Disclosure)
    expected_cost_usd: float = Field(ge=0.0, default=0.0)

    @model_validator(mode="before")
    @classmethod
    def unwrap_root_plan(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            # Unwrap nested wrapper keys if model returned e.g. {"video_plan": {...}} or {"plan": {...}}
            for wrapper_key in ["video_plan", "plan", "video_plan_v1", "videoplan", "VideoPlan"]:
                if wrapper_key in d and isinstance(d[wrapper_key], dict):
                    d = dict(d[wrapper_key])
                    break

            topic = d.get("topic") or d.get("title")
            if topic and "topic" not in d:
                d["topic"] = topic

            # Schema version normalization
            if str(d.get("schema_version", "")).lower() in ("v1", "1", "v1.0", "1.0"):
                d["schema_version"] = SCHEMA_VERSION

            # Cost normalization
            if "expected_cost_usd" not in d:
                for cost_key in ["estimated_cost_usd", "cost_usd", "cost"]:
                    if cost_key in d:
                        with suppress(ValueError, TypeError):
                            d["expected_cost_usd"] = float(d[cost_key])
                        break

            # Compliance / disclosure normalization
            if "compliance" in d and isinstance(d["compliance"], dict):
                comp = d["compliance"]
                if "disclosure" not in d or not isinstance(d.get("disclosure"), dict):
                    d["disclosure"] = {
                        "synthetic_media": True,
                        "made_for_kids": bool(comp.get("made_for_kids", False)),
                        "content_risks": [str(r) for r in comp.get("content_risks", []) if r] if isinstance(comp.get("content_risks"), list) else [],
                    }

            # Style guide string normalization
            if isinstance(d.get("style_guide"), str):
                d["style_guide"] = {
                    "palette": "",
                    "mood": d["style_guide"],
                    "camera": "",
                    "consistency_notes": "",
                }

            # Unwrap scenes if named differently
            if "scenes" not in d:
                for alt in ["scene_plans", "scene_list", "scenes_plan", "video_scenes", "shots"]:
                    if alt in d and isinstance(d[alt], list):
                        d["scenes"] = d[alt]
                        break

            # Normalize individual scenes
            if "scenes" in d and isinstance(d["scenes"], list):
                norm_scenes = []
                for i, s in enumerate(d["scenes"]):
                    if isinstance(s, dict):
                        sc = dict(s)
                        if "index" not in sc:
                            sc["index"] = int(sc.get("scene_number", i + 1)) - 1 if "scene_number" in sc else i
                        if "negative_prompt_en" not in sc and "negative_prompt" in sc:
                            sc["negative_prompt_en"] = sc["negative_prompt"]
                        norm_scenes.append(sc)
                    else:
                        norm_scenes.append(s)
                d["scenes"] = norm_scenes

            # Normalize locales from scenes[].vi/en and metadata_vi/en if locales is missing/empty
            if ("locales" not in d or not isinstance(d.get("locales"), dict)) and "scenes" in d and isinstance(d["scenes"], list):
                meta_vi = d.get("metadata_vi", {}) if isinstance(d.get("metadata_vi"), dict) else {}
                meta_en = d.get("metadata_en", {}) if isinstance(d.get("metadata_en"), dict) else {}

                narr_vi, ost_vi = [], []
                narr_en, ost_en = [], []
                for s in d["scenes"]:
                    if isinstance(s, dict):
                        vi_obj = s.get("vi", {}) if isinstance(s.get("vi"), dict) else {}
                        en_obj = s.get("en", {}) if isinstance(s.get("en"), dict) else {}
                        narr_vi.append(str(vi_obj.get("narration", "")))
                        ost_vi.append(str(vi_obj.get("on_screen_text", "")))
                        narr_en.append(str(en_obj.get("narration", "")))
                        ost_en.append(str(en_obj.get("on_screen_text", "")))

                d["locales"] = {
                    "vi": {
                        "narration": narr_vi,
                        "on_screen_text": ost_vi,
                        "title": str(meta_vi.get("title") or topic or "Short Video"),
                        "description": str(meta_vi.get("description", "")),
                        "hashtags": list(meta_vi.get("hashtags", [])),
                    },
                    "en": {
                        "narration": narr_en,
                        "on_screen_text": ost_en,
                        "title": str(meta_en.get("title") or topic or "Short Video"),
                        "description": str(meta_en.get("description", "")),
                        "hashtags": list(meta_en.get("hashtags", [])),
                    },
                }

            # Ensure locales title from topic if present
            if "locales" in d and isinstance(d["locales"], dict):
                for loc in ["vi", "en"]:
                    if loc in d["locales"] and isinstance(d["locales"][loc], dict):
                        loc_dict = dict(d["locales"][loc])
                        if not loc_dict.get("title") and topic:
                            loc_dict["title"] = topic
                        d["locales"][loc] = loc_dict
            return d
        return d

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = list(v.values())
        if isinstance(v, list):
            return [
                {"source_id": f"src_{i + 1}", "url": s, "title": s} if isinstance(s, str) else s
                for i, s in enumerate(v)
            ]
        return v

    @field_validator("facts", mode="before")
    @classmethod
    def parse_facts(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            v = list(v.values())
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            parsed = []
            for i, item in enumerate(v):
                if isinstance(item, str):
                    parsed.append({
                        "fact_id": f"fact_{i + 1}",
                        "claim": item,
                        "source_ids": ["src_1"],
                    })
                else:
                    parsed.append(item)
            return parsed
        return v

    @field_validator("scenes", mode="before")
    @classmethod
    def parse_scenes(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return list(v.values())
        return v


class SemanticIssue(BaseModel):
    code: str
    message: str
    scene_index: int | None = None
    locale: str | None = None
    blocking: bool = True


def estimate_speech_seconds(text: str, locale: str) -> float:
    rate = SPEECH_CHARS_PER_SECOND.get(locale, 14.0)
    return len(text.strip()) / rate if text.strip() else 0.0


def validate_semantics(plan: VideoPlan, cost_cap_usd: float = 6.0) -> list[SemanticIssue]:
    """Business-rule validation. Blocking issues stop approval and auto mode."""
    issues: list[SemanticIssue] = []

    if plan.schema_version != SCHEMA_VERSION:
        issues.append(
            SemanticIssue(
                code="schema_version",
                message=f"schema_version must be {SCHEMA_VERSION}, got {plan.schema_version}",
            )
        )

    # Scene indices must be exactly 0..4 and each scene 8.0 s.
    indices = sorted(s.index for s in plan.scenes)
    if indices != list(range(SCENE_COUNT)):
        issues.append(
            SemanticIssue(code="scene_indices", message=f"scene indices must be 0..4, got {indices}")
        )
    for s in plan.scenes:
        if abs(s.duration_seconds - SCENE_SECONDS) > 1e-6:
            issues.append(
                SemanticIssue(
                    code="scene_duration",
                    message=f"scene {s.index} must be {SCENE_SECONDS}s, got {s.duration_seconds}",
                    scene_index=s.index,
                )
            )

    total = sum(s.duration_seconds for s in plan.scenes) - (SCENE_COUNT - 1) * CROSSFADE_SECONDS
    if abs(total - MASTER_DURATION) > 0.5:
        issues.append(
            SemanticIssue(
                code="total_duration",
                message=f"master duration {total:.1f}s deviates from {MASTER_DURATION}s",
            )
        )

    # Source integrity: >=2 independent sources, prefer >=1 official (warning only).
    domains = {registrable_domain(s.url) for s in plan.sources}
    if len(domains) < 2:
        issues.append(
            SemanticIssue(code="sources_min", message="at least 2 independent sources required")
        )
    if not any(s.is_official for s in plan.sources):
        issues.append(
            SemanticIssue(
                code="sources_official",
                message="no official source; auto mode discouraged",
                blocking=False,
            )
        )

    # Every fact must reference known sources; every scene fact_id must exist.
    source_ids = {s.source_id for s in plan.sources}
    fact_ids = {f.fact_id for f in plan.facts}
    for f in plan.facts:
        missing = set(f.source_ids) - source_ids
        if missing:
            issues.append(
                SemanticIssue(
                    code="fact_source_missing",
                    message=f"fact {f.fact_id} references unknown sources {sorted(missing)}",
                )
            )
    for s in plan.scenes:
        missing = set(s.fact_ids) - fact_ids
        if missing:
            issues.append(
                SemanticIssue(
                    code="scene_fact_missing",
                    message=f"scene {s.index} references unknown facts {sorted(missing)}",
                    scene_index=s.index,
                )
            )

    # Locales: both vi and en required.
    for locale in ("vi", "en"):
        if locale not in plan.locales:
            issues.append(SemanticIssue(code="locale_missing", message=f"locale {locale} missing"))
            continue
        content = plan.locales[locale]

        for i, narration in enumerate(content.narration):
            est = estimate_speech_seconds(narration, locale)
            if est > NARRATION_MAX_SECONDS * (1 + NARRATION_TOLERANCE):
                issues.append(
                    SemanticIssue(
                        code="narration_too_long",
                        message=(
                            f"scene {i} narration ~{est:.1f}s exceeds "
                            f"{NARRATION_MAX_SECONDS}s budget by >5%; shorten required"
                        ),
                        scene_index=i,
                        locale=locale,
                    )
                )
        for i, text in enumerate(content.on_screen_text):
            if len(text) > ON_SCREEN_TEXT_MAX_CHARS:
                issues.append(
                    SemanticIssue(
                        code="on_screen_text_too_long",
                        message=f"scene {i} on-screen text exceeds {ON_SCREEN_TEXT_MAX_CHARS} chars",
                        scene_index=i,
                        locale=locale,
                        blocking=False,
                    )
                )
        if len(content.title) > TITLE_MAX:
            issues.append(
                SemanticIssue(
                    code="title_too_long",
                    message=f"title exceeds {TITLE_MAX} chars",
                    locale=locale,
                )
            )
        if len(content.description) > DESCRIPTION_MAX:
            issues.append(
                SemanticIssue(
                    code="description_too_long",
                    message=f"description exceeds {DESCRIPTION_MAX} chars",
                    locale=locale,
                )
            )
        if len(content.hashtags) > HASHTAG_MAX_COUNT:
            issues.append(
                SemanticIssue(
                    code="too_many_hashtags",
                    message=f"more than {HASHTAG_MAX_COUNT} hashtags",
                    locale=locale,
                    blocking=False,
                )
            )

    if not plan.disclosure.synthetic_media:
        issues.append(
            SemanticIssue(
                code="disclosure_required",
                message="synthetic media disclosure must be enabled for AI-generated video",
            )
        )

    if plan.expected_cost_usd > cost_cap_usd:
        issues.append(
            SemanticIssue(
                code="cost_cap",
                message=(
                    f"expected cost ${plan.expected_cost_usd:.2f} exceeds "
                    f"hard cap ${cost_cap_usd:.2f}"
                ),
            )
        )

    return issues


def blocking_issues(issues: list[SemanticIssue]) -> list[SemanticIssue]:
    return [i for i in issues if i.blocking]


def auto_mode_allowed(plan: VideoPlan, issues: list[SemanticIssue] | None = None) -> bool:
    """Auto mode requires clean gates, sourced scenes and no declared content risk."""
    issues = issues if issues is not None else validate_semantics(plan)
    if blocking_issues(issues) or plan.disclosure.content_risks:
        return False
    return all(s.fact_ids for s in plan.scenes)
