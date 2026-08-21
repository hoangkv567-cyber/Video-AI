"""VideoPlan v1 — the structured contract between scripting and generation.

Pydantic handles shape validation; `validate_semantics()` enforces the business
rules from PLAN.md §2 (duration, sources, narration length, metadata limits).
"""

from pydantic import BaseModel, Field, field_validator

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
    source_id: str
    url: str
    title: str = ""
    publisher: str = ""
    is_official: bool = False


class Fact(BaseModel):
    fact_id: str
    claim: str
    source_ids: list[str] = Field(min_length=1)


class ScenePlan(BaseModel):
    index: int = Field(ge=0, le=SCENE_COUNT - 1)
    duration_seconds: float = SCENE_SECONDS
    keyframe_prompt_en: str = Field(min_length=1)
    visual_prompt_en: str = Field(min_length=1)
    negative_prompt_en: str = ""
    continuity_note: str = ""
    fact_ids: list[str] = Field(default_factory=list)
    is_hero: bool = False


class LocaleContent(BaseModel):
    narration: list[str] = Field(min_length=SCENE_COUNT, max_length=SCENE_COUNT)
    on_screen_text: list[str] = Field(min_length=SCENE_COUNT, max_length=SCENE_COUNT)
    title: str = Field(min_length=1)
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)

    @field_validator("hashtags")
    @classmethod
    def normalize_hashtags(cls, v: list[str]) -> list[str]:
        return [h if h.startswith("#") else f"#{h}" for h in v]


class Disclosure(BaseModel):
    synthetic_media: bool = True
    made_for_kids: bool = False
    content_risks: list[str] = Field(default_factory=list)


class StyleGuide(BaseModel):
    palette: str = ""
    mood: str = ""
    camera: str = ""
    consistency_notes: str = ""


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
    urls = {s.url for s in plan.sources}
    if len(urls) < 2:
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
    """Auto mode requires zero blocking issues AND every scene mapped to sourced facts."""
    issues = issues if issues is not None else validate_semantics(plan)
    if blocking_issues(issues):
        return False
    return all(s.fact_ids for s in plan.scenes)
