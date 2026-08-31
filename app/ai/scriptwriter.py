"""ScriptService: VideoPlan v1 generation, shorten loop, cost projection and cap gate.

Flow: provider produces a plan -> semantic validation -> narrations flagged
`narration_too_long` are sent back to the provider to shorten (bounded retries)
-> expected cost computed from the versioned model config -> cap enforced via
CostLedger -> projected CostEvent rows recorded. Facts without sources surface
as blocking issues which disable auto mode (`auto_mode_allowed`).
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.base import ScriptProvider, SourceInfo
from app.config import ModelConfig, Settings, get_model_config, get_settings
from app.costs import CostLedger, tts_price_per_char_usd
from app.errors import ValidationFailed
from app.models import Creative
from app.schemas.videoplan import (
    SemanticIssue,
    VideoPlan,
    auto_mode_allowed,
    blocking_issues,
    validate_semantics,
)

MAX_SHORTEN_RETRIES = 2
MAX_FACT_REPAIR_RETRIES = 1
_FACT_REPAIR_CODES = frozenset({"scene_fact_missing", "fact_source_missing"})


@dataclass(frozen=True)
class ProjectedCost:
    kind: str
    model_id: str
    units: float
    unit_price_usd: float
    note: str

    @property
    def amount_usd(self) -> float:
        return self.units * self.unit_price_usd


@dataclass(frozen=True)
class ScriptResult:
    plan: VideoPlan
    plan_dict: dict
    issues: list[SemanticIssue]
    blocking: list[SemanticIssue]
    auto_mode_allowed: bool
    expected_cost_usd: float
    shorten_attempts: int


def _narration_issues(issues: list[SemanticIssue]) -> list[SemanticIssue]:
    return [i for i in issues if i.code == "narration_too_long"]


def _canonicalize_topic_and_sources(raw: dict, topic: str, sources: Sequence[SourceInfo]) -> dict:
    """Never trust the model to echo factual source URLs back unchanged."""
    normalized = copy.deepcopy(raw)
    normalized["topic"] = topic
    canonical_sources = [
        {
            "source_id": f"src{index}",
            "url": source.url,
            "title": source.title,
            "publisher": source.publisher,
            "is_official": source.is_official,
        }
        for index, source in enumerate(sources, start=1)
    ]
    normalized["sources"] = canonical_sources
    canonical_src_ids = [f"src{index}" for index in range(1, len(canonical_sources) + 1)]

    # Normalize formatting of fact source_ids (e.g. src_1 -> src1, preserve unknown IDs)
    if "facts" in normalized and isinstance(normalized["facts"], list):
        for fact in normalized["facts"]:
            if isinstance(fact, dict):
                raw_sids = fact.get("source_ids") or []
                mapped = []
                for sid in raw_sids:
                    clean_sid = str(sid).replace("_", "").lower()
                    found = False
                    for cid in canonical_src_ids:
                        if clean_sid == cid.replace("_", "").lower():
                            mapped.append(cid)
                            found = True
                            break
                    if not found:
                        mapped.append(sid)
                fact["source_ids"] = mapped

    return normalized


def _require_shorten_only_changed_narration(
    before: VideoPlan,
    after: VideoPlan,
    issues: list[SemanticIssue],
) -> None:
    expected = before.model_dump(mode="json")
    actual = after.model_dump(mode="json")
    for issue in issues:
        if issue.locale is None or issue.scene_index is None:
            continue
        expected["locales"][issue.locale]["narration"][issue.scene_index] = actual["locales"][
            issue.locale
        ]["narration"][issue.scene_index]
    if actual != expected:
        raise ValidationFailed(
            "narration shortening changed fields outside the requested narration entries",
            code="unsafe_shorten_response",
        )


class ScriptService:
    def __init__(
        self,
        provider: ScriptProvider,
        ledger: CostLedger,
        model_config: ModelConfig | None = None,
        max_shorten_retries: int = MAX_SHORTEN_RETRIES,
        settings: Settings | None = None,
    ) -> None:
        self._provider = provider
        self._ledger = ledger
        self._cfg = model_config or get_model_config()
        self._max_shorten_retries = max_shorten_retries
        self._settings = settings or get_settings()

    # -- cost projection ----------------------------------------------------

    def cost_breakdown(self, plan: VideoPlan) -> list[ProjectedCost]:
        """Projection matched to the configured video and TTS providers."""
        cfg = self._cfg
        items: list[ProjectedCost] = []
        if self._settings.video_provider == "veo":
            items.append(
                ProjectedCost(
                    kind="veo",
                    model_id=cfg.veo_model_lite,
                    units=cfg.scene_count * cfg.scene_seconds,
                    unit_price_usd=cfg.veo_lite_usd_per_second,
                    note=f"projected: {cfg.scene_count} scenes x {cfg.scene_seconds:g}s veo lite",
                )
            )
        elif self._settings.video_provider == "wan":
            items.append(
                ProjectedCost(
                    kind="wan",
                    model_id=cfg.wan_i2v_model,
                    units=cfg.scene_count * cfg.scene_seconds,
                    unit_price_usd=cfg.wan_usd_per_second_720p,
                    note=(
                        f"projected: {cfg.scene_count} scenes x {cfg.scene_seconds:g}s "
                        "wan2.7-i2v 720P (50s free quota applies first)"
                    ),
                )
            )
        if self._settings.image_provider == "procedural":
            items.append(
                ProjectedCost(
                    kind="free_image",
                    model_id="procedural",
                    units=float(cfg.scene_count + 1),
                    unit_price_usd=0.0,
                    note="projected: style board + scene keyframes via free image providers",
                )
            )
        else:
            items.append(
                ProjectedCost(
                    kind="gemini_image",
                    model_id=cfg.gemini_image_model,
                    units=float(cfg.scene_count + 1),
                    unit_price_usd=cfg.gemini_image_usd_per_image,
                    note="projected: style board + scene keyframes",
                )
            )
        items.append(
            ProjectedCost(
                kind="gemini_text",
                model_id=cfg.gemini_text_model,
                units=2.0,
                unit_price_usd=cfg.gemini_text_usd_per_call_estimate,
                note="projected: research + structured output calls",
            )
        )
        for locale, content in plan.locales.items():
            choice = (
                self._settings.tts_provider_vi if locale == "vi" else self._settings.tts_provider_en
            )
            narration_chars = sum(len(narration) for narration in content.narration)
            if choice == "google":
                model_id = cfg.tts_voice_vi if locale == "vi" else cfg.tts_voice_en
                price = tts_price_per_char_usd(cfg)
            elif choice == "groq":
                model_id = cfg.groq_tts_model_en
                price = 0.0
            else:
                model_id = cfg.edge_tts_voice_vi if locale == "vi" else "edge-tts-en"
                price = 0.0
            items.append(
                ProjectedCost(
                    kind="tts",
                    model_id=model_id,
                    units=float(narration_chars),
                    unit_price_usd=price,
                    note=f"projected: {locale} narration via {choice}",
                )
            )
        return items

    def estimate_expected_cost(self, plan: VideoPlan) -> float:
        return round(sum(item.amount_usd for item in self.cost_breakdown(plan)), 4)

    # -- fact binding repair -------------------------------------------------

    def _repair_fact_bindings(
        self,
        plan: VideoPlan,
        issues: list[SemanticIssue],
        topic: str,
        sources: Sequence[SourceInfo],
        creative: Creative,
    ) -> tuple[VideoPlan, list[SemanticIssue]]:
        """One bounded provider pass to fix fact/source binding errors.

        Models occasionally emit scene ``fact_ids`` without the matching
        ``facts`` array, which blocks approval. The repair provider must return
        the full plan with truthful facts citing existing sources; the repaired
        plan is accepted only when it still validates and every fact-binding
        blocking issue is resolved. Any failure keeps the original plan and its
        issues (the manual gate stays in charge).
        """
        repairable = [
            i for i in blocking_issues(issues) if i.code in _FACT_REPAIR_CODES
        ]
        if not repairable or not hasattr(self._provider, "repair_plan"):
            return plan, issues

        try:
            raw = _canonicalize_topic_and_sources(
                self._provider.repair_plan(
                    plan.model_dump(mode="json"), [i.model_dump() for i in repairable]
                ),
                topic,
                sources,
            )
            repaired = VideoPlan.model_validate(raw)
            repaired_issues = validate_semantics(
                repaired, cost_cap_usd=creative.cost_cap_usd
            )
        except Exception:  # noqa: BLE001 — repair is best-effort, never fatal
            return plan, issues
        if any(
            i.code in _FACT_REPAIR_CODES for i in blocking_issues(repaired_issues)
        ) or len(repaired.scenes) != len(plan.scenes):
            return plan, issues
        return repaired, repaired_issues

    # -- main entry ---------------------------------------------------------

    def create_plan(
        self,
        creative: Creative,
        topic: str,
        sources: Sequence[SourceInfo],
        brief: str = "",
    ) -> ScriptResult:
        raw = _canonicalize_topic_and_sources(
            self._provider.generate_plan(topic, list(sources), brief), topic, sources
        )
        plan = VideoPlan.model_validate(raw)
        issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)

        attempts = 0
        while attempts < self._max_shorten_retries:
            narration = _narration_issues(issues)
            if not narration:
                break
            raw = _canonicalize_topic_and_sources(
                self._provider.shorten_narrations(
                    plan.model_dump(mode="json"), [i.model_dump() for i in narration]
                ),
                topic,
                sources,
            )
            shortened = VideoPlan.model_validate(raw)
            _require_shorten_only_changed_narration(plan, shortened, narration)
            plan = shortened
            attempts += 1
            issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)

        plan, issues = self._repair_fact_bindings(
            plan, issues, topic, sources, creative
        )

        breakdown = self.cost_breakdown(plan)
        expected = round(sum(item.amount_usd for item in breakdown), 4)
        plan.expected_cost_usd = expected
        # Re-validate so the cost-cap semantic check sees the final expected cost.
        issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)

        # Hard gate BEFORE recording projections (avoids double counting the check).
        self._ledger.check_cap(creative, expected)
        for item in breakdown:
            self._ledger.record_projected(
                creative.id,
                kind=item.kind,
                model_id=item.model_id,
                units=item.units,
                unit_price_usd=item.unit_price_usd,
                note=item.note,
            )

        return ScriptResult(
            plan=plan,
            plan_dict=plan.model_dump(mode="json"),
            issues=issues,
            blocking=blocking_issues(issues),
            auto_mode_allowed=auto_mode_allowed(plan, issues),
            expected_cost_usd=expected,
            shorten_attempts=attempts,
        )
