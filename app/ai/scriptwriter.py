"""ScriptService: VideoPlan v1 generation, shorten loop, cost projection and cap gate.

Flow: provider produces a plan -> semantic validation -> narrations flagged
`narration_too_long` are sent back to the provider to shorten (bounded retries)
-> expected cost computed from the versioned model config -> cap enforced via
CostLedger -> projected CostEvent rows recorded. Facts without sources surface
as blocking issues which disable auto mode (`auto_mode_allowed`).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.base import ScriptProvider, SourceInfo
from app.config import ModelConfig, get_model_config
from app.costs import CostLedger, tts_price_per_char_usd
from app.models import Creative
from app.schemas.videoplan import (
    SemanticIssue,
    VideoPlan,
    auto_mode_allowed,
    blocking_issues,
    validate_semantics,
)

MAX_SHORTEN_RETRIES = 2

TTS_MODEL_LABEL = "google-cloud-tts"


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


class ScriptService:
    def __init__(
        self,
        provider: ScriptProvider,
        ledger: CostLedger,
        model_config: ModelConfig | None = None,
        max_shorten_retries: int = MAX_SHORTEN_RETRIES,
    ) -> None:
        self._provider = provider
        self._ledger = ledger
        self._cfg = model_config or get_model_config()
        self._max_shorten_retries = max_shorten_retries

    # -- cost projection ----------------------------------------------------

    def cost_breakdown(self, plan: VideoPlan) -> list[ProjectedCost]:
        """5 scenes x 8 s Veo lite + style board & keyframes + text calls + TTS estimate."""
        cfg = self._cfg
        narration_chars = sum(
            len(narration)
            for content in plan.locales.values()
            for narration in content.narration
        )
        return [
            ProjectedCost(
                kind="veo",
                model_id=cfg.veo_model_lite,
                units=cfg.scene_count * cfg.scene_seconds,
                unit_price_usd=cfg.veo_lite_usd_per_second,
                note=f"projected: {cfg.scene_count} scenes x {cfg.scene_seconds:g}s veo lite",
            ),
            ProjectedCost(
                kind="gemini_image",
                model_id=cfg.gemini_image_model,
                units=float(cfg.scene_count + 1),
                unit_price_usd=cfg.gemini_image_usd_per_image,
                note="projected: style board + scene keyframes",
            ),
            ProjectedCost(
                kind="gemini_text",
                model_id=cfg.gemini_text_model,
                units=2.0,
                unit_price_usd=cfg.gemini_text_usd_per_call_estimate,
                note="projected: research + structured output calls",
            ),
            ProjectedCost(
                kind="tts",
                model_id=TTS_MODEL_LABEL,
                units=float(narration_chars),
                unit_price_usd=tts_price_per_char_usd(cfg),
                note="projected: vi+en narration synthesis",
            ),
        ]

    def estimate_expected_cost(self, plan: VideoPlan) -> float:
        return round(sum(item.amount_usd for item in self.cost_breakdown(plan)), 4)

    # -- main entry ---------------------------------------------------------

    def create_plan(
        self,
        creative: Creative,
        topic: str,
        sources: Sequence[SourceInfo],
        brief: str = "",
    ) -> ScriptResult:
        raw = self._provider.generate_plan(topic, list(sources), brief)
        plan = VideoPlan.model_validate(raw)
        issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)

        attempts = 0
        while attempts < self._max_shorten_retries:
            narration = _narration_issues(issues)
            if not narration:
                break
            raw = self._provider.shorten_narrations(
                plan.model_dump(mode="json"), [i.model_dump() for i in narration]
            )
            plan = VideoPlan.model_validate(raw)
            attempts += 1
            issues = validate_semantics(plan, cost_cap_usd=creative.cost_cap_usd)

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
