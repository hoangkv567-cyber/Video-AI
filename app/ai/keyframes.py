"""Style board + per-scene keyframe generation through an ImageProvider.

Every image call is cap-checked and cost-recorded via the CostLedger. Results
carry a prompt hash (sha256 of model + prompt), the image bytes and metadata
ready for the immutable asset store.
"""

import hashlib
from dataclasses import dataclass

from app.ai.base import ImageProvider
from app.config import ModelConfig, get_model_config
from app.costs import CostLedger, image_price_usd
from app.models import Creative
from app.schemas.videoplan import ScenePlan, VideoPlan

KIND_STYLEBOARD = "styleboard"
KIND_KEYFRAME = "keyframe"


def prompt_hash(prompt: str, model_id: str) -> str:
    """sha256 over model + prompt: the provenance key stored on Asset rows."""
    return hashlib.sha256(f"{model_id}\n{prompt}".encode()).hexdigest()


def build_style_board_prompt(plan: VideoPlan) -> str:
    style = plan.style_guide
    parts = [f"Style board for a vertical 9:16 short video about: {plan.topic}."]
    if style.palette:
        parts.append(f"Palette: {style.palette}.")
    if style.mood:
        parts.append(f"Mood: {style.mood}.")
    if style.camera:
        parts.append(f"Camera: {style.camera}.")
    if style.consistency_notes:
        parts.append(f"Consistency: {style.consistency_notes}.")
    parts.append("No text, no logos, no watermarks.")
    return " ".join(parts)


def build_keyframe_prompt(plan: VideoPlan, scene: ScenePlan) -> str:
    parts = [scene.keyframe_prompt_en]
    if scene.continuity_note:
        parts.append(f"Continuity: {scene.continuity_note}.")
    if plan.style_guide.consistency_notes:
        parts.append(f"Consistency: {plan.style_guide.consistency_notes}.")
    parts.append("Vertical 9:16 composition. No text, no logos, no watermarks.")
    return " ".join(parts)


@dataclass(frozen=True)
class KeyframeImage:
    """Generated image + metadata for the asset store."""

    kind: str  # styleboard | keyframe
    scene_index: int | None
    prompt: str
    negative_prompt: str
    model_id: str
    prompt_hash: str
    image_bytes: bytes
    mime_type: str
    sha256: str
    cost_usd: float

    def as_asset_meta(self) -> dict:
        return {
            "kind": self.kind,
            "scene_index": self.scene_index,
            "model_id": self.model_id,
            "prompt_hash": self.prompt_hash,
            "sha256": self.sha256,
            "size_bytes": len(self.image_bytes),
            "mime_type": self.mime_type,
            "cost_usd": self.cost_usd,
        }


class KeyframeService:
    def __init__(
        self,
        provider: ImageProvider,
        ledger: CostLedger,
        model_config: ModelConfig | None = None,
    ) -> None:
        self._provider = provider
        self._ledger = ledger
        self._cfg = model_config or get_model_config()

    def generate_style_board(self, creative: Creative, plan: VideoPlan) -> KeyframeImage:
        prompt = build_style_board_prompt(plan)
        return self._generate(
            creative, prompt, negative_prompt="", kind=KIND_STYLEBOARD, scene_index=None
        )

    def generate_scene_keyframe(
        self, creative: Creative, plan: VideoPlan, scene_index: int
    ) -> KeyframeImage:
        scene = next(s for s in plan.scenes if s.index == scene_index)
        prompt = build_keyframe_prompt(plan, scene)
        return self._generate(
            creative,
            prompt,
            negative_prompt=scene.negative_prompt_en,
            kind=KIND_KEYFRAME,
            scene_index=scene.index,
        )

    def generate_all(
        self, creative: Creative, plan: VideoPlan
    ) -> tuple[KeyframeImage, list[KeyframeImage]]:
        """Style board first (consistency), then one keyframe per scene in order."""
        board = self.generate_style_board(creative, plan)
        keyframes = [
            self.generate_scene_keyframe(creative, plan, scene.index)
            for scene in sorted(plan.scenes, key=lambda s: s.index)
        ]
        return board, keyframes

    def _generate(
        self,
        creative: Creative,
        prompt: str,
        *,
        negative_prompt: str,
        kind: str,
        scene_index: int | None,
    ) -> KeyframeImage:
        price = image_price_usd(self._cfg)
        self._ledger.check_cap(creative, price)
        result = self._provider.generate_image(
            prompt, model_id=self._cfg.gemini_image_model, negative_prompt=negative_prompt
        )
        note = f"{kind}" if scene_index is None else f"{kind} scene {scene_index}"
        self._ledger.record_actual(
            creative.id,
            kind="gemini_image",
            model_id=result.model_id,
            units=1.0,
            unit_price_usd=price,
            note=note,
        )
        return KeyframeImage(
            kind=kind,
            scene_index=scene_index,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=result.model_id,
            prompt_hash=prompt_hash(prompt, result.model_id),
            image_bytes=result.image_bytes,
            mime_type=result.mime_type,
            sha256=hashlib.sha256(result.image_bytes).hexdigest(),
            cost_usd=price,
        )
