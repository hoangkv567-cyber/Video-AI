"""Cost ledger: projected/actual CostEvent rows, totals and the per-creative hard cap.

Every external cost-bearing call must be recorded here. `check_cap` raises
`CostCapExceeded` when actual + projected + additional would exceed the
creative's cap unless an admin override (with a mandatory reason) is supplied,
in which case an `AuditEvent` row is written instead.
"""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import ModelConfig, get_model_config, get_settings
from app.errors import CostCapExceeded, ValidationFailed
from app.models import AuditEvent, CostEvent, Creative

# ---------------------------------------------------------------------------
# Price lookups — always sourced from the versioned model config, never inline.
# ---------------------------------------------------------------------------


def veo_price_per_second(model_id: str, cfg: ModelConfig | None = None) -> float:
    """USD per generated second for a Veo model (lite 0.05 / fast 0.15 by default)."""
    cfg = cfg or get_model_config()
    prices = {
        cfg.veo_model_lite: cfg.veo_lite_usd_per_second,
        cfg.veo_model_fast: cfg.veo_fast_usd_per_second,
    }
    try:
        return prices[model_id]
    except KeyError:
        raise ValueError(f"unknown veo model: {model_id}") from None


def image_price_usd(cfg: ModelConfig | None = None) -> float:
    cfg = cfg or get_model_config()
    return cfg.gemini_image_usd_per_image


def text_call_price_usd(cfg: ModelConfig | None = None) -> float:
    cfg = cfg or get_model_config()
    return cfg.gemini_text_usd_per_call_estimate


def tts_price_per_char_usd(cfg: ModelConfig | None = None) -> float:
    cfg = cfg or get_model_config()
    return cfg.tts_usd_per_million_chars / 1_000_000.0


# ---------------------------------------------------------------------------
# Ledger service
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapCheck:
    """Result of a cap check; `overridden` means an audit-logged admin override."""

    cap_usd: float
    total_actual_usd: float
    total_projected_usd: float
    additional_usd: float
    overridden: bool = False

    @property
    def committed_total_usd(self) -> float:
        return self.total_actual_usd + self.total_projected_usd + self.additional_usd


class CostLedger:
    """Records CostEvent rows for a creative and enforces the hard cost cap."""

    def __init__(
        self,
        db: Session,
        model_config: ModelConfig | None = None,
        default_cap_usd: float | None = None,
    ) -> None:
        self._db = db
        self._cfg = model_config or get_model_config()
        self._default_cap = (
            default_cap_usd if default_cap_usd is not None else get_settings().cost_hard_cap_usd
        )

    # -- recording ----------------------------------------------------------

    def record(
        self,
        creative_id: str,
        *,
        kind: str,
        model_id: str = "",
        units: float = 0.0,
        unit_price_usd: float = 0.0,
        amount_usd: float | None = None,
        projected: bool = False,
        job_id: str | None = None,
        note: str = "",
    ) -> CostEvent:
        amount = amount_usd if amount_usd is not None else units * unit_price_usd
        event = CostEvent(
            creative_id=creative_id,
            kind=kind,
            model_id=model_id,
            units=units,
            unit_price_usd=unit_price_usd,
            amount_usd=round(amount, 6),
            projected=projected,
            job_id=job_id,
            note=note,
        )
        self._db.add(event)
        self._db.flush()
        return event

    def record_projected(
        self,
        creative_id: str,
        *,
        kind: str,
        model_id: str = "",
        units: float = 0.0,
        unit_price_usd: float = 0.0,
        amount_usd: float | None = None,
        job_id: str | None = None,
        note: str = "",
    ) -> CostEvent:
        return self.record(
            creative_id,
            kind=kind,
            model_id=model_id,
            units=units,
            unit_price_usd=unit_price_usd,
            amount_usd=amount_usd,
            projected=True,
            job_id=job_id,
            note=note,
        )

    def record_actual(
        self,
        creative_id: str,
        *,
        kind: str,
        model_id: str = "",
        units: float = 0.0,
        unit_price_usd: float = 0.0,
        amount_usd: float | None = None,
        job_id: str | None = None,
        note: str = "",
    ) -> CostEvent:
        return self.record(
            creative_id,
            kind=kind,
            model_id=model_id,
            units=units,
            unit_price_usd=unit_price_usd,
            amount_usd=amount_usd,
            projected=False,
            job_id=job_id,
            note=note,
        )

    def remove_projected(
        self, creative_id: str, *, kind: str | None = None, note: str | None = None
    ) -> int:
        """Delete projected rows once the matching actuals are recorded."""
        query = self._db.query(CostEvent).filter(
            CostEvent.creative_id == creative_id, CostEvent.projected.is_(True)
        )
        if kind is not None:
            query = query.filter(CostEvent.kind == kind)
        if note is not None:
            query = query.filter(CostEvent.note == note)
        rows = query.all()
        for row in rows:
            self._db.delete(row)
        self._db.flush()
        return len(rows)

    # -- totals -------------------------------------------------------------

    def _total(self, creative_id: str, projected: bool) -> float:
        stmt = select(func.coalesce(func.sum(CostEvent.amount_usd), 0.0)).where(
            CostEvent.creative_id == creative_id, CostEvent.projected.is_(projected)
        )
        return float(self._db.execute(stmt).scalar_one())

    def total_actual(self, creative_id: str) -> float:
        return self._total(creative_id, projected=False)

    def total_projected(self, creative_id: str) -> float:
        return self._total(creative_id, projected=True)

    # -- cap enforcement ----------------------------------------------------

    def effective_cap_usd(self, creative: Creative) -> float:
        cap = creative.cost_cap_usd
        return cap if cap and cap > 0 else self._default_cap

    def check_cap(
        self,
        creative: Creative,
        additional_usd: float = 0.0,
        *,
        override: bool = False,
        override_reason: str | None = None,
        actor_id: str | None = None,
    ) -> CapCheck:
        """Raise `CostCapExceeded` when actual + projected + additional exceeds the cap.

        An admin override requires a reason and records an AuditEvent row instead
        of raising; overrides without a reason raise `ValidationFailed`.
        """
        cap = self.effective_cap_usd(creative)
        actual = self.total_actual(creative.id)
        projected = self.total_projected(creative.id)
        total = actual + projected + additional_usd

        if total <= cap + 1e-9:
            return CapCheck(cap, actual, projected, additional_usd)

        if not override:
            raise CostCapExceeded(
                f"cost ${total:.2f} would exceed cap ${cap:.2f} for creative {creative.id}",
                details={
                    "creative_id": creative.id,
                    "cap_usd": cap,
                    "total_actual_usd": round(actual, 6),
                    "total_projected_usd": round(projected, 6),
                    "additional_usd": round(additional_usd, 6),
                },
            )

        if not (override_reason and override_reason.strip()):
            raise ValidationFailed(
                "cost cap override requires a reason",
                details={"creative_id": creative.id},
            )

        audit = AuditEvent(
            actor_id=actor_id,
            actor_kind="user" if actor_id else "system",
            action="cost_cap_override",
            entity_type="creative",
            entity_id=creative.id,
            data={
                "reason": override_reason.strip(),
                "cap_usd": cap,
                "total_actual_usd": round(actual, 6),
                "total_projected_usd": round(projected, 6),
                "additional_usd": round(additional_usd, 6),
            },
        )
        self._db.add(audit)
        self._db.flush()
        return CapCheck(cap, actual, projected, additional_usd, overridden=True)
