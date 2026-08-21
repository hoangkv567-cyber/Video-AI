"""Cost-ledger implementations for the publishing layer.

Every external cost-bearing call is recorded as a CostEvent row (DbCostLedger)
or an in-memory event (InMemoryCostLedger, for tests). ``ensure_within_cap``
enforces the 6 USD hard cap per creative before any spend proceeds.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.errors import CostCapExceeded
from app.models import CostEvent
from app.publishing.base import CostLedger


class InMemoryCostLedger:
    """Injectable fake ledger for offline tests."""

    def __init__(self, initial_spend_usd: float = 0.0) -> None:
        self.events: list[dict[str, Any]] = []
        self._initial_spend_usd = initial_spend_usd

    def record(
        self,
        *,
        kind: str,
        model_id: str = "",
        units: float = 0.0,
        unit_price_usd: float = 0.0,
        amount_usd: float = 0.0,
        note: str = "",
    ) -> None:
        self.events.append(
            {
                "kind": kind,
                "model_id": model_id,
                "units": units,
                "unit_price_usd": unit_price_usd,
                "amount_usd": amount_usd,
                "note": note,
            }
        )

    def total_spent_usd(self) -> float:
        return self._initial_spend_usd + sum(e["amount_usd"] for e in self.events)


class DbCostLedger:
    """CostEvent-backed ledger scoped to one creative."""

    def __init__(self, session: Session, creative_id: str, *, job_id: str | None = None) -> None:
        self._session = session
        self._creative_id = creative_id
        self._job_id = job_id

    def record(
        self,
        *,
        kind: str,
        model_id: str = "",
        units: float = 0.0,
        unit_price_usd: float = 0.0,
        amount_usd: float = 0.0,
        note: str = "",
    ) -> None:
        self._session.add(
            CostEvent(
                creative_id=self._creative_id,
                kind=kind,
                model_id=model_id,
                units=units,
                unit_price_usd=unit_price_usd,
                amount_usd=amount_usd,
                projected=False,
                job_id=self._job_id,
                note=note,
            )
        )
        self._session.flush()

    def total_spent_usd(self) -> float:
        stmt = select(func.coalesce(func.sum(CostEvent.amount_usd), 0.0)).where(
            CostEvent.creative_id == self._creative_id,
            CostEvent.projected.is_(False),
        )
        return float(self._session.execute(stmt).scalar_one())


def ensure_within_cap(
    ledger: CostLedger,
    *,
    creative_cap_usd: float | None = None,
    hard_cap_usd: float | None = None,
    projected_usd: float = 0.0,
) -> None:
    """Raise CostCapExceeded when actual + projected spend passes the effective cap.

    The effective cap is min(creative.cost_cap_usd, settings.cost_hard_cap_usd).
    """
    hard = hard_cap_usd if hard_cap_usd is not None else get_settings().cost_hard_cap_usd
    cap = hard if creative_cap_usd is None else min(hard, creative_cap_usd)
    total = ledger.total_spent_usd() + projected_usd
    if total > cap:
        raise CostCapExceeded(
            f"cost ${total:.2f} exceeds the effective cap ${cap:.2f}",
            details={"total_usd": round(total, 4), "cap_usd": cap, "projected_usd": projected_usd},
        )
