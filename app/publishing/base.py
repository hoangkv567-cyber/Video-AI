"""Publisher abstract contract shared by every platform adapter (PLAN.md §2-3).

Every adapter implements exactly the publish-flow operations ``capabilities``,
``validate``, ``prepare``, ``upload``, ``finalize``, ``poll_status`` and
``refresh_credentials``, plus the account-level ``probe_capability`` that the
registry uses to record real capabilities on a ConnectedAccount.

External HTTP always goes through an injectable ``httpx.Client`` so contract
tests run offline with ``httpx.MockTransport``. Every external call is recorded
through the cost-ledger concept and the 6 USD hard cap is enforced before any
upload starts.
"""

from __future__ import annotations

import abc
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import httpx

from app.config import Settings, get_settings
from app.errors import AppError, CostCapExceeded
from app.schemas.videoplan import Disclosure
from app.states import Capability

if TYPE_CHECKING:
    from app.publishing.retry import RetryPolicy


# ---------------------------------------------------------------------------
# Typed publish errors
# ---------------------------------------------------------------------------


class PublishError(AppError):
    """Base error for the publishing layer."""

    status_code = 502
    code = "publish_error"
    retryable = False


class PublishRetryable(PublishError):
    """Transient upstream failure (HTTP 408, 429 or 5xx). Safe to retry with backoff."""

    code = "publish_retryable"
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        remote_status_code: int | None = None,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.remote_status_code = remote_status_code
        self.retry_after = retry_after


class PublishTimeout(PublishRetryable):
    """Request timed out. The adapter MUST probe remote status before any retry."""

    code = "publish_timeout"


class PublishTokenExpired(PublishError):
    """Credentials rejected (HTTP 401): triggers the refresh_credentials path."""

    code = "token_expired"
    status_code = 401


class PublishNeedsAction(PublishError):
    """Permission/policy/validation failure. Never retried; an operator must act."""

    code = "publish_needs_action"
    status_code = 422
    retryable = False


# ---------------------------------------------------------------------------
# Shared dataclasses and protocols
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PublishContext:
    """Everything an adapter needs to publish one rendition to one target."""

    creative_id: str
    rendition_id: str
    locale: str
    title: str
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    file_path: str | None = None
    file_url: str | None = None
    thumbnail_path: str | None = None
    privacy: str = "private"
    scheduled_at: datetime | None = None
    disclosure: Disclosure = field(default_factory=Disclosure)
    # Explicit user consent token for flows that must never post silently (TikTok).
    consent_token: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PublishResult:
    """Outcome of a finalize/poll operation on the remote platform."""

    remote_post_id: str | None
    remote_status: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CostLedger(Protocol):
    """Cost-ledger concept backing the CostEvent table (injectable in tests)."""

    def record(
        self,
        *,
        kind: str,
        model_id: str = "",
        units: float = 0.0,
        unit_price_usd: float = 0.0,
        amount_usd: float = 0.0,
        note: str = "",
    ) -> None: ...

    def total_spent_usd(self) -> float: ...


@runtime_checkable
class AccountLike(Protocol):
    """Structural view of ConnectedAccount used by capability probing."""

    platform: str
    capability: str
    scopes: list[Any]
    status: str
    last_probe_at: datetime | None
    last_probe_result: dict[str, Any] | None


# ---------------------------------------------------------------------------
# Abstract publisher
# ---------------------------------------------------------------------------


class Publisher(abc.ABC):
    """Abstract publisher contract. Subclasses implement one platform each."""

    platform: ClassVar[str] = "base"

    def __init__(
        self,
        credentials: dict[str, Any] | None = None,
        *,
        client: httpx.Client | None = None,
        ledger: CostLedger | None = None,
        settings: Settings | None = None,
        policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._credentials: dict[str, Any] = dict(credentials or {})
        self._client = client
        self._ledger = ledger
        self._settings = settings or get_settings()
        self._policy = policy
        self._sleep: Callable[[float], None] = sleep if sleep is not None else time.sleep
        self._rng: random.Random = rng if rng is not None else random.Random()

    # -- contract operations (PLAN.md §2) -----------------------------------

    @abc.abstractmethod
    def capabilities(self) -> frozenset[Capability]:
        raise NotImplementedError

    @abc.abstractmethod
    def validate(
        self, target: PublishContext, rendition: Mapping[str, Any] | None = None
    ) -> None:
        """Raise PublishNeedsAction when the target/rendition violates platform rules."""
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(self, ctx: PublishContext) -> dict[str, Any]:
        """Open an upload session; returns opaque session state for upload/finalize."""
        raise NotImplementedError

    @abc.abstractmethod
    def upload(self, ctx: PublishContext, session: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def finalize(self, ctx: PublishContext, session: dict[str, Any]) -> PublishResult:
        raise NotImplementedError

    @abc.abstractmethod
    def poll_status(self, remote_post_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def refresh_credentials(self) -> dict[str, Any]:
        """Refresh stored OAuth tokens; returns the new credential dict."""
        raise NotImplementedError

    @abc.abstractmethod
    def probe_capability(self, account: AccountLike) -> Capability:
        """Determine the real capability of a connected account (registry hook)."""
        raise NotImplementedError

    # -- shared plumbing -----------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(60.0))
        return self._client

    @property
    def credentials(self) -> dict[str, Any]:
        """Copy of the current credential dict (never log this un-redacted)."""
        return dict(self._credentials)

    def _send(self, do_request: Callable[[], httpx.Response]) -> httpx.Response:
        """Run one HTTP call, converting transport timeouts to PublishTimeout."""
        try:
            return do_request()
        except httpx.TimeoutException as exc:
            raise PublishTimeout(
                f"{self.platform} request timed out ({type(exc).__name__})"
            ) from exc

    def _run[T](
        self,
        op: Callable[[], T],
        probe: Callable[[], T | None] | None = None,
    ) -> T:
        """Run op with retry policy, timeout->status-probe recovery and token refresh."""
        from app.publishing.retry import run_with_recovery, run_with_token_refresh

        def _with_retry() -> T:
            return run_with_recovery(
                op,
                probe if probe is not None else lambda: None,
                policy=self._policy,
                sleep=self._sleep,
                rng=self._rng,
            )

        return run_with_token_refresh(_with_retry, self.refresh_credentials)

    def record_api_call(
        self,
        note: str,
        *,
        amount_usd: float = 0.0,
        units: float = 0.0,
        unit_price_usd: float = 0.0,
    ) -> None:
        """Record one external call on the cost ledger (publish calls are 0 USD)."""
        if self._ledger is None:
            return
        self._ledger.record(
            kind="other",
            model_id="",
            units=units,
            unit_price_usd=unit_price_usd,
            amount_usd=amount_usd,
            note=f"{self.platform}:{note}",
        )

    def check_cost_cap(self, ctx: PublishContext) -> None:
        """Block publishing when the creative already exceeds its cost cap."""
        if self._ledger is None:
            return
        hard_cap = self._settings.cost_hard_cap_usd
        raw_cap = ctx.extra.get("cost_cap_usd")
        creative_cap = float(raw_cap) if raw_cap is not None else hard_cap
        cap = min(hard_cap, creative_cap)
        spent = self._ledger.total_spent_usd()
        if spent > cap:
            raise CostCapExceeded(
                f"creative spend ${spent:.2f} exceeds cap ${cap:.2f}; publishing blocked",
                details={"spent_usd": round(spent, 4), "cap_usd": cap},
            )


def to_utc_z(dt: datetime) -> str:
    """RFC3339 UTC string with a Z suffix (YouTube publishAt format)."""
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def utcnow() -> datetime:
    return datetime.now(UTC)
