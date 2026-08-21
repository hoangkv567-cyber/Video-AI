"""Platform registry: platform string -> Publisher factory, capability probing.

``probe_and_record`` runs a Publisher's ``probe_capability`` against a
ConnectedAccount (or anything account-shaped) and records the outcome on the
account row; auto mode is only allowed when the probe recorded DIRECT or
SCHEDULE. TikTok's DRAFT capability still requires per-publish user consent,
so it never qualifies for auto mode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.publishing.base import AccountLike, Publisher, PublishError, PublishNeedsAction, utcnow
from app.publishing.facebook import FacebookPublisher
from app.publishing.tiktok import TikTokPublisher
from app.publishing.youtube import YouTubePublisher
from app.publishing.zalo import ZaloPublisher
from app.states import Capability, Platform

PublisherFactory = Callable[..., Publisher]

_REGISTRY: dict[str, PublisherFactory] = {
    Platform.YOUTUBE.value: YouTubePublisher,
    Platform.FACEBOOK.value: FacebookPublisher,
    Platform.TIKTOK.value: TikTokPublisher,
    Platform.ZALO.value: ZaloPublisher,
}

AUTO_CAPABILITIES = frozenset({Capability.DIRECT, Capability.SCHEDULE})


def supported_platforms() -> list[str]:
    return sorted(_REGISTRY)


def register_publisher(platform: str, factory: PublisherFactory) -> None:
    """Register or replace a factory (used by tests to inject fakes)."""
    _REGISTRY[platform] = factory


def create_publisher(
    platform: str, credentials: dict[str, Any] | None = None, **kwargs: Any
) -> Publisher:
    factory = _REGISTRY.get(platform)
    if factory is None:
        raise PublishNeedsAction(
            f"unsupported platform {platform!r}",
            details={"supported": supported_platforms()},
        )
    return factory(credentials, **kwargs)


def probe_and_record(
    account: AccountLike,
    publisher: Publisher | None = None,
    **factory_kwargs: Any,
) -> Capability:
    """Probe an account's real capability and record it on the account row.

    A failed probe records BLOCKED — auto mode is only enabled when the probe
    succeeds with an auto-capable result.
    """
    pub = publisher or create_publisher(account.platform, **factory_kwargs)
    now = utcnow()
    result: dict[str, Any]
    try:
        capability = pub.probe_capability(account)
    except PublishError as exc:
        capability = Capability.BLOCKED
        result = {"capability": capability.value, "error": exc.message}
    else:
        result = {"capability": capability.value}
    account.capability = capability.value
    account.last_probe_at = now
    account.last_probe_result = {**result, "probed_at": now.isoformat()}
    return capability


def auto_mode_allowed(account: AccountLike) -> bool:
    """Auto publishing only when the recorded capability is DIRECT or SCHEDULE."""
    return account.capability in {c.value for c in AUTO_CAPABILITIES}
