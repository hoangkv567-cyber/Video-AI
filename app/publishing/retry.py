"""Retry policy for publisher adapters (PLAN.md §2).

Rules encoded here:

- ONLY HTTP 408, 429 and 5xx are retryable, with exponential backoff plus full
  jitter (base delay, cap, max attempts; sleep and rng are injectable so tests
  are deterministic and instant).
- Permission/policy/validation errors map to NEEDS_ACTION: adapters raise the
  typed ``PublishNeedsAction`` and the loop never retries them.
- After a timeout the adapter MUST query remote status before any retry —
  ``run_with_recovery`` calls the ``status_probe`` first and returns its result
  when the upload already exists, so a completed upload is never duplicated.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.publishing.base import (
    PublishError,
    PublishNeedsAction,
    PublishRetryable,
    PublishTimeout,
    PublishTokenExpired,
)

RETRYABLE_STATUSES = frozenset({408, 429})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    base_delay_seconds: float = 1.0
    cap_delay_seconds: float = 30.0
    max_attempts: int = 4


DEFAULT_POLICY = RetryPolicy()


def is_retryable_status(status_code: int) -> bool:
    """True only for HTTP 408, 429 and the 5xx range."""
    return status_code in RETRYABLE_STATUSES or 500 <= status_code <= 599


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None  # HTTP-date form: ignore, fall back to computed backoff


def raise_for_publish_status(response: httpx.Response, *, context: str = "") -> httpx.Response:
    """Map an HTTP response onto the typed publish errors.

    2xx/3xx pass through; 401 raises PublishTokenExpired (refresh path);
    408 raises PublishTimeout so a status probe is mandatory; 429/5xx raise
    PublishRetryable; every other 4xx is a permission,
    policy or validation failure and raises PublishNeedsAction (no retry).
    """
    code = response.status_code
    if code < 400:
        return response
    details = {"status_code": code, "context": context}
    if code == 401:
        raise PublishTokenExpired(f"credentials rejected by upstream ({context})", details=details)
    if code == 408:
        raise PublishTimeout(
            f"upstream request timed out ({context})",
            remote_status_code=code,
            retry_after=_parse_retry_after(response),
            details=details,
        )
    if is_retryable_status(code):
        raise PublishRetryable(
            f"retryable upstream error {code} ({context})",
            remote_status_code=code,
            retry_after=_parse_retry_after(response),
            details=details,
        )
    raise PublishNeedsAction(f"upstream rejected the request with {code} ({context})", details=details)


def backoff_delay(policy: RetryPolicy, attempt: int, rng: random.Random) -> float:
    """Full-jitter exponential backoff for the given zero-based retry index."""
    upper = min(policy.cap_delay_seconds, policy.base_delay_seconds * (2**attempt))
    return rng.uniform(0.0, upper)


def run_with_recovery[T](
    op: Callable[[], T],
    status_probe: Callable[[], T | None],
    *,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Run ``op`` under the retry policy with timeout->status-probe recovery.

    On a timeout the ``status_probe`` is consulted exactly once before any
    retry; a non-None probe result is returned as-is (the upload already
    exists remotely — retrying would duplicate it). PublishNeedsAction and
    any other non-retryable error propagate immediately.
    """
    policy = policy or DEFAULT_POLICY
    rng = rng if rng is not None else random.Random()
    last_error: PublishRetryable | None = None
    for attempt in range(policy.max_attempts):
        if attempt > 0:
            delay = backoff_delay(policy, attempt - 1, rng)
            if last_error is not None and last_error.retry_after is not None:
                delay = max(delay, last_error.retry_after)
            sleep(delay)
        try:
            return op()
        except (httpx.TimeoutException, PublishTimeout) as exc:
            last_error = (
                exc
                if isinstance(exc, PublishTimeout)
                else PublishTimeout(str(exc) or "request timed out")
            )
            recovered = status_probe()
            if recovered is not None:
                return recovered
        except PublishRetryable as exc:
            last_error = exc
    if last_error is None:  # pragma: no cover - loop always records an error
        raise PublishError("retry loop exhausted without recording an error")
    raise last_error


def run_with_retry[T](
    op: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Retry loop without a remote-status probe (for idempotent GET/POST-init calls)."""
    return run_with_recovery(op, lambda: None, policy=policy, sleep=sleep, rng=rng)


def run_with_token_refresh[T](op: Callable[[], T], refresh: Callable[[], object]) -> T:
    """Run ``op``; on PublishTokenExpired refresh credentials once and rerun."""
    try:
        return op()
    except PublishTokenExpired:
        refresh()
        return op()
