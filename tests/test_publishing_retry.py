"""Retry policy tests: 408/429/5xx only, backoff+jitter, timeout->probe recovery."""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import random  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from app.publishing.base import (  # noqa: E402
    PublishNeedsAction,
    PublishResult,
    PublishRetryable,
    PublishTimeout,
    PublishTokenExpired,
)
from app.publishing.retry import (  # noqa: E402
    RetryPolicy,
    backoff_delay,
    is_retryable_status,
    raise_for_publish_status,
    run_with_recovery,
    run_with_retry,
    run_with_token_refresh,
)

POLICY = RetryPolicy(base_delay_seconds=0.01, cap_delay_seconds=0.05, max_attempts=4)


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status_code, headers=headers, request=httpx.Request("GET", "https://api.example/x")
    )


class TestClassification:
    @pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 599])
    def test_retryable_statuses(self, code: int) -> None:
        assert is_retryable_status(code) is True

    @pytest.mark.parametrize("code", [200, 302, 400, 401, 403, 404, 409, 422, 451])
    def test_non_retryable_statuses(self, code: int) -> None:
        assert is_retryable_status(code) is False

    def test_success_passes_through(self) -> None:
        resp = _response(200)
        assert raise_for_publish_status(resp) is resp

    def test_429_maps_to_retryable_with_retry_after(self) -> None:
        with pytest.raises(PublishRetryable) as excinfo:
            raise_for_publish_status(_response(429, {"Retry-After": "7"}))
        assert excinfo.value.retryable is True
        assert excinfo.value.remote_status_code == 429
        assert excinfo.value.retry_after == 7.0

    def test_408_maps_to_timeout_that_requires_a_probe(self) -> None:
        with pytest.raises(PublishTimeout) as excinfo:
            raise_for_publish_status(_response(408), context="mutating.request")
        assert excinfo.value.retryable is True
        assert excinfo.value.remote_status_code == 408

    def test_401_maps_to_token_expired(self) -> None:
        with pytest.raises(PublishTokenExpired):
            raise_for_publish_status(_response(401))

    @pytest.mark.parametrize("code", [400, 403, 404, 422])
    def test_policy_and_validation_errors_map_to_needs_action(self, code: int) -> None:
        with pytest.raises(PublishNeedsAction) as excinfo:
            raise_for_publish_status(_response(code), context="policy check")
        assert excinfo.value.retryable is False
        assert excinfo.value.details["status_code"] == code


class TestBackoff:
    def test_full_jitter_within_exponential_bounds(self) -> None:
        rng = random.Random(1234)
        for attempt in range(6):
            upper = min(POLICY.cap_delay_seconds, POLICY.base_delay_seconds * (2**attempt))
            for _ in range(50):
                delay = backoff_delay(POLICY, attempt, rng)
                assert 0.0 <= delay <= upper

    def test_cap_is_respected(self) -> None:
        class UpperRng(random.Random):
            def uniform(self, a: float, b: float) -> float:
                return b

        assert backoff_delay(POLICY, 30, UpperRng()) == POLICY.cap_delay_seconds


class TestRetryLoop:
    def test_429_twice_then_success_backs_off_with_jitter(self) -> None:
        sleeps = SleepRecorder()
        attempts = {"n": 0}

        def op() -> str:
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise PublishRetryable("throttled", remote_status_code=429)
            return "ok"

        result = run_with_retry(op, policy=POLICY, sleep=sleeps, rng=random.Random(42))
        assert result == "ok"
        assert attempts["n"] == 3
        # The exact jitter sequence is reproducible from the seeded rng.
        replay = random.Random(42)
        expected = [
            replay.uniform(0.0, min(POLICY.cap_delay_seconds, POLICY.base_delay_seconds * 1)),
            replay.uniform(0.0, min(POLICY.cap_delay_seconds, POLICY.base_delay_seconds * 2)),
        ]
        assert sleeps.calls == pytest.approx(expected)

    def test_retry_after_hint_raises_the_delay_floor(self) -> None:
        sleeps = SleepRecorder()
        attempts = {"n": 0}

        def op() -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise PublishRetryable("throttled", remote_status_code=429, retry_after=7.5)
            return "ok"

        assert run_with_retry(op, policy=POLICY, sleep=sleeps, rng=random.Random(0)) == "ok"
        assert len(sleeps.calls) == 1
        assert sleeps.calls[0] >= 7.5

    def test_5xx_exhausts_attempts_then_raises(self) -> None:
        sleeps = SleepRecorder()
        attempts = {"n": 0}

        def op() -> str:
            attempts["n"] += 1
            raise PublishRetryable("server error", remote_status_code=503)

        with pytest.raises(PublishRetryable):
            run_with_retry(op, policy=POLICY, sleep=sleeps, rng=random.Random(0))
        assert attempts["n"] == POLICY.max_attempts
        assert len(sleeps.calls) == POLICY.max_attempts - 1

    def test_needs_action_is_never_retried(self) -> None:
        sleeps = SleepRecorder()
        attempts = {"n": 0}

        def op() -> str:
            attempts["n"] += 1
            raise PublishNeedsAction("permission denied")

        with pytest.raises(PublishNeedsAction):
            run_with_retry(op, policy=POLICY, sleep=sleeps, rng=random.Random(0))
        assert attempts["n"] == 1
        assert sleeps.calls == []


class TestTimeoutRecovery:
    def test_ambiguous_probe_failure_prevents_blind_retry(self) -> None:
        op_calls = {"n": 0}

        def op() -> str:
            op_calls["n"] += 1
            raise PublishTimeout("remote outcome unknown")

        def fail_closed_probe() -> str | None:
            raise PublishNeedsAction("operator must reconcile the remote object")

        with pytest.raises(PublishNeedsAction):
            run_with_recovery(
                op,
                fail_closed_probe,
                policy=POLICY,
                sleep=SleepRecorder(),
                rng=random.Random(0),
            )
        assert op_calls["n"] == 1

    def test_timeout_triggers_exactly_one_probe_and_no_duplicate_upload(self) -> None:
        """Probe says the upload exists -> its result is used, op is NOT re-run."""
        sleeps = SleepRecorder()
        op_calls = {"n": 0}
        probe_calls = {"n": 0}
        existing = PublishResult(remote_post_id="vid-1", remote_status={"uploadStatus": "uploaded"})

        def op() -> PublishResult:
            op_calls["n"] += 1
            raise PublishTimeout("upload timed out")

        def probe() -> PublishResult | None:
            probe_calls["n"] += 1
            return existing

        result = run_with_recovery(op, probe, policy=POLICY, sleep=sleeps, rng=random.Random(0))
        assert result is existing
        assert op_calls["n"] == 1  # no duplicate upload
        assert probe_calls["n"] == 1  # exactly one status probe
        assert sleeps.calls == []  # recovery without any backoff sleep

    def test_timeout_with_negative_probe_retries_the_upload(self) -> None:
        sleeps = SleepRecorder()
        op_calls = {"n": 0}
        probe_calls = {"n": 0}

        def op() -> str:
            op_calls["n"] += 1
            if op_calls["n"] == 1:
                raise PublishTimeout("upload timed out")
            return "uploaded"

        def probe() -> str | None:
            probe_calls["n"] += 1
            return None

        result = run_with_recovery(op, probe, policy=POLICY, sleep=sleeps, rng=random.Random(0))
        assert result == "uploaded"
        assert op_calls["n"] == 2
        assert probe_calls["n"] == 1
        assert len(sleeps.calls) == 1

    def test_raw_httpx_timeout_is_also_probed(self) -> None:
        probe_calls = {"n": 0}
        recovered = {"id": "vid-2"}

        def op() -> dict:
            raise httpx.ReadTimeout("simulated read timeout")

        def probe() -> dict | None:
            probe_calls["n"] += 1
            return recovered

        result = run_with_recovery(
            op, probe, policy=POLICY, sleep=SleepRecorder(), rng=random.Random(0)
        )
        assert result is recovered
        assert probe_calls["n"] == 1


class TestTokenRefresh:
    def test_expired_token_refreshes_once_and_reruns(self) -> None:
        calls = {"op": 0, "refresh": 0}

        def op() -> str:
            calls["op"] += 1
            if calls["op"] == 1:
                raise PublishTokenExpired("401 from upstream")
            return "ok"

        def refresh() -> None:
            calls["refresh"] += 1

        assert run_with_token_refresh(op, refresh) == "ok"
        assert calls == {"op": 2, "refresh": 1}

    def test_no_refresh_on_success(self) -> None:
        calls = {"refresh": 0}

        def refresh() -> None:
            calls["refresh"] += 1

        assert run_with_token_refresh(lambda: "ok", refresh) == "ok"
        assert calls["refresh"] == 0
