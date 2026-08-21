"""Provider protocols for the AI layer plus deterministic fakes.

The fakes are used by unit tests and dev mode: they are fully offline,
deterministic (outputs derived from hashes of the inputs) and cheap.
"""

import copy
import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from xml.sax.saxutils import unescape

# Long-running-operation states shared by video providers.
OP_RUNNING = "running"
OP_SUCCEEDED = "succeeded"
OP_FAILED = "failed"


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInfo:
    """A citation: URL, title, publisher and the time we accessed it (UTC)."""

    url: str
    title: str = ""
    publisher: str = ""
    is_official: bool = False
    accessed_at: datetime = field(default_factory=_utcnow)


@dataclass
class TopicCandidate:
    """A researched topic with citations and optional provider-supplied scores (0..1)."""

    title: str
    summary: str = ""
    category: str = ""
    published_at: datetime | None = None
    sources: list[SourceInfo] = field(default_factory=list)
    audience_fit: float | None = None
    visual_potential: float | None = None
    freshness: float | None = None
    cross_verification: float | None = None


@dataclass(frozen=True)
class ImageResult:
    image_bytes: bytes
    model_id: str
    mime_type: str = "image/png"


@dataclass(frozen=True)
class VideoOperation:
    """State of a Veo long-running operation. `video_bytes` set when succeeded."""

    operation_name: str
    status: str  # OP_RUNNING | OP_SUCCEEDED | OP_FAILED
    video_bytes: bytes | None = None
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class Timepoint:
    mark_name: str
    seconds: float


@dataclass(frozen=True)
class TTSResult:
    audio_bytes: bytes
    timepoints: tuple[Timepoint, ...]
    duration_seconds: float
    voice: str
    audio_mime_type: str = "audio/mpeg"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ResearchProvider(Protocol):
    def research(self, brief: str, category: str, window_hours: int) -> list[TopicCandidate]:
        """Return topic candidates published within `window_hours`, with citations."""
        ...


@runtime_checkable
class ScriptProvider(Protocol):
    def generate_plan(self, topic: str, sources: list[SourceInfo], brief: str) -> dict:
        """Return a VideoPlan v1 dict (structured output, no search tools)."""
        ...

    def shorten_narrations(self, plan: dict, issues: list[dict]) -> dict:
        """Return the plan with the flagged narrations shortened; all else unchanged."""
        ...


@runtime_checkable
class ImageProvider(Protocol):
    def generate_image(
        self, prompt: str, *, model_id: str | None = None, negative_prompt: str = ""
    ) -> ImageResult: ...


@runtime_checkable
class VideoProvider(Protocol):
    def submit(
        self,
        *,
        prompt: str,
        model_id: str,
        duration_seconds: float,
        keyframe_bytes: bytes | None = None,
    ) -> str:
        """Start a long-running generation; returns the operation name."""
        ...

    def poll(self, operation_name: str) -> VideoOperation:
        """Poll an operation. On success the result carries the downloaded bytes."""
        ...


@runtime_checkable
class TTSProvider(Protocol):
    def synthesize(self, *, ssml: str, voice: str, language_code: str) -> TTSResult: ...


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class FakeResearchProvider:
    """Returns pre-seeded candidates per window; records the windows requested."""

    def __init__(
        self,
        by_window: dict[int, list[TopicCandidate]] | None = None,
        default: list[TopicCandidate] | None = None,
    ) -> None:
        self.by_window = by_window or {}
        self.default = default or []
        self.calls: list[int] = []

    def research(self, brief: str, category: str, window_hours: int) -> list[TopicCandidate]:
        self.calls.append(window_hours)
        return list(self.by_window.get(window_hours, self.default))


class FakeScriptProvider:
    """Serves a fixed plan dict; `shorten_narrations` truncates flagged narrations."""

    def __init__(self, plan: dict, shorten_to_chars: int = 60) -> None:
        self._plan = plan
        self.shorten_to_chars = shorten_to_chars
        self.generate_calls = 0
        self.shorten_calls: list[list[dict]] = []

    def generate_plan(self, topic: str, sources: list[SourceInfo], brief: str) -> dict:
        self.generate_calls += 1
        return copy.deepcopy(self._plan)

    def shorten_narrations(self, plan: dict, issues: list[dict]) -> dict:
        self.shorten_calls.append(issues)
        out = copy.deepcopy(plan)
        for issue in issues:
            locale = issue.get("locale")
            index = issue.get("scene_index")
            if locale is None or index is None:
                continue
            narration = out["locales"][locale]["narration"]
            narration[index] = narration[index][: self.shorten_to_chars].rstrip()
        return out


class FakeImageProvider:
    """Returns tiny deterministic PNG-stub bytes derived from the prompt."""

    PNG_STUB = b"\x89PNG\r\n\x1a\n"

    def __init__(self, model_id: str = "fake-image-model") -> None:
        self.model_id = model_id
        self.calls: list[str] = []

    def generate_image(
        self, prompt: str, *, model_id: str | None = None, negative_prompt: str = ""
    ) -> ImageResult:
        self.calls.append(prompt)
        model = model_id or self.model_id
        digest = hashlib.sha256(f"{model}\n{prompt}\n{negative_prompt}".encode()).digest()
        return ImageResult(image_bytes=self.PNG_STUB + digest[:16], model_id=model)


class FakeVideoProvider:
    """LRO-style fake: submit returns an operation name, poll completes deterministically.

    Polling an operation name this instance never submitted still succeeds (with a
    default 8 s duration) so a "worker restart" against a persisted operation name
    can be simulated with a fresh provider instance.
    """

    def __init__(self, pending_polls: int = 0) -> None:
        self.pending_polls = pending_polls
        self.submit_calls: list[dict] = []
        self.poll_calls: list[str] = []
        self.fail_operations: set[str] = set()
        self._ops: dict[str, dict] = {}
        self._pending: dict[str, int] = {}

    def submit(
        self,
        *,
        prompt: str,
        model_id: str,
        duration_seconds: float,
        keyframe_bytes: bytes | None = None,
    ) -> str:
        self.submit_calls.append(
            {
                "prompt": prompt,
                "model_id": model_id,
                "duration_seconds": duration_seconds,
                "has_keyframe": keyframe_bytes is not None,
            }
        )
        name = "fake-op-" + hashlib.sha256(f"{model_id}:{prompt}".encode()).hexdigest()[:12]
        self._ops[name] = {"duration_seconds": duration_seconds}
        self._pending[name] = self.pending_polls
        return name

    def poll(self, operation_name: str) -> VideoOperation:
        self.poll_calls.append(operation_name)
        if operation_name in self.fail_operations:
            return VideoOperation(operation_name, OP_FAILED, error="generation failed")
        remaining = self._pending.get(operation_name, 0)
        if remaining > 0:
            self._pending[operation_name] = remaining - 1
            return VideoOperation(operation_name, OP_RUNNING)
        meta = self._ops.get(operation_name, {"duration_seconds": 8.0})
        video = b"FAKEMP4:" + hashlib.sha256(operation_name.encode()).digest()[:16]
        return VideoOperation(
            operation_name,
            OP_SUCCEEDED,
            video_bytes=video,
            duration_seconds=float(meta["duration_seconds"]),
        )


_MARK_RE = re.compile(r'<mark name="([^"]+)"\s*/>')
_TAG_RE = re.compile(r"<[^>]+>")


class FakeTTSProvider:
    """Duration derived from character count; timepoints spread across the marks."""

    DEFAULT_RATES: dict[str, float] = {"vi": 15.0, "en": 14.0}

    def __init__(self, chars_per_second: dict[str, float] | None = None) -> None:
        self._rates = chars_per_second or dict(self.DEFAULT_RATES)
        self.calls: list[dict] = []

    def synthesize(self, *, ssml: str, voice: str, language_code: str) -> TTSResult:
        self.calls.append({"ssml": ssml, "voice": voice, "language_code": language_code})
        text = unescape(_TAG_RE.sub("", ssml)).strip()
        lang = language_code.split("-")[0].lower()
        rate = self._rates.get(lang, 14.0)
        duration = round(len(text) / rate, 4) if text else 0.0
        marks = _MARK_RE.findall(ssml)
        count = len(marks)
        timepoints = tuple(
            Timepoint(name, round(duration * (i + 1) / count, 4)) for i, name in enumerate(marks)
        )
        audio = b"FAKEAUDIO:" + hashlib.sha256(f"{voice}\n{ssml}".encode()).digest()[:16]
        return TTSResult(
            audio_bytes=audio, timepoints=timepoints, duration_seconds=duration, voice=voice
        )
