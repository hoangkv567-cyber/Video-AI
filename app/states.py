"""Creative pipeline state machine and related enums.

Single source of truth for lifecycle transitions; every state change must go
through `advance()` so invalid jumps raise instead of silently corrupting rows.
"""

import enum


class CreativeState(str, enum.Enum):
    DRAFT = "DRAFT"
    RESEARCHED = "RESEARCHED"
    SCRIPT_READY = "SCRIPT_READY"
    SCRIPT_APPROVED = "SCRIPT_APPROVED"
    GENERATING = "GENERATING"
    QC_REQUIRED = "QC_REQUIRED"
    READY = "READY"
    FINAL_APPROVED = "FINAL_APPROVED"
    SCHEDULED = "SCHEDULED"
    PUBLISHING = "PUBLISHING"
    PUBLISHED = "PUBLISHED"
    PARTIAL = "PARTIAL"
    NEEDS_ACTION = "NEEDS_ACTION"
    FAILED = "FAILED"


TRANSITIONS: dict[CreativeState, frozenset[CreativeState]] = {
    CreativeState.DRAFT: frozenset({CreativeState.RESEARCHED}),
    CreativeState.RESEARCHED: frozenset({CreativeState.SCRIPT_READY}),
    # New script version while still unapproved stays in SCRIPT_READY.
    CreativeState.SCRIPT_READY: frozenset(
        {CreativeState.SCRIPT_APPROVED, CreativeState.SCRIPT_READY}
    ),
    CreativeState.SCRIPT_APPROVED: frozenset({CreativeState.GENERATING}),
    CreativeState.GENERATING: frozenset({CreativeState.QC_REQUIRED, CreativeState.FAILED}),
    # Scene-level retry re-enters GENERATING; only failed scenes are re-rendered.
    CreativeState.QC_REQUIRED: frozenset(
        {CreativeState.READY, CreativeState.GENERATING, CreativeState.FAILED}
    ),
    CreativeState.READY: frozenset({CreativeState.FINAL_APPROVED}),
    CreativeState.FINAL_APPROVED: frozenset({CreativeState.SCHEDULED}),
    CreativeState.SCHEDULED: frozenset({CreativeState.PUBLISHING}),
    CreativeState.PUBLISHING: frozenset(
        {
            CreativeState.PUBLISHED,
            CreativeState.PARTIAL,
            CreativeState.NEEDS_ACTION,
            CreativeState.FAILED,
        }
    ),
    # Retry remaining targets after partial success or operator action.
    CreativeState.PARTIAL: frozenset({CreativeState.PUBLISHING, CreativeState.PUBLISHED}),
    CreativeState.NEEDS_ACTION: frozenset(
        {CreativeState.PUBLISHING, CreativeState.SCHEDULED, CreativeState.PUBLISHED}
    ),
    CreativeState.PUBLISHED: frozenset(),
    CreativeState.FAILED: frozenset(),
}

APPROVAL_STATES = frozenset({CreativeState.SCRIPT_APPROVED, CreativeState.FINAL_APPROVED})
TERMINAL_STATES = frozenset({CreativeState.PUBLISHED, CreativeState.FAILED})


class InvalidTransition(Exception):
    def __init__(self, current: CreativeState, target: CreativeState):
        self.current = current
        self.target = target
        super().__init__(f"invalid transition {current.value} -> {target.value}")


def can_transition(current: CreativeState, target: CreativeState) -> bool:
    return target in TRANSITIONS[current]


def advance(current: CreativeState, target: CreativeState) -> CreativeState:
    if not can_transition(current, target):
        raise InvalidTransition(current, target)
    return target


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PublishTargetStatus(str, enum.Enum):
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    UPLOADING = "UPLOADING"
    SCHEDULED_REMOTE = "SCHEDULED_REMOTE"
    PUBLISHED = "PUBLISHED"
    NEEDS_ACTION = "NEEDS_ACTION"
    MANUAL_BUNDLE = "MANUAL_BUNDLE"
    FAILED = "FAILED"


class Capability(str, enum.Enum):
    DIRECT = "DIRECT"
    SCHEDULE = "SCHEDULE"
    DRAFT = "DRAFT"
    MANUAL = "MANUAL"
    BLOCKED = "BLOCKED"


class Platform(str, enum.Enum):
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    ZALO = "zalo"


class Locale(str, enum.Enum):
    VI = "vi"
    EN = "en"


class Role(str, enum.Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    PUBLISHER = "publisher"
