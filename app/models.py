"""SQLAlchemy ORM models — the core tables from PLAN.md §2.

UUIDs are stored as 36-char strings for cross-database (Postgres/SQLite) compat.
All timestamps are UTC; display conversion to Asia/Ho_Chi_Minh happens in the UI.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.states import (
    Capability,
    CreativeState,
    JobStatus,
    Platform,
    PublishTargetStatus,
    Role,
)


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default=Role.EDITOR.value)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Campaign(Base, TimestampMixin):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255))
    brief: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(100), default="")
    mode: Mapped[str] = mapped_column(String(16), default="manual")  # manual | auto
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    creatives: Mapped[list["Creative"]] = relationship(back_populates="campaign")


class Creative(Base, TimestampMixin):
    """One visual master + its per-locale renditions."""

    __tablename__ = "creatives"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    campaign_id: Mapped[str] = mapped_column(ForeignKey("campaigns.id"), index=True)
    state: Mapped[str] = mapped_column(String(32), default=CreativeState.DRAFT.value, index=True)
    topic_title: Mapped[str] = mapped_column(String(500), default="")
    angle: Mapped[str] = mapped_column(Text, default="")
    mode: Mapped[str] = mapped_column(String(16), default="manual")
    cost_cap_usd: Mapped[float] = mapped_column(Float, default=6.0)
    cost_cap_override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    campaign: Mapped[Campaign] = relationship(back_populates="creatives")
    sources: Mapped[list["Source"]] = relationship(back_populates="creative")
    script_versions: Mapped[list["ScriptVersion"]] = relationship(back_populates="creative")
    scenes: Mapped[list["Scene"]] = relationship(back_populates="creative")
    assets: Mapped[list["Asset"]] = relationship(back_populates="creative")
    renditions: Mapped[list["Rendition"]] = relationship(back_populates="creative")
    cost_events: Mapped[list["CostEvent"]] = relationship(back_populates="creative")


class Source(Base, TimestampMixin):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    creative_id: Mapped[str] = mapped_column(ForeignKey("creatives.id"), index=True)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text, default="")
    publisher: Mapped[str] = mapped_column(String(255), default="")
    is_official: Mapped[bool] = mapped_column(Boolean, default=False)
    accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    citation: Mapped[str] = mapped_column(Text, default="")

    creative: Mapped[Creative] = relationship(back_populates="sources")


class ScriptVersion(Base, TimestampMixin):
    __tablename__ = "script_versions"
    __table_args__ = (UniqueConstraint("creative_id", "version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    creative_id: Mapped[str] = mapped_column(ForeignKey("creatives.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    video_plan: Mapped[dict] = mapped_column(JSON)  # validated VideoPlan v1
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    creative: Mapped[Creative] = relationship(back_populates="script_versions")


class Scene(Base, TimestampMixin):
    __tablename__ = "scenes"
    __table_args__ = (UniqueConstraint("creative_id", "script_version_id", "index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    creative_id: Mapped[str] = mapped_column(ForeignKey("creatives.id"), index=True)
    script_version_id: Mapped[str] = mapped_column(ForeignKey("script_versions.id"))
    index: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[float] = mapped_column(Float, default=8.0)
    keyframe_prompt_en: Mapped[str] = mapped_column(Text, default="")
    visual_prompt_en: Mapped[str] = mapped_column(Text, default="")
    negative_prompt_en: Mapped[str] = mapped_column(Text, default="")
    continuity_note: Mapped[str] = mapped_column(Text, default="")
    fact_ids: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|generating|done|failed
    is_hero: Mapped[bool] = mapped_column(Boolean, default=False)
    last_error: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    creative: Mapped[Creative] = relationship(back_populates="scenes")


class Asset(Base):
    """Immutable stored artifact with checksum, provenance and probe result."""

    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    creative_id: Mapped[str] = mapped_column(ForeignKey("creatives.id"), index=True)
    scene_id: Mapped[str | None] = mapped_column(ForeignKey("scenes.id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))  # keyframe|clip|voice|master|derivative|thumbnail|caption|styleboard
    locale: Mapped[str | None] = mapped_column(String(8), nullable=True)
    platform: Mapped[str | None] = mapped_column(String(16), nullable=True)
    storage_key: Mapped[str] = mapped_column(String(500))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    ffprobe: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    creative: Mapped[Creative] = relationship(back_populates="assets")


class Rendition(Base, TimestampMixin):
    """Per-locale finished video sharing the same visual master."""

    __tablename__ = "renditions"
    __table_args__ = (UniqueConstraint("creative_id", "locale"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    creative_id: Mapped[str] = mapped_column(ForeignKey("creatives.id"), index=True)
    locale: Mapped[str] = mapped_column(String(8))
    master_asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"), nullable=True)
    thumbnail_asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"), nullable=True)
    srt_asset_id: Mapped[str | None] = mapped_column(ForeignKey("assets.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(500), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    hashtags: Mapped[list] = mapped_column(JSON, default=list)
    qc_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    creative: Mapped[Creative] = relationship(back_populates="renditions")


class CostEvent(Base):
    __tablename__ = "cost_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    creative_id: Mapped[str] = mapped_column(ForeignKey("creatives.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # gemini_text|gemini_image|veo|tts|other
    model_id: Mapped[str] = mapped_column(String(100), default="")
    units: Mapped[float] = mapped_column(Float, default=0.0)
    unit_price_usd: Mapped[float] = mapped_column(Float, default=0.0)
    amount_usd: Mapped[float] = mapped_column(Float, default=0.0)
    projected: Mapped[bool] = mapped_column(Boolean, default=False)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    note: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    creative: Mapped[Creative] = relationship(back_populates="cost_events")


class ConnectedAccount(Base, TimestampMixin):
    __tablename__ = "connected_accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    platform: Mapped[str] = mapped_column(String(16), index=True)
    display_name: Mapped[str] = mapped_column(String(255), default="")
    remote_account_id: Mapped[str] = mapped_column(String(255), default="")
    capability: Mapped[str] = mapped_column(String(16), default=Capability.MANUAL.value)
    encrypted_credentials: Mapped[str] = mapped_column(Text, default="")
    scopes: Mapped[list] = mapped_column(JSON, default=list)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")  # active|expired|revoked
    last_probe_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_probe_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index(
            "uq_jobs_active_veo_operation",
            "idempotency_key",
            unique=True,
            postgresql_where=text("kind = 'veo_operation' AND status = 'RUNNING'"),
            sqlite_where=text("kind = 'veo_operation' AND status = 'RUNNING'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(64), index=True)  # discover|generate|render|publish|...
    queue: Mapped[str] = mapped_column(String(16), default="ai")  # ai|render|publish
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    creative_id: Mapped[str | None] = mapped_column(ForeignKey("creatives.id"), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    dispatched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PublishTarget(Base, TimestampMixin):
    __tablename__ = "publish_targets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    creative_id: Mapped[str] = mapped_column(ForeignKey("creatives.id"), index=True)
    rendition_id: Mapped[str] = mapped_column(ForeignKey("renditions.id"))
    platform: Mapped[str] = mapped_column(String(16))
    connected_account_id: Mapped[str | None] = mapped_column(
        ForeignKey("connected_accounts.id"), nullable=True
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    privacy: Mapped[str] = mapped_column(String(16), default="private")  # private|public|unlisted
    status: Mapped[str] = mapped_column(
        String(24), default=PublishTargetStatus.PENDING.value, index=True
    )
    remote_post_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    remote_status: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    last_error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    bundle_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Row-level claim used by the scheduler's FOR UPDATE SKIP LOCKED sweep.
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PublishAttempt(Base):
    __tablename__ = "publish_attempts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    target_id: Mapped[str] = mapped_column(ForeignKey("publish_targets.id"), index=True)
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(24), default="running")
    response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    actor_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    actor_kind: Mapped[str] = mapped_column(String(16), default="user")  # user|system
    action: Mapped[str] = mapped_column(String(100), index=True)
    entity_type: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[str] = mapped_column(String(36), index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"
    __table_args__ = (UniqueConstraint("key", "endpoint"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    key: Mapped[str] = mapped_column(String(255), index=True)
    endpoint: Mapped[str] = mapped_column(String(255))
    request_hash: Mapped[str] = mapped_column(String(64), default="")
    response_status: Mapped[int] = mapped_column(Integer, default=202)
    response_body: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


PLATFORMS = [p.value for p in Platform]
