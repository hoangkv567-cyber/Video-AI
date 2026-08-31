"""Validation boundary for topic snapshots persisted in discovery jobs.

Discovery output is stored as JSON in ``Job.result`` and can outlive the
provider response (or even the code version) that produced it.  Treat that
JSON as untrusted when a user selects a topic: validate only the selected
entry and reconstruct the small domain objects used by the scripting layer.
"""

import re
from collections.abc import Mapping
from datetime import UTC
from ipaddress import ip_address
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.ai.base import SourceInfo, TopicCandidate
from app.ai.topics import registrable_domain
from app.errors import ValidationFailed

MAX_TOPICS = 5
MAX_SOURCES = 20
MAX_TITLE_LENGTH = 500
MAX_SUMMARY_LENGTH = 4_000
MAX_CATEGORY_LENGTH = 100
MAX_SOURCE_URL_LENGTH = 2_048
MAX_SOURCE_TITLE_LENGTH = 1_000
MAX_PUBLISHER_LENGTH = 255

_DOMAIN_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")

TrimmedTitle = Annotated[str, Field(min_length=1, max_length=MAX_TITLE_LENGTH)]
TrimmedSummary = Annotated[str, Field(max_length=MAX_SUMMARY_LENGTH)]
TrimmedCategory = Annotated[str, Field(max_length=MAX_CATEGORY_LENGTH)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


def validate_http_source_url(value: str) -> str:
    """Validate a persisted/user-supplied source URL for safe display and citation.

    Sources are rendered as clickable links in the operator dashboard.  Keeping
    this validator public lets both discovery snapshots and direct creative
    input share the same scheme/host boundary instead of accepting executable
    ``javascript:`` links or local/IP targets.
    """
    value = value.strip()
    if not value or len(value) > MAX_SOURCE_URL_LENGTH:
        raise ValueError(f"URL must contain 1..{MAX_SOURCE_URL_LENGTH} characters")
    if any(char.isspace() for char in value) or "\\" in value:
        raise ValueError("URL must not contain whitespace or backslashes")

    try:
        parsed = urlsplit(value)
        port = parsed.port  # Forces validation of malformed/out-of-range ports.
    except ValueError as exc:
        raise ValueError("URL is malformed") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("URL scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not contain credentials")
    if parsed.hostname is None:
        raise ValueError("URL must contain a hostname")
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("URL port is invalid")

    host = parsed.hostname.rstrip(".")
    try:
        ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("URL hostname must be a registrable domain, not an IP address")

    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("URL hostname is invalid") from exc
    labels = ascii_host.split(".")
    if len(labels) < 2 or len(ascii_host) > 253 or any(
        not _DOMAIN_LABEL.fullmatch(label) for label in labels
    ):
        raise ValueError("URL hostname must be a registrable domain")
    return value


class SourceSnapshot(BaseModel):
    """Validated source as serialized by the topic-discovery worker."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    url: str
    title: str = Field(default="", max_length=MAX_SOURCE_TITLE_LENGTH)
    publisher: str = Field(default="", max_length=MAX_PUBLISHER_LENGTH)
    is_official: bool = False
    accessed_at: AwareDatetime

    _http_url = field_validator("url")(validate_http_source_url)

    def to_source_info(self) -> SourceInfo:
        return SourceInfo(
            url=self.url,
            title=self.title,
            publisher=self.publisher,
            is_official=self.is_official,
            accessed_at=self.accessed_at.astimezone(UTC),
        )


class TopicSnapshot(BaseModel):
    """The trusted, typed form of one selectable discovery result."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: TrimmedTitle
    # Empty summary/category remain valid for compatibility with older providers.
    summary: TrimmedSummary = ""
    category: TrimmedCategory = ""
    published_at: AwareDatetime | None = None
    sources: list[SourceSnapshot] = Field(min_length=2, max_length=MAX_SOURCES)
    audience_fit: Score | None = None
    visual_potential: Score | None = None
    freshness: Score | None = None
    cross_verification: Score | None = None

    @field_validator("sources")
    @classmethod
    def require_independent_domains(
        cls, sources: list[SourceSnapshot]
    ) -> list[SourceSnapshot]:
        domains = {registrable_domain(source.url) for source in sources}
        if len(domains) < 2:
            raise ValueError("topic must cite at least 2 independent registrable domains")
        return sources

    def to_topic_candidate(self) -> TopicCandidate:
        return TopicCandidate(
            title=self.title,
            summary=self.summary,
            category=self.category,
            published_at=(self.published_at.astimezone(UTC) if self.published_at else None),
            sources=[source.to_source_info() for source in self.sources],
            audience_fit=self.audience_fit,
            visual_potential=self.visual_potential,
            freshness=self.freshness,
            cross_verification=self.cross_verification,
        )


def validate_topic_snapshot(raw: Any) -> TopicSnapshot:
    """Validate one raw snapshot and raise the API's safe 422 error on failure."""
    try:
        return TopicSnapshot.model_validate(raw)
    except ValidationError as exc:
        issues = [
            {
                "field": ".".join(str(part) for part in error["loc"]),
                "type": error["type"],
            }
            for error in exc.errors(include_url=False, include_context=False, include_input=False)
        ]
        raise ValidationFailed(
            "selected topic snapshot is invalid",
            code="invalid_topic_snapshot",
            details={"issues": issues},
        ) from exc


def topic_snapshot_from_job_result(
    result: Mapping[str, Any] | None, topic_index: int
) -> TopicSnapshot:
    """Select and validate one topic from a discovery job's JSON result."""
    if not isinstance(result, Mapping):
        raise ValidationFailed(
            "discovery job has no valid result",
            code="invalid_discovery_result",
        )
    topics = result.get("topics")
    if not isinstance(topics, list) or not 1 <= len(topics) <= MAX_TOPICS:
        raise ValidationFailed(
            "discovery result must contain between 1 and 5 topics",
            code="invalid_discovery_result",
        )
    if isinstance(topic_index, bool) or not isinstance(topic_index, int):
        raise ValidationFailed("topic_index must be an integer", code="invalid_topic_index")
    if topic_index < 0 or topic_index >= len(topics):
        raise ValidationFailed("topic_index is out of range", code="invalid_topic_index")
    return validate_topic_snapshot(topics[topic_index])


def topic_candidate_from_job_result(
    result: Mapping[str, Any] | None, topic_index: int
) -> TopicCandidate:
    """Return provider-neutral objects ready for source/script persistence."""
    return topic_snapshot_from_job_result(result, topic_index).to_topic_candidate()
