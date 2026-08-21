"""Topic discovery: 72 h window (widened to 7 days), 2-source rule, weighted scoring.

Rules from PLAN.md §2:
- Search the last 72 hours; widen to 7 days when fewer than 3 solid candidates.
- Keep only topics with >= 2 independent sources (distinct registrable domains),
  prefer at least one official source.
- score = 0.35*freshness + 0.25*cross_verification + 0.25*audience_fit
  + 0.15*visual_potential (each 0..1, provider-supplied or computed).
- Return 3–5 ranked candidates with citations; persist Source rows on selection.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.ai.base import ResearchProvider, TopicCandidate
from app.models import Creative, Source

PRIMARY_WINDOW_HOURS = 72
WIDENED_WINDOW_HOURS = 7 * 24
MIN_SOLID_CANDIDATES = 3
MAX_RESULTS = 5
MIN_INDEPENDENT_SOURCES = 2

WEIGHT_FRESHNESS = 0.35
WEIGHT_CROSS_VERIFICATION = 0.25
WEIGHT_AUDIENCE_FIT = 0.25
WEIGHT_VISUAL_POTENTIAL = 0.15

# Common second-level suffixes so bbc.co.uk / vnexpress.com.vn collapse correctly.
_MULTI_PART_SLDS = frozenset({"co", "com", "net", "org", "gov", "edu", "ac"})


def registrable_domain(url: str) -> str:
    """Best-effort registrable domain: strips www., handles two-part ccTLD suffixes."""
    host = urlparse(url).netloc.lower()
    host = host.split("@")[-1].split(":")[0].removeprefix("www.")
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    if labels[-2] in _MULTI_PART_SLDS and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def independent_domains(candidate: TopicCandidate) -> set[str]:
    return {registrable_domain(s.url) for s in candidate.sources if s.url} - {""}


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class ScoredTopic:
    candidate: TopicCandidate
    score: float
    freshness: float
    cross_verification: float
    audience_fit: float
    visual_potential: float
    domains: tuple[str, ...]
    has_official: bool
    window_hours: int


class TopicDiscoveryService:
    def __init__(
        self,
        provider: ResearchProvider,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._provider = provider
        self._now = now or (lambda: datetime.now(UTC))

    # -- discovery ----------------------------------------------------------

    def discover(self, brief: str, category: str = "") -> list[ScoredTopic]:
        """Research, filter by the 2-source rule, widen if needed, score and rank."""
        candidates = self._provider.research(brief, category, PRIMARY_WINDOW_HOURS)
        window = PRIMARY_WINDOW_HOURS
        solid = [c for c in candidates if self.is_solid(c)]
        if len(solid) < MIN_SOLID_CANDIDATES:
            widened = self._provider.research(brief, category, WIDENED_WINDOW_HOURS)
            window = WIDENED_WINDOW_HOURS
            merged = _merge_candidates(candidates, widened)
            solid = [c for c in merged if self.is_solid(c)]
        scored = [self.score(c, window_hours=window) for c in solid]
        scored.sort(key=lambda s: (s.score, s.has_official, s.freshness), reverse=True)
        return scored[:MAX_RESULTS]

    def is_solid(self, candidate: TopicCandidate) -> bool:
        return len(independent_domains(candidate)) >= MIN_INDEPENDENT_SOURCES

    # -- scoring ------------------------------------------------------------

    def score(self, candidate: TopicCandidate, window_hours: int = PRIMARY_WINDOW_HOURS) -> ScoredTopic:
        domains = tuple(sorted(independent_domains(candidate)))
        has_official = any(s.is_official for s in candidate.sources)

        freshness = (
            _clamp01(candidate.freshness)
            if candidate.freshness is not None
            else self._computed_freshness(candidate.published_at)
        )
        cross = (
            _clamp01(candidate.cross_verification)
            if candidate.cross_verification is not None
            else _computed_cross_verification(len(domains), has_official)
        )
        audience = _clamp01(candidate.audience_fit) if candidate.audience_fit is not None else 0.5
        visual = (
            _clamp01(candidate.visual_potential) if candidate.visual_potential is not None else 0.5
        )

        score = round(
            WEIGHT_FRESHNESS * freshness
            + WEIGHT_CROSS_VERIFICATION * cross
            + WEIGHT_AUDIENCE_FIT * audience
            + WEIGHT_VISUAL_POTENTIAL * visual,
            6,
        )
        return ScoredTopic(
            candidate=candidate,
            score=score,
            freshness=round(freshness, 6),
            cross_verification=round(cross, 6),
            audience_fit=round(audience, 6),
            visual_potential=round(visual, 6),
            domains=domains,
            has_official=has_official,
            window_hours=window_hours,
        )

    def _computed_freshness(self, published_at: datetime | None) -> float:
        if published_at is None:
            return 0.5
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=UTC)
        age_hours = (self._now() - published_at).total_seconds() / 3600.0
        if age_hours <= 0:
            return 1.0
        return _clamp01(1.0 - age_hours / WIDENED_WINDOW_HOURS)

    # -- persistence --------------------------------------------------------

    def persist_sources(
        self, db: Session, creative: Creative, topic: ScoredTopic | TopicCandidate
    ) -> list[Source]:
        """Store Source rows (URL, title, access time, citation) for the chosen topic."""
        candidate = topic.candidate if isinstance(topic, ScoredTopic) else topic
        rows: list[Source] = []
        for info in candidate.sources:
            citation = f"{info.title or info.url} — {info.publisher or registrable_domain(info.url)} ({info.url})"
            rows.append(
                Source(
                    creative_id=creative.id,
                    url=info.url,
                    title=info.title,
                    publisher=info.publisher,
                    is_official=info.is_official,
                    accessed_at=info.accessed_at,
                    citation=citation,
                )
            )
        db.add_all(rows)
        db.flush()
        return rows


def _computed_cross_verification(domain_count: int, has_official: bool) -> float:
    base = _clamp01((domain_count - 1) / 3.0)
    if has_official:
        base = _clamp01(base + 0.25)
    return base


def _merge_candidates(
    primary: list[TopicCandidate], widened: list[TopicCandidate]
) -> list[TopicCandidate]:
    seen: set[str] = set()
    merged: list[TopicCandidate] = []
    for candidate in [*primary, *widened]:
        key = candidate.title.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    return merged
