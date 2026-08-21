"""TopicDiscoveryService: 2-source rule, window widening, scoring, persistence."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.ai.base import FakeResearchProvider, SourceInfo, TopicCandidate
from app.ai.gemini import extract_json, parse_topic_candidates
from app.ai.topics import (
    PRIMARY_WINDOW_HOURS,
    WIDENED_WINDOW_HOURS,
    ScoredTopic,
    TopicDiscoveryService,
    independent_domains,
    registrable_domain,
)
from app.errors import UpstreamError
from app.models import Creative, Source

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def fixed_now() -> datetime:
    return NOW


def make_candidate(
    title: str,
    domains: list[str],
    *,
    official_first: bool = True,
    hours_old: float = 10.0,
    audience_fit: float | None = 0.8,
    visual_potential: float | None = 0.6,
    freshness: float | None = None,
    cross_verification: float | None = None,
) -> TopicCandidate:
    sources = [
        SourceInfo(
            url=f"https://{domain}/articles/{title.lower().replace(' ', '-')}",
            title=f"{title} on {domain}",
            publisher=domain,
            is_official=official_first and i == 0,
            accessed_at=NOW,
        )
        for i, domain in enumerate(domains)
    ]
    return TopicCandidate(
        title=title,
        summary=f"Summary of {title}",
        published_at=NOW - timedelta(hours=hours_old),
        sources=sources,
        audience_fit=audience_fit,
        visual_potential=visual_potential,
        freshness=freshness,
        cross_verification=cross_verification,
    )


class TestRegistrableDomain:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.example.com/x", "example.com"),
            ("https://news.bbc.co.uk/tech", "bbc.co.uk"),
            ("https://sub.blog.vnexpress.net/a", "vnexpress.net"),
            ("https://example.com:8080/path", "example.com"),
            ("https://baochinhphu.gov.vn/story", "baochinhphu.gov.vn"),
        ],
    )
    def test_cases(self, url: str, expected: str) -> None:
        assert registrable_domain(url) == expected

    def test_same_registrable_domain_not_independent(self) -> None:
        candidate = TopicCandidate(
            title="dup",
            sources=[
                SourceInfo(url="https://www.foo.com/a"),
                SourceInfo(url="https://foo.com/b"),
            ],
        )
        assert independent_domains(candidate) == {"foo.com"}


class TestTwoSourceRule:
    def test_single_domain_candidates_are_dropped(self) -> None:
        provider = FakeResearchProvider(
            by_window={
                PRIMARY_WINDOW_HOURS: [
                    make_candidate("Solid A", ["openai.com", "techcrunch.com"]),
                    make_candidate("Solid B", ["deepmind.com", "theverge.com"]),
                    make_candidate("Solid C", ["meta.com", "wired.com"]),
                    make_candidate("Weak", ["blog.foo.com", "www.foo.com"]),
                ]
            }
        )
        service = TopicDiscoveryService(provider, now=fixed_now)
        results = service.discover("ai news")
        titles = [r.candidate.title for r in results]
        assert "Weak" not in titles
        assert len(results) == 3
        assert provider.calls == [PRIMARY_WINDOW_HOURS]


class TestWindowWidening:
    def test_widens_to_7_days_when_fewer_than_3_solid(self) -> None:
        primary = [
            make_candidate("Fresh A", ["openai.com", "techcrunch.com"], hours_old=5),
            make_candidate("Fresh B", ["deepmind.com", "theverge.com"], hours_old=8),
        ]
        widened = primary + [
            make_candidate("Older C", ["meta.com", "wired.com"], hours_old=100),
            make_candidate("Older D", ["anthropic.com", "arstechnica.com"], hours_old=120),
        ]
        provider = FakeResearchProvider(
            by_window={PRIMARY_WINDOW_HOURS: primary, WIDENED_WINDOW_HOURS: widened}
        )
        service = TopicDiscoveryService(provider, now=fixed_now)
        results = service.discover("ai news")

        assert provider.calls == [PRIMARY_WINDOW_HOURS, WIDENED_WINDOW_HOURS]
        titles = {r.candidate.title for r in results}
        assert titles == {"Fresh A", "Fresh B", "Older C", "Older D"}
        assert all(r.window_hours == WIDENED_WINDOW_HOURS for r in results)

    def test_no_widening_when_enough_solid(self) -> None:
        provider = FakeResearchProvider(
            by_window={
                PRIMARY_WINDOW_HOURS: [
                    make_candidate(f"Topic {i}", ["openai.com", "techcrunch.com"])
                    for i in range(3)
                ]
            }
        )
        service = TopicDiscoveryService(provider, now=fixed_now)
        service.discover("ai news")
        assert provider.calls == [PRIMARY_WINDOW_HOURS]


class TestScoring:
    def test_weighted_formula_with_provider_scores(self) -> None:
        candidate = make_candidate(
            "Scored",
            ["openai.com", "techcrunch.com"],
            freshness=0.8,
            cross_verification=0.6,
            audience_fit=1.0,
            visual_potential=0.5,
        )
        service = TopicDiscoveryService(FakeResearchProvider(), now=fixed_now)
        scored = service.score(candidate)
        expected = 0.35 * 0.8 + 0.25 * 0.6 + 0.25 * 1.0 + 0.15 * 0.5
        assert scored.score == pytest.approx(expected)

    def test_computed_freshness_decays_over_the_7_day_window(self) -> None:
        service = TopicDiscoveryService(FakeResearchProvider(), now=fixed_now)
        fresh = service.score(make_candidate("Now", ["a.com", "b.com"], hours_old=0))
        stale = service.score(make_candidate("Old", ["a.com", "b.com"], hours_old=168))
        assert fresh.freshness == pytest.approx(1.0)
        assert stale.freshness == pytest.approx(0.0)

    def test_computed_cross_verification_rewards_domains_and_official(self) -> None:
        service = TopicDiscoveryService(FakeResearchProvider(), now=fixed_now)
        unofficial = service.score(
            make_candidate("U", ["a.com", "b.com"], official_first=False)
        )
        official = service.score(make_candidate("O", ["a.com", "b.com"], official_first=True))
        assert unofficial.cross_verification == pytest.approx(1 / 3)
        assert official.cross_verification == pytest.approx(1 / 3 + 0.25)
        assert official.has_official
        assert not unofficial.has_official

    def test_returns_at_most_5_ranked_desc(self) -> None:
        candidates = [
            make_candidate(f"T{i}", ["openai.com", "techcrunch.com"], audience_fit=i / 10)
            for i in range(7)
        ]
        provider = FakeResearchProvider(by_window={PRIMARY_WINDOW_HOURS: candidates})
        service = TopicDiscoveryService(provider, now=fixed_now)
        results = service.discover("ai news")
        assert len(results) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestPersistSources:
    def test_source_rows_written_for_chosen_topic(
        self, db_session: Session, creative: Creative
    ) -> None:
        service = TopicDiscoveryService(FakeResearchProvider(), now=fixed_now)
        scored: ScoredTopic = service.score(
            make_candidate("Chosen", ["openai.com", "techcrunch.com"])
        )
        rows = service.persist_sources(db_session, creative, scored)
        db_session.commit()

        stored = db_session.query(Source).filter_by(creative_id=creative.id).all()
        assert len(stored) == len(rows) == 2
        by_publisher = {s.publisher: s for s in stored}
        assert by_publisher["openai.com"].is_official
        assert not by_publisher["techcrunch.com"].is_official
        for source in stored:
            assert source.url.startswith("https://")
            assert source.accessed_at is not None
            assert source.citation


class TestGeminiParsing:
    """Offline checks of the pure gemini.py helpers."""

    def test_extract_json_strips_code_fences(self) -> None:
        assert extract_json('```json\n{"topics": []}\n```') == {"topics": []}

    def test_extract_json_rejects_non_json(self) -> None:
        with pytest.raises(UpstreamError):
            extract_json("no json here")

    def test_parse_topic_candidates_maps_fields(self) -> None:
        data = {
            "topics": [
                {
                    "title": "New chip",
                    "summary": "A chip",
                    "published_at": "2026-08-21T00:00:00Z",
                    "audience_fit": 0.9,
                    "visual_potential": 1.7,  # clamped to 1.0
                    "sources": [
                        {"url": "https://nvidia.com/x", "is_official": True},
                        {"url": "https://reuters.com/y", "title": "Reuters"},
                    ],
                },
                {"summary": "missing title is skipped"},
            ]
        }
        candidates = parse_topic_candidates(data, accessed_at=NOW)
        assert len(candidates) == 1
        topic = candidates[0]
        assert topic.title == "New chip"
        assert topic.published_at == datetime(2026, 8, 21, tzinfo=UTC)
        assert topic.visual_potential == pytest.approx(1.0)
        assert [s.accessed_at for s in topic.sources] == [NOW, NOW]
        assert independent_domains(topic) == {"nvidia.com", "reuters.com"}
