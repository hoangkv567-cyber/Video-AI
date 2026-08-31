from datetime import UTC, datetime, timedelta

import pytest

from app.ai.base import TopicCandidate
from app.ai.topic_snapshots import (
    MAX_CATEGORY_LENGTH,
    MAX_PUBLISHER_LENGTH,
    MAX_SOURCE_TITLE_LENGTH,
    MAX_SOURCE_URL_LENGTH,
    MAX_SOURCES,
    MAX_SUMMARY_LENGTH,
    MAX_TITLE_LENGTH,
    topic_candidate_from_job_result,
    topic_snapshot_from_job_result,
    validate_topic_snapshot,
)
from app.errors import ValidationFailed

NOW = datetime(2026, 8, 22, 9, 30, tzinfo=UTC)


def source(url: str, **overrides: object) -> dict:
    value = {
        "url": url,
        "title": "Source title",
        "publisher": "Publisher",
        "is_official": False,
        "accessed_at": NOW.isoformat(),
    }
    value.update(overrides)
    return value


def topic(**overrides: object) -> dict:
    value = {
        "title": " New AI model ",
        "summary": " A concise summary ",
        "category": " AI ",
        "published_at": "2026-08-21T15:00:00+07:00",
        "audience_fit": 0.8,
        "visual_potential": 0.7,
        "freshness": 0.9,
        "cross_verification": 1.0,
        "sources": [
            source(
                "https://www.openai.com/research/model",
                title=" Official announcement ",
                publisher=" OpenAI ",
                is_official=True,
            ),
            source("https://news.bbc.co.uk/technology/model"),
        ],
        # Scoring metadata serialized by the worker is intentionally ignored.
        "score": 0.88,
        "domains": ["openai.com", "bbc.co.uk"],
    }
    value.update(overrides)
    return value


def assert_invalid(raw: object, *, code: str = "invalid_topic_snapshot") -> ValidationFailed:
    with pytest.raises(ValidationFailed) as caught:
        validate_topic_snapshot(raw)
    assert caught.value.code == code
    return caught.value


def test_converts_valid_snapshot_to_domain_objects_and_normalizes_timestamps() -> None:
    candidate = topic_candidate_from_job_result({"topics": [topic()]}, 0)

    assert isinstance(candidate, TopicCandidate)
    assert candidate.title == "New AI model"
    assert candidate.summary == "A concise summary"
    assert candidate.category == "AI"
    assert candidate.published_at == datetime(2026, 8, 21, 8, 0, tzinfo=UTC)
    assert candidate.audience_fit == pytest.approx(0.8)
    assert [item.publisher for item in candidate.sources] == ["OpenAI", "Publisher"]
    assert all(item.accessed_at.tzinfo is UTC for item in candidate.sources)


def test_blank_summary_and_category_are_backward_compatible() -> None:
    snapshot = validate_topic_snapshot(topic(summary="   ", category=""))

    assert snapshot.summary == ""
    assert snapshot.category == ""


def test_only_selected_topic_is_validated() -> None:
    selected = topic_snapshot_from_job_result(
        {"topics": [{"corrupt": True}, topic(title="Second")]}, 1
    )

    assert selected.title == "Second"


@pytest.mark.parametrize(
    "result,index,code",
    [
        (None, 0, "invalid_discovery_result"),
        ({}, 0, "invalid_discovery_result"),
        ({"topics": "not-a-list"}, 0, "invalid_discovery_result"),
        ({"topics": []}, 0, "invalid_discovery_result"),
        ({"topics": [topic()] * 6}, 0, "invalid_discovery_result"),
        ({"topics": [topic()]}, -1, "invalid_topic_index"),
        ({"topics": [topic()]}, 1, "invalid_topic_index"),
        ({"topics": [topic()]}, True, "invalid_topic_index"),
        ({"topics": [topic()]}, "0", "invalid_topic_index"),
    ],
)
def test_rejects_invalid_job_result_or_index(result: object, index: object, code: str) -> None:
    with pytest.raises(ValidationFailed) as caught:
        topic_snapshot_from_job_result(result, index)  # type: ignore[arg-type]
    assert caught.value.code == code


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", ""),
        ("title", "x" * (MAX_TITLE_LENGTH + 1)),
        ("summary", "x" * (MAX_SUMMARY_LENGTH + 1)),
        ("category", "x" * (MAX_CATEGORY_LENGTH + 1)),
        ("audience_fit", -0.01),
        ("visual_potential", 1.01),
    ],
)
def test_rejects_invalid_topic_fields(field: str, value: object) -> None:
    assert_invalid(topic(**{field: value}))


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "/relative/path",
        "https://user:secret@example.com/story",
        "https://localhost/story",
        "https://127.0.0.1/story",
        "https://example.com/path with spaces",
        "https://example.com\\@evil.test/story",
        "https://example.com:99999/story",
        "https://bad_label.example/story",
        "x" * (MAX_SOURCE_URL_LENGTH + 1),
    ],
)
def test_rejects_non_http_or_non_registrable_source_urls(url: str) -> None:
    assert_invalid(topic(sources=[source(url), source("https://independent.test/story")]))


def test_rejects_two_sources_on_same_registrable_domain() -> None:
    exc = assert_invalid(
        topic(
            sources=[
                source("https://news.example.com/a"),
                source("https://www.example.com/b"),
            ]
        )
    )

    assert exc.details["issues"] == [{"field": "sources", "type": "value_error"}]


@pytest.mark.parametrize("field", ["accessed_at"])
def test_rejects_naive_required_timestamps(field: str) -> None:
    assert_invalid(
        topic(
            sources=[
                source("https://one.example/a", **{field: "2026-08-22T09:30:00"}),
                source("https://two.example/b"),
            ]
        )
    )


def test_rejects_naive_optional_published_timestamp() -> None:
    assert_invalid(topic(published_at="2026-08-22T09:30:00"))


@pytest.mark.parametrize(
    "source_overrides",
    [
        {"title": "x" * (MAX_SOURCE_TITLE_LENGTH + 1)},
        {"publisher": "x" * (MAX_PUBLISHER_LENGTH + 1)},
    ],
)
def test_rejects_oversized_source_fields(source_overrides: dict[str, object]) -> None:
    assert_invalid(
        topic(
            sources=[
                source("https://one.example/a", **source_overrides),
                source("https://two.example/b"),
            ]
        )
    )


def test_rejects_too_many_sources() -> None:
    sources = [source(f"https://source{i}.example{i}.com/story") for i in range(MAX_SOURCES + 1)]
    assert_invalid(topic(sources=sources))


def test_validation_error_details_do_not_echo_untrusted_input() -> None:
    secret = "do-not-leak-this-token"
    exc = assert_invalid(topic(title="", summary=secret))

    assert secret not in str(exc.details)


def test_accepts_two_distinct_multi_part_registrable_domains() -> None:
    candidate = topic_candidate_from_job_result(
        {
            "topics": [
                topic(
                    sources=[
                        source("https://news.bbc.co.uk/technology/a"),
                        source("https://ai.gov.vn/releases/b"),
                    ]
                )
            ]
        },
        0,
    )

    assert len(candidate.sources) == 2
    assert candidate.sources[0].accessed_at == NOW


def test_non_utc_aware_access_time_is_converted_to_utc() -> None:
    local_time = (NOW + timedelta(hours=7)).isoformat().replace("+00:00", "+07:00")
    candidate = topic_candidate_from_job_result(
        {
            "topics": [
                topic(
                    sources=[
                        source("https://one.example/a", accessed_at=local_time),
                        source("https://two.example/b"),
                    ]
                )
            ]
        },
        0,
    )

    assert candidate.sources[0].accessed_at == NOW
