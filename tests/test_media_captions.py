"""Tests for caption segmentation, SRT golden output, and the ASS style block."""

import pytest

from app.media.captions import (
    MAX_LINE_CHARS,
    CaptionEvent,
    ass_style_block,
    build_ass,
    build_srt,
    events_from_timepoints,
    format_ass_timestamp,
    format_srt_timestamp,
    segment_caption_blocks,
    wrap_caption_lines,
)


class TestWrap:
    def test_lines_respect_42_char_limit(self):
        text = (
            "Google vua cong bo mo hinh AI moi co kha nang tao video chat luong cao "
            "chi tu mot cau lenh van ban don gian"
        )
        lines = wrap_caption_lines(text)
        assert lines
        assert all(len(line) <= MAX_LINE_CHARS for line in lines)
        assert " ".join(lines) == text

    def test_long_word_hard_split(self):
        lines = wrap_caption_lines("a" * 100)
        assert [len(line) for line in lines] == [42, 42, 16]

    def test_empty_text(self):
        assert wrap_caption_lines("") == []

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            wrap_caption_lines("x", max_chars=0)


class TestSegment:
    def test_blocks_have_at_most_two_lines(self):
        text = "word " * 60
        blocks = segment_caption_blocks(text.strip())
        assert blocks
        assert all(1 <= len(block) <= 2 for block in blocks)
        assert all(len(line) <= MAX_LINE_CHARS for block in blocks for line in block)

    def test_short_text_single_block(self):
        assert segment_caption_blocks("Xin chao") == [("Xin chao",)]


class TestTimestamps:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (3.204, "00:00:03,204"),
            (7.7, "00:00:07,700"),
            (3671.5, "01:01:11,500"),
            (-1.0, "00:00:00,000"),  # clamp
        ],
    )
    def test_srt_timestamp(self, seconds, expected):
        assert format_srt_timestamp(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0:00:00.00"),
            (3.2, "0:00:03.20"),
            (3671.55, "1:01:11.55"),
            (-0.5, "0:00:00.00"),
        ],
    )
    def test_ass_timestamp(self, seconds, expected):
        assert format_ass_timestamp(seconds) == expected


class TestEventsFromTimepoints:
    def test_golden_srt(self):
        narrations = ["Xin chao AI", "Cong nghe moi"]
        timepoints = {"s0": 0.0, "s1": 7.7}
        events = events_from_timepoints(narrations, timepoints, total_duration=15.4)
        expected = (
            "1\n"
            "00:00:00,000 --> 00:00:07,700\n"
            "Xin chao AI\n"
            "\n"
            "2\n"
            "00:00:07,700 --> 00:00:15,400\n"
            "Cong nghe moi\n"
        )
        assert build_srt(events) == expected

    def test_blocks_split_scene_span_proportionally(self):
        events = events_from_timepoints(
            ["aaaa bbbb"],
            {"s0": 0.0},
            total_duration=8.0,
            max_chars=4,
            max_lines=1,
        )
        assert events == [
            CaptionEvent(start=0.0, end=4.0, lines=("aaaa",)),
            CaptionEvent(start=4.0, end=8.0, lines=("bbbb",)),
        ]

    def test_events_contiguous_and_monotonic(self):
        narrations = [
            "Mot mo hinh AI moi vua duoc cong bo voi kha nang sinh video sieu thuc",
            "No co the tao ra video do phan giai cao chi tu mot dong mo ta ngan gon",
        ]
        timepoints = {"s0": 0.0, "s1": 7.7}
        events = events_from_timepoints(narrations, timepoints, total_duration=15.4)
        assert events[0].start == 0.0
        assert events[-1].end == 15.4
        for prev, cur in zip(events, events[1:], strict=False):
            assert prev.end == cur.start or prev.end <= cur.start
            assert prev.start < prev.end

    def test_empty_narration_skipped(self):
        events = events_from_timepoints(["", "Hello"], {"s0": 0.0, "s1": 5.0}, total_duration=10.0)
        assert len(events) == 1
        assert events[0].start == 5.0
        assert events[0].end == 10.0

    def test_whitespace_normalized(self):
        events = events_from_timepoints(["Hello\n   world"], {"s0": 0.0}, total_duration=5.0)
        assert events[0].lines == ("Hello world",)

    def test_missing_mark_raises(self):
        with pytest.raises(KeyError):
            events_from_timepoints(["Hello"], {"wrong0": 0.0}, total_duration=5.0)

    def test_non_positive_span_raises(self):
        with pytest.raises(ValueError):
            events_from_timepoints(["A", "B"], {"s0": 5.0, "s1": 5.0}, total_duration=10.0)

    def test_event_text_property(self):
        event = CaptionEvent(start=0.0, end=1.0, lines=("line one", "line two"))
        assert event.text == "line one\nline two"

    def test_empty_events_build_empty_srt(self):
        assert build_srt([]) == ""


class TestAss:
    def test_style_block_noto_sans_safe_zone(self):
        block = ass_style_block()
        assert "PlayResX: 1080" in block
        assert "PlayResY: 1920" in block
        assert "Style: Caption,Noto Sans,54," in block
        # ..., Alignment=2 (bottom-center), MarginL=60, MarginR=60, MarginV=320, Encoding=1
        assert block.rstrip().endswith("2,60,60,320,1")

    def test_build_ass_dialogue_lines(self):
        events = [CaptionEvent(start=0.0, end=3.2, lines=("line one", "line two"))]
        doc = build_ass(events)
        assert "[Events]" in doc
        assert (
            "Dialogue: 0,0:00:00.00,0:00:03.20,Caption,,0,0,0,,line one\\Nline two" in doc
        )

    def test_build_ass_sanitizes_override_braces(self):
        events = [CaptionEvent(start=0.0, end=1.0, lines=("{\\b1}bold{\\b0}",))]
        doc = build_ass(events)
        assert "{" not in doc.split("[Events]")[1]
        assert "(\\b1)bold(\\b0)" in doc
