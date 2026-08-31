"""Unit tests for the pure ffmpeg argv builders and escaping helpers."""

import pytest

from app.media.ffmpeg import (
    LoudnormMeasurement,
    PlatformProfile,
    burn_captions_cmd,
    concat_crossfade_cmd,
    crossfade_offsets,
    crossfade_total_duration,
    escape_drawtext_text,
    escape_filter_path,
    loudnorm_apply_cmd,
    loudnorm_linear_ok,
    loudnorm_measure_cmd,
    mix_voiceover_cmd,
    normalize_clip_cmd,
    parse_loudnorm_json,
    platform_derivative_cmd,
    portrait_crop_image_cmd,
    quote_filter_value,
    thumbnail_cmd,
)


def arg_value(cmd: list[str], flag: str) -> str:
    """Value following a flag in argv."""
    return cmd[cmd.index(flag) + 1]


def has_pair(cmd: list[str], flag: str, value: str) -> bool:
    return any(cmd[i] == flag and cmd[i + 1] == value for i in range(len(cmd) - 1))


# ---------------------------------------------------------------------------
# normalize_clip_cmd
# ---------------------------------------------------------------------------


class TestNormalizeClip:
    def test_required_flags(self):
        cmd = normalize_clip_cmd("scene0.mp4", "scene0_norm.mp4")
        assert cmd[0] == "ffmpeg"
        assert has_pair(cmd, "-i", "scene0.mp4")
        assert cmd[-1] == "scene0_norm.mp4"
        vf = arg_value(cmd, "-vf")
        assert vf == (
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,fps=30,setsar=1"
        )
        assert has_pair(cmd, "-c:v", "libx264")
        assert has_pair(cmd, "-profile:v", "high")
        assert has_pair(cmd, "-pix_fmt", "yuv420p")
        assert has_pair(cmd, "-movflags", "+faststart")

    def test_strips_source_audio(self):
        # Veo native audio must be removed; the visual master is silent.
        cmd = normalize_clip_cmd("in.mp4", "out.mp4")
        assert "-an" in cmd


# ---------------------------------------------------------------------------
# crossfade offsets and concat
# ---------------------------------------------------------------------------


class TestCrossfadeOffsets:
    def test_five_8s_clips_match_plan(self):
        assert crossfade_offsets([8.0] * 5, 0.3) == [7.7, 15.4, 23.1, 30.8]

    def test_total_duration_is_38_8(self):
        assert crossfade_total_duration([8.0] * 5, 0.3) == 38.8

    def test_computed_from_durations_not_hardcoded(self):
        assert crossfade_offsets([5.0, 6.0, 7.0], 0.5) == [4.5, 10.0]
        assert crossfade_total_duration([5.0, 6.0, 7.0], 0.5) == 17.0

    def test_needs_two_clips(self):
        with pytest.raises(ValueError):
            crossfade_offsets([8.0], 0.3)

    def test_fade_must_be_positive(self):
        with pytest.raises(ValueError):
            crossfade_offsets([8.0, 8.0], 0.0)

    def test_duration_must_exceed_fade(self):
        with pytest.raises(ValueError):
            crossfade_offsets([8.0, 0.2, 8.0], 0.3)


class TestConcatCrossfade:
    def test_filter_chain_exact(self):
        inputs = [f"scene{i}.mp4" for i in range(5)]
        cmd = concat_crossfade_cmd(inputs, "master.mp4")
        assert cmd.count("-i") == 5
        fc = arg_value(cmd, "-filter_complex")
        assert fc == (
            "[0:v][1:v]xfade=transition=fade:duration=0.3:offset=7.7[vx1];"
            "[vx1][2:v]xfade=transition=fade:duration=0.3:offset=15.4[vx2];"
            "[vx2][3:v]xfade=transition=fade:duration=0.3:offset=23.1[vx3];"
            "[vx3][4:v]xfade=transition=fade:duration=0.3:offset=30.8[vout]"
        )
        assert has_pair(cmd, "-map", "[vout]")
        assert "-an" in cmd  # master is video-only
        assert has_pair(cmd, "-r", "30")
        assert has_pair(cmd, "-pix_fmt", "yuv420p")
        assert has_pair(cmd, "-movflags", "+faststart")
        assert cmd[-1] == "master.mp4"

    def test_durations_drive_offsets(self):
        cmd = concat_crossfade_cmd(["a.mp4", "b.mp4"], "out.mp4", durations=[4.0, 4.0], fade=0.5)
        fc = arg_value(cmd, "-filter_complex")
        assert fc == "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=3.5[vout]"

    def test_duration_count_mismatch(self):
        with pytest.raises(ValueError):
            concat_crossfade_cmd(["a.mp4", "b.mp4"], "out.mp4", durations=[8.0])

    def test_needs_two_inputs(self):
        with pytest.raises(ValueError):
            concat_crossfade_cmd(["a.mp4"], "out.mp4")


# ---------------------------------------------------------------------------
# mix_voiceover_cmd
# ---------------------------------------------------------------------------


class TestMixVoiceover:
    def test_two_tracks_delayed_and_mixed(self):
        cmd = mix_voiceover_cmd(
            "master.mp4", [("vi_scene0.wav", 0.0), ("vi_scene1.wav", 7.7)], "vi.mp4"
        )
        assert has_pair(cmd, "-i", "master.mp4")
        assert has_pair(cmd, "-i", "vi_scene0.wav")
        assert has_pair(cmd, "-i", "vi_scene1.wav")
        fc = arg_value(cmd, "-filter_complex")
        assert fc == (
            "[1:a]adelay=0:all=1[a1];"
            "[2:a]adelay=7700:all=1[a2];"
            "[a1][a2]amix=inputs=2:normalize=0[aout]"
        )
        assert has_pair(cmd, "-map", "0:v")
        assert has_pair(cmd, "-map", "[aout]")
        # Video stream-copied so both locales share the visual master.
        assert has_pair(cmd, "-c:v", "copy")
        assert has_pair(cmd, "-c:a", "aac")
        assert has_pair(cmd, "-ar", "48000")
        assert cmd[-1] == "vi.mp4"

    def test_single_track(self):
        cmd = mix_voiceover_cmd("master.mp4", [("en.wav", 1.5)], "en.mp4")
        fc = arg_value(cmd, "-filter_complex")
        assert fc == "[1:a]adelay=1500:all=1[a1]"
        assert has_pair(cmd, "-map", "[a1]")

    def test_empty_tracks_rejected(self):
        with pytest.raises(ValueError):
            mix_voiceover_cmd("master.mp4", [], "out.mp4")

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError):
            mix_voiceover_cmd("master.mp4", [("v.wav", -0.1)], "out.mp4")


# ---------------------------------------------------------------------------
# loudnorm two-pass
# ---------------------------------------------------------------------------

LOUDNORM_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers
  built with gcc 13.2.0 (Rev5, Built by MSYS2 project)
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'rendition_vi.mp4':
  Duration: 00:00:38.80, start: 0.000000, bitrate: 4523 kb/s
Stream mapping:
  Stream #0:1 -> #0:0 (aac (native) -> pcm_s16le (native))
Output #0, null, to '-':
size=N/A time=00:00:38.80 bitrate=N/A speed= 512x
[Parsed_loudnorm_0 @ 0000023c4f2b3c40]
{
\t"input_i" : "-23.62",
\t"input_tp" : "-6.47",
\t"input_lra" : "2.30",
\t"input_thresh" : "-34.13",
\t"output_i" : "-14.03",
\t"output_tp" : "-1.50",
\t"output_lra" : "2.10",
\t"output_thresh" : "-24.53",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.03"
}
"""


class TestLoudnorm:
    def test_measure_cmd(self):
        cmd = loudnorm_measure_cmd("mixed.mp4")
        assert arg_value(cmd, "-af") == "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json"
        assert cmd[-3:] == ["-f", "null", "-"]
        assert "-vn" in cmd

    def test_parse_measure_json_from_stderr(self):
        m = parse_loudnorm_json(LOUDNORM_STDERR)
        assert m.input_i == -23.62
        assert m.input_tp == -6.47
        assert m.input_lra == 2.30
        assert m.input_thresh == -34.13
        assert m.output_i == -14.03
        assert m.output_tp == -1.50
        assert m.normalization_type == "dynamic"
        assert m.target_offset == 0.03

    def test_parse_rejects_missing_block(self):
        with pytest.raises(ValueError):
            parse_loudnorm_json("frame=  100 fps=30 ... no json here")

    def test_apply_cmd_uses_measured_values(self):
        measured = parse_loudnorm_json(LOUDNORM_STDERR)
        # input_tp + gain = -6.47 + 9.62 = +3.15 dBTP > ceiling -> dynamic mode.
        assert loudnorm_linear_ok(measured, target_tp=-2.0) is False
        cmd = loudnorm_apply_cmd("mixed.mp4", "normalized.mp4", measured)
        af = arg_value(cmd, "-af")
        assert af.startswith("loudnorm=I=-14:TP=-2:LRA=11:")
        assert "measured_I=-23.62" in af
        assert "measured_TP=-6.47" in af
        assert "measured_LRA=2.3:" in af
        assert "measured_thresh=-34.13" in af
        assert "offset=0.03" in af
        assert "linear=false" in af
        assert has_pair(cmd, "-c:v", "copy")
        assert has_pair(cmd, "-c:a", "aac")
        assert has_pair(cmd, "-ar", "48000")

    def test_apply_cmd_linear_when_headroom_allows(self):
        measured = parse_loudnorm_json(LOUDNORM_STDERR)
        # Quiet, low-peak input: linear gain reaches -14 LUFS under the ceiling.
        linear_input = measured.__class__(
            input_i=-20.0, input_tp=-18.0, input_lra=2.3, input_thresh=-30.0,
            output_i=-14.0, output_tp=-12.0, output_lra=2.3, output_thresh=-24.0,
            normalization_type="linear", target_offset=0.0,
        )
        assert loudnorm_linear_ok(linear_input, target_tp=-2.0) is True
        cmd = loudnorm_apply_cmd("mixed.mp4", "normalized.mp4", linear_input)
        assert "linear=true" in arg_value(cmd, "-af")
        # Explicit override wins over the automatic decision.
        forced = loudnorm_apply_cmd(
            "mixed.mp4", "normalized.mp4", measured, linear=True
        )
        assert "linear=true" in arg_value(forced, "-af")


# ---------------------------------------------------------------------------
# escaping (mandatory unit-test list)
# ---------------------------------------------------------------------------


class TestEscaping:
    def test_windows_path_colon_and_backslashes(self):
        assert escape_filter_path(r"C:\Users\hoang\subs.srt") == r"'C\:/Users/hoang/subs.srt'"

    def test_path_with_single_quote(self):
        assert escape_filter_path(r"C:\it's\subs.srt") == r"'C\:/it'\''s/subs.srt'"

    def test_path_posix_untouched_except_quotes(self):
        assert escape_filter_path("/tmp/subs.srt") == "'/tmp/subs.srt'"

    def test_drawtext_quotes_colons_commas_percent(self):
        assert escape_drawtext_text("it's 10:30, 100% done") == r"'it\'s 10\:30\, 100\% done'"

    def test_drawtext_backslash(self):
        assert escape_drawtext_text("a\\b") == "'a\\\\b'"

    def test_drawtext_filtergraph_specials(self):
        assert escape_drawtext_text("[x]=y;") == r"'\[x\]\=y\;'"

    def test_drawtext_newlines_normalized_and_preserved(self):
        # argv is passed without a shell; drawtext renders literal newlines.
        assert escape_drawtext_text("line1\r\nline2\rline3") == "'line1\nline2\nline3'"

    def test_quote_filter_value_commas_safe(self):
        quoted = quote_filter_value("FontName=Noto Sans,Fontsize=54")
        assert quoted == "'FontName=Noto Sans,Fontsize=54'"
        assert quote_filter_value("a'b") == "'a'\\''b'"


# ---------------------------------------------------------------------------
# burn_captions_cmd
# ---------------------------------------------------------------------------


class TestBurnCaptions:
    def test_srt_windows_path_and_default_style(self):
        cmd = burn_captions_cmd("vi.mp4", r"C:\subs\vi.srt", "vi_cc.mp4")
        vf = arg_value(cmd, "-vf")
        assert vf.startswith(r"subtitles=filename='C\:/subs/vi.srt'")
        assert "force_style='FontName=Noto Sans," in vf
        assert "Alignment=2" in vf
        assert "MarginV=320" in vf  # bottom safe zone from PLAN.md
        assert has_pair(cmd, "-c:a", "copy")
        assert has_pair(cmd, "-c:v", "libx264")
        assert has_pair(cmd, "-movflags", "+faststart")

    def test_fonts_dir_escaped(self):
        cmd = burn_captions_cmd(
            "vi.mp4", r"C:\subs\vi.srt", "out.mp4", fonts_dir=r"C:\assets\fonts"
        )
        vf = arg_value(cmd, "-vf")
        assert r":fontsdir='C\:/assets/fonts'" in vf

    def test_ass_carries_its_own_style(self):
        cmd = burn_captions_cmd("vi.mp4", r"C:\subs\vi.ass", "out.mp4")
        assert "force_style" not in arg_value(cmd, "-vf")

    def test_force_style_opt_out(self):
        cmd = burn_captions_cmd("vi.mp4", "vi.srt", "out.mp4", force_style="")
        assert "force_style" not in arg_value(cmd, "-vf")


# ---------------------------------------------------------------------------
# thumbnail + platform derivatives
# ---------------------------------------------------------------------------


class TestThumbnail:
    def test_frame_extract(self):
        cmd = thumbnail_cmd("master.mp4", "thumb.jpg", at_seconds=2.5)
        assert has_pair(cmd, "-ss", "2.5")
        assert cmd.index("-ss") < cmd.index("-i")  # fast seek before input
        assert has_pair(cmd, "-frames:v", "1")
        assert "-vf" not in cmd
        assert cmd[-1] == "thumb.jpg"

    def test_overlay_text_escaped_no_expansion(self):
        cmd = thumbnail_cmd(
            "master.mp4",
            "thumb_vi.jpg",
            overlay_text="100% AI: 'nong'",
            font_file=r"C:\fonts\NotoSans-Bold.ttf",
        )
        vf = arg_value(cmd, "-vf")
        assert vf.startswith(r"drawtext=text='100\% AI\: \'nong\''")
        assert ":expansion=none" in vf
        assert ":y=h-320-text_h" in vf  # bottom safe zone
        assert r":fontfile='C\:/fonts/NotoSans-Bold.ttf'" in vf

    def test_negative_seek_rejected(self):
        with pytest.raises(ValueError):
            thumbnail_cmd("m.mp4", "t.jpg", at_seconds=-1.0)


class TestPlatformDerivative:
    def test_default_profile_is_identical_stream_copy(self):
        cmd = platform_derivative_cmd("vi.mp4", "vi_youtube.mp4")
        assert cmd == [
            "ffmpeg",
            "-y",
            "-i",
            "vi.mp4",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "vi_youtube.mp4",
        ]

    def test_custom_profile_reencodes(self):
        profile = PlatformProfile(
            name="lowres", width=720, height=1280, identical_to_master=False
        )
        cmd = platform_derivative_cmd("vi.mp4", "vi_lowres.mp4", profile)
        vf = arg_value(cmd, "-vf")
        assert "scale=720:1280" in vf
        assert "fps=30" in vf
        assert has_pair(cmd, "-c:v", "libx264")
        assert has_pair(cmd, "-ar", "48000")
        assert has_pair(cmd, "-movflags", "+faststart")


class TestLoudnormMeasurementShape:
    def test_dataclass_fields(self):
        m = LoudnormMeasurement(
            input_i=-20.0,
            input_tp=-4.0,
            input_lra=3.0,
            input_thresh=-30.0,
            output_i=-14.0,
            output_tp=-1.5,
            output_lra=3.0,
            output_thresh=-24.0,
            normalization_type="linear",
            target_offset=0.0,
        )
        assert m.input_i == -20.0
        assert m.normalization_type == "linear"


class TestPortraitCropImage:
    def test_cmd_scales_to_cover_and_crops(self):
        cmd = portrait_crop_image_cmd("in.jpg", "out.png")
        vf = arg_value(cmd, "-vf")
        assert vf.startswith("scale=1080:1920:force_original_aspect_ratio=increase,")
        assert "crop=1080:1920" in vf
        assert has_pair(cmd, "-frames:v", "1")
        assert cmd[-1] == "out.png"
