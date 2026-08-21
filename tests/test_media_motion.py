"""Ken Burns motion builder tests: pure argv, zoompan expressions, direction
cycling, and drop-in compatibility with the crossfade concat (38.8 s master)."""

from __future__ import annotations

import pytest

from app.media.ffmpeg import (
    concat_crossfade_cmd,
    crossfade_offsets,
    crossfade_total_duration,
)
from app.media.motion import (
    DIRECTION_DIAGONAL,
    DIRECTION_PAN_DOWN,
    DIRECTION_PAN_UP,
    DIRECTION_ZOOM_IN_CENTER,
    DIRECTION_ZOOM_OUT,
    DIRECTIONS,
    direction_for_scene,
    kenburns_clip_cmd,
)


def _vf(cmd: list[str]) -> str:
    return cmd[cmd.index("-vf") + 1]


class TestKenburnsClipCmd:
    def test_default_argv_contract(self) -> None:
        cmd = kenburns_clip_cmd("kf.png", "out.mp4")
        assert cmd[0] == "ffmpeg"
        assert cmd[-1] == "out.mp4"
        assert "-loop" in cmd and cmd[cmd.index("-loop") + 1] == "1"
        assert cmd[cmd.index("-framerate") + 1] == "30"
        assert cmd[cmd.index("-i") + 1] == "kf.png"
        # 8.0 s x 30 fps = 240 output frames, capped explicitly.
        assert cmd[cmd.index("-frames:v") + 1] == "240"
        assert cmd[cmd.index("-r") + 1] == "30"
        assert "-an" in cmd  # silent by design; voice-over is mixed later
        assert cmd[cmd.index("-c:v") + 1] == "libx264"
        assert cmd[cmd.index("-profile:v") + 1] == "high"
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert cmd[cmd.index("-movflags") + 1] == "+faststart"

        vf = _vf(cmd)
        assert "zoompan=" in vf
        assert "d=240" in vf  # computed frame count inside the filter
        assert "s=1080x1920" in vf  # target resolution
        assert "fps=30" in vf
        # Standard iterative increment: 0.12 zoom delta over 240 frames.
        assert "zoom+0.0005" in vf
        assert "1.12" in vf
        assert "format=yuv420p" in vf
        # Supersampled pre-scale keeps the zoompan crop sub-pixel smooth.
        assert "scale=2160:3840" in vf and "crop=2160:3840" in vf

    def test_custom_duration_and_fps_recompute_frames_and_increment(self) -> None:
        cmd = kenburns_clip_cmd("kf.png", "out.mp4", duration_s=4.0, fps=25)
        vf = _vf(cmd)
        assert cmd[cmd.index("-frames:v") + 1] == "100"
        assert "d=100" in vf
        assert "fps=25" in vf
        assert "zoom+0.0012" in vf  # 0.12 / 100 frames

    def test_direction_expressions(self) -> None:
        center = kenburns_clip_cmd("k.png", "o.mp4", direction=DIRECTION_ZOOM_IN_CENTER)
        assert "iw/2-(iw/zoom/2)" in _vf(center) and "min(zoom+" in _vf(center)

        out = kenburns_clip_cmd("k.png", "o.mp4", direction=DIRECTION_ZOOM_OUT)
        assert "if(eq(on,0),1.12" in _vf(out) and "max(zoom-" in _vf(out)

        pan_up = kenburns_clip_cmd("k.png", "o.mp4", direction=DIRECTION_PAN_UP)
        assert "(ih-ih/zoom)*(1-on/239)" in _vf(pan_up)

        pan_down = kenburns_clip_cmd("k.png", "o.mp4", direction=DIRECTION_PAN_DOWN)
        assert "(ih-ih/zoom)*on/239" in _vf(pan_down)

        diagonal = kenburns_clip_cmd("k.png", "o.mp4", direction=DIRECTION_DIAGONAL)
        assert "(iw-iw/zoom)*on/239" in _vf(diagonal)
        assert "(ih-ih/zoom)*on/239" in _vf(diagonal)

    def test_all_directions_build_and_differ(self) -> None:
        graphs = {d: _vf(kenburns_clip_cmd("k.png", "o.mp4", direction=d)) for d in DIRECTIONS}
        assert len(set(graphs.values())) == len(DIRECTIONS)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="unknown direction"):
            kenburns_clip_cmd("k.png", "o.mp4", direction="spin")
        with pytest.raises(ValueError, match="duration_s"):
            kenburns_clip_cmd("k.png", "o.mp4", duration_s=0)
        with pytest.raises(ValueError, match="fps"):
            kenburns_clip_cmd("k.png", "o.mp4", fps=0)
        with pytest.raises(ValueError, match="geometry"):
            kenburns_clip_cmd("k.png", "o.mp4", width=0)


class TestDirectionCycling:
    def test_cycles_deterministically_per_scene_index(self) -> None:
        assert [direction_for_scene(i) for i in range(5)] == list(DIRECTIONS)
        assert direction_for_scene(5) == direction_for_scene(0)
        assert direction_for_scene(7) == direction_for_scene(2)
        # Deterministic: same index, same direction, every time.
        assert direction_for_scene(3) == direction_for_scene(3) == DIRECTION_ZOOM_OUT

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValueError):
            direction_for_scene(-1)


class TestConcatCompatibility:
    def test_motion_clips_feed_concat_crossfade_with_unchanged_offsets(self) -> None:
        """5 x 8 s Ken Burns clips slot into the standard 38.8 s crossfade chain."""
        clips = [
            kenburns_clip_cmd(f"kf{i}.png", f"motion{i}.mp4", direction=direction_for_scene(i))
            for i in range(5)
        ]
        durations = [8.0] * 5
        assert crossfade_offsets(durations) == [7.7, 15.4, 23.1, 30.8]
        assert crossfade_total_duration(durations) == pytest.approx(38.8)

        cmd = concat_crossfade_cmd([c[-1] for c in clips], "master.mp4", durations)
        graph = cmd[cmd.index("-filter_complex") + 1]
        for offset in ("offset=7.7", "offset=15.4", "offset=23.1", "offset=30.8"):
            assert offset in graph
        assert graph.count("xfade=transition=fade:duration=0.3") == 4
        # All five motion clip outputs are inputs to the chain, in scene order.
        inputs = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-i"]
        assert inputs == [f"motion{i}.mp4" for i in range(5)]
