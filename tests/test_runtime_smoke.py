"""Small real-binary smoke tests for the local/container runtime."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@pytest.mark.runtime
@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="FFmpeg/FFprobe is not installed")
def test_ffmpeg_and_ffprobe_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "runtime-smoke.mp4"
    subprocess.run(
        [
            str(FFMPEG),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x568:r=24:d=0.5",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    probe = subprocess.run(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    streams = json.loads(probe.stdout)["streams"]

    assert output.stat().st_size > 0
    assert any(
        stream.get("codec_type") == "video"
        and stream.get("codec_name") == "h264"
        and stream.get("width") == 320
        and stream.get("height") == 568
        for stream in streams
    )
    assert any(
        stream.get("codec_type") == "audio" and stream.get("codec_name") == "aac"
        for stream in streams
    )
