"""Ken Burns motion-clip builders: the free-tier replacement for Veo scenes.

PLAN.md "Chế độ miễn phí": Veo has no free tier, so with
``VIDEO_PROVIDER=keyframe_motion`` each 8 s scene is cut from its still
keyframe with an ffmpeg ``zoompan`` move (slow zoom 1.0 -> ~1.12 plus a
direction-dependent pan), keeping the 5-scene / 0.3 s-crossfade structure.

Same contract as :mod:`app.media.ffmpeg`: every public ``*_cmd`` function is a
pure argv builder returning ``list[str]`` — no subprocess, no I/O; execution
goes through ``app.media.runner``. Output clips match the master spec
(1080x1920 @ 30 fps, H.264 High, yuv420p, ``+faststart``, silent) so they are
drop-in inputs for ``normalize_clip_cmd`` / ``concat_crossfade_cmd``.

Jitter note: zoompan quantizes the crop to input pixels, so the keyframe is
first supersampled 2x; the zoom uses the canonical iterative
``zoom+<increment>`` expression with the increment computed from
``duration * fps`` (0.0005 for 240 frames), which stays monotonic and
sub-pixel-smooth.
"""

from __future__ import annotations

from app.media.ffmpeg import (
    TARGET_FPS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    VIDEO_CRF,
    VIDEO_PRESET,
)

DIRECTION_ZOOM_IN_CENTER = "zoom-in-center"
DIRECTION_PAN_UP = "pan-up"
DIRECTION_PAN_DOWN = "pan-down"
DIRECTION_ZOOM_OUT = "zoom-out"
DIRECTION_DIAGONAL = "diagonal"

#: Cycle order used by :func:`direction_for_scene` (scene index modulo 5).
DIRECTIONS: tuple[str, ...] = (
    DIRECTION_ZOOM_IN_CENTER,
    DIRECTION_PAN_UP,
    DIRECTION_PAN_DOWN,
    DIRECTION_ZOOM_OUT,
    DIRECTION_DIAGONAL,
)

MAX_ZOOM = 1.12
SUPERSAMPLE = 2  # pre-scale factor so zoompan's integer crop never jitters


def _fmt(value: float) -> str:
    """Compact float formatting for filter args: 0.0005 -> '0.0005'."""
    return f"{value:g}"


def direction_for_scene(index: int) -> str:
    """Deterministic per-scene camera move: cycles through ``DIRECTIONS``."""
    if index < 0:
        raise ValueError(f"scene index must be >= 0, got {index}")
    return DIRECTIONS[index % len(DIRECTIONS)]


def _zoompan_exprs(direction: str, frames: int) -> tuple[str, str, str]:
    """(z, x, y) zoompan expressions for one direction over ``frames`` frames."""
    increment = (MAX_ZOOM - 1.0) / frames
    last = frames - 1
    zoom_in = f"min(zoom+{_fmt(increment)},{_fmt(MAX_ZOOM)})"
    # Zoom-out must start at MAX_ZOOM on the first output frame, then shrink.
    zoom_out = f"if(eq(on,0),{_fmt(MAX_ZOOM)},max(zoom-{_fmt(increment)},1))"
    center_x = "iw/2-(iw/zoom/2)"
    center_y = "ih/2-(ih/zoom/2)"
    if direction == DIRECTION_ZOOM_IN_CENTER:
        return zoom_in, center_x, center_y
    if direction == DIRECTION_ZOOM_OUT:
        return zoom_out, center_x, center_y
    if direction == DIRECTION_PAN_UP:
        return zoom_in, center_x, f"(ih-ih/zoom)*(1-on/{last})"
    if direction == DIRECTION_PAN_DOWN:
        return zoom_in, center_x, f"(ih-ih/zoom)*on/{last}"
    if direction == DIRECTION_DIAGONAL:
        return zoom_in, f"(iw-iw/zoom)*on/{last}", f"(ih-ih/zoom)*on/{last}"
    raise ValueError(f"unknown direction: {direction!r} (expected one of {DIRECTIONS})")


def kenburns_clip_cmd(
    image_path: str,
    output_path: str,
    *,
    duration_s: float = 8.0,
    fps: int = TARGET_FPS,
    width: int = TARGET_WIDTH,
    height: int = TARGET_HEIGHT,
    direction: str = DIRECTION_ZOOM_IN_CENTER,
) -> list[str]:
    """One silent Ken Burns scene clip from a still keyframe.

    The keyframe is scale-to-cover cropped onto a supersampled canvas, animated
    with zoompan (``d`` = ``round(duration_s * fps)`` output frames at the
    target geometry) and encoded to the master spec: H.264 High, yuv420p,
    ``+faststart``, no audio — a drop-in input for ``concat_crossfade_cmd``.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid geometry {width}x{height}")
    frames = round(duration_s * fps)
    if frames < 2:
        raise ValueError(f"duration_s*fps must be >= 2 frames, got {frames}")

    z_expr, x_expr, y_expr = _zoompan_exprs(direction, frames)
    super_w, super_h = width * SUPERSAMPLE, height * SUPERSAMPLE
    vf = (
        f"scale={super_w}:{super_h}:force_original_aspect_ratio=increase,"
        f"crop={super_w}:{super_h},setsar=1,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={frames}:s={width}x{height}:fps={fps},"
        "format=yuv420p"
    )
    return [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        image_path,
        "-vf",
        vf,
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        VIDEO_PRESET,
        "-crf",
        str(VIDEO_CRF),
        "-movflags",
        "+faststart",
        output_path,
    ]
