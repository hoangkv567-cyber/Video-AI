"""Manual bundle builder tests: full manifest contents, zip output, Zalo fallback."""

import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import json  # noqa: E402
import zipfile  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from app.publishing.base import PublishContext, PublishNeedsAction  # noqa: E402
from app.publishing.bundle import OA_MANAGER_DEEP_LINK, build_bundle  # noqa: E402
from app.publishing.zalo import ZaloPublisher  # noqa: E402
from app.schemas.videoplan import Disclosure  # noqa: E402
from app.states import Capability  # noqa: E402

ALL_PLATFORMS = ["youtube", "facebook", "tiktok", "zalo"]


def make_ctx(tmp_path: Path, **overrides) -> PublishContext:
    video = tmp_path / "video.mp4"
    thumb = tmp_path / "thumb.jpg"
    video.write_bytes(b"\x00" * 128)
    thumb.write_bytes(b"\xff\xd8fakejpeg")
    defaults = dict(
        creative_id="creative-9",
        rendition_id="rendition-vi",
        locale="vi",
        title="Chip AI moi cua Viet Nam",
        description="Ban tin 40 giay ve chip AI.",
        hashtags=["#AI", "#chip", "#congnghe"],
        file_path=str(video),
        thumbnail_path=str(thumb),
        privacy="private",
        disclosure=Disclosure(synthetic_media=True, made_for_kids=False, content_risks=["ai_faces"]),
    )
    defaults.update(overrides)
    return PublishContext(**defaults)


class TestBundleContents:
    def test_full_manifest_and_files(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, scheduled_at=datetime(2026, 9, 2, 1, 0, tzinfo=UTC))
        out = tmp_path / "bundles"

        manifest = build_bundle(ctx, "tiktok", out)

        bundle_dir = Path(manifest["bundle_dir"])
        assert bundle_dir.is_dir()
        assert bundle_dir.parent == out
        assert manifest["platform"] == "tiktok"
        assert manifest["creative_id"] == "creative-9"
        assert manifest["rendition_id"] == "rendition-vi"
        assert manifest["locale"] == "vi"
        assert manifest["title"] == ctx.title
        assert manifest["hashtags"] == ctx.hashtags
        assert manifest["video_path"] == ctx.file_path
        assert manifest["zip_path"] is None
        created = datetime.fromisoformat(manifest["created_at"])
        assert created.tzinfo is not None  # UTC timestamp

        files = manifest["files"]
        expected_keys = {
            "video_reference",
            "thumbnail",
            "title",
            "description",
            "hashtags",
            "disclosure",
            "instructions",
            "manifest",
        }
        assert expected_keys.issubset(files)
        for path in files.values():
            assert Path(path).is_file()

        assert Path(files["title"]).read_text(encoding="utf-8").strip() == ctx.title
        assert (
            Path(files["description"]).read_text(encoding="utf-8").strip() == ctx.description
        )
        assert (
            Path(files["hashtags"]).read_text(encoding="utf-8").splitlines() == ctx.hashtags
        )
        assert ctx.file_path in Path(files["video_reference"]).read_text(encoding="utf-8")
        # Thumbnail was copied into the bundle.
        assert Path(files["thumbnail"]).read_bytes() == Path(ctx.thumbnail_path).read_bytes()

        disclosure = Path(files["disclosure"]).read_text(encoding="utf-8")
        assert "AI" in disclosure
        assert "synthetic media" in disclosure
        assert "ai_faces" in disclosure

        stored = json.loads(Path(files["manifest"]).read_text(encoding="utf-8"))
        assert stored["platform"] == "tiktok"
        assert stored["scheduled_at"] == "2026-09-02T01:00:00+00:00"

    @pytest.mark.parametrize("platform", ALL_PLATFORMS)
    def test_vietnamese_instructions_per_platform(self, tmp_path: Path, platform: str) -> None:
        ctx = make_ctx(tmp_path)
        manifest = build_bundle(ctx, platform, tmp_path / "bundles")
        instructions_path = Path(manifest["files"]["instructions"])
        assert instructions_path.name == f"instructions_{platform}.md"
        text = instructions_path.read_text(encoding="utf-8")
        assert "Hướng dẫn đăng thủ công" in text
        assert "video_path.txt" in text
        assert "1." in text  # numbered step-by-step guide

    def test_zip_output_contains_all_files(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path)
        manifest = build_bundle(ctx, "youtube", tmp_path / "bundles", zip_output=True)
        zip_path = Path(manifest["zip_path"])
        assert zip_path.is_file()
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
        assert {
            "title.txt",
            "description.txt",
            "hashtags.txt",
            "disclosure.txt",
            "instructions_youtube.md",
            "video_path.txt",
            "manifest.json",
        }.issubset(names)

    def test_missing_thumbnail_falls_back_to_reference(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path, thumbnail_path=str(tmp_path / "missing.jpg"))
        manifest = build_bundle(ctx, "facebook", tmp_path / "bundles")
        thumb_ref = Path(manifest["files"]["thumbnail"])
        assert thumb_ref.name == "thumbnail_path.txt"
        assert "missing.jpg" in thumb_ref.read_text(encoding="utf-8")

    def test_rebuild_is_idempotent(self, tmp_path: Path) -> None:
        ctx = make_ctx(tmp_path)
        first = build_bundle(ctx, "zalo", tmp_path / "bundles")
        second = build_bundle(ctx, "zalo", tmp_path / "bundles")
        assert first["bundle_dir"] == second["bundle_dir"]
        assert Path(second["files"]["title"]).read_text(encoding="utf-8").strip() == ctx.title


class TestZaloFallback:
    def test_manual_capability_raises_needs_action_with_full_bundle(
        self, tmp_path: Path
    ) -> None:
        ctx = make_ctx(tmp_path, extra={"bundle_dir": str(tmp_path / "bundles")})
        pub = ZaloPublisher({}, capability=Capability.MANUAL)

        with pytest.raises(PublishNeedsAction) as excinfo:
            pub.prepare(ctx)

        details = excinfo.value.details
        assert details["deep_link"] == OA_MANAGER_DEEP_LINK
        bundle = details["bundle"]
        assert bundle["platform"] == "zalo"
        assert bundle["video_path"] == ctx.file_path
        assert Path(bundle["files"]["thumbnail"]).is_file()
        assert Path(bundle["files"]["instructions"]).is_file()
        instructions = Path(bundle["files"]["instructions"]).read_text(encoding="utf-8")
        assert OA_MANAGER_DEEP_LINK in instructions

    def test_blocked_capability_also_bundles_on_upload_and_finalize(
        self, tmp_path: Path
    ) -> None:
        ctx = make_ctx(tmp_path, extra={"bundle_dir": str(tmp_path / "bundles")})
        pub = ZaloPublisher({}, capability=Capability.BLOCKED)
        with pytest.raises(PublishNeedsAction):
            pub.upload(ctx, {})
        with pytest.raises(PublishNeedsAction):
            pub.finalize(ctx, {})
