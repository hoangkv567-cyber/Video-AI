"""Manual publish bundle builder — the mandatory fallback for every platform.

Used whenever a target's capability is MANUAL/BLOCKED or platform review has
not been granted. Writes a folder (optionally zipped) containing the
actual derivative MP4, thumbnail, title, description, hashtags,
disclosure and a step-by-step Vietnamese posting guide, and returns a manifest
whose internal paths stay valid after the ZIP is moved or downloaded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.publishing.base import PublishContext

OA_MANAGER_DEEP_LINK = "https://oa.zalo.me/manage"

VN_TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh (no DST); offline-safe

PLATFORM_NAMES = {
    "youtube": "YouTube Shorts",
    "facebook": "Facebook Page Reels",
    "tiktok": "TikTok",
    "zalo": "Zalo OA",
}

_PLATFORM_STEPS: dict[str, list[str]] = {
    "youtube": [
        "Mở YouTube Studio (https://studio.youtube.com) và đăng nhập đúng kênh.",
        "Chọn **Tạo → Tải video lên**, rồi chọn tệp MP4 theo đường dẫn trong `video_path.txt`.",
        "Dán tiêu đề từ `title.txt` và mô tả từ `description.txt`; thêm hashtag từ `hashtags.txt`.",
        "Tải ảnh thumbnail trong gói này lên làm hình thu nhỏ.",
        "Trong mục khai báo nội dung, bật nhãn **nội dung do AI tạo/chỉnh sửa** và đặt "
        "**Video dành cho trẻ em** đúng theo `disclosure.txt`.",
        "Đặt chế độ hiển thị (Riêng tư/Công khai) hoặc đặt lịch đăng theo thời gian bên dưới.",
        "Nhấn **Xuất bản**, sau đó dán ID video (URL) vào hệ thống để theo dõi trạng thái.",
    ],
    "facebook": [
        "Mở Meta Business Suite (https://business.facebook.com) và chọn đúng Trang.",
        "Chọn **Tạo Reel**, tải lên tệp MP4 theo đường dẫn trong `video_path.txt`.",
        "Dán caption gồm tiêu đề (`title.txt`), mô tả (`description.txt`) và hashtag "
        "(`hashtags.txt`).",
        "Gắn nhãn **nội dung do AI tạo** nếu Trang có tùy chọn này (xem `disclosure.txt`).",
        "Đăng ngay hoặc chọn **Lên lịch** theo thời gian bên dưới (cửa sổ hợp lệ: "
        "từ 10 phút đến 30 ngày kể từ hiện tại).",
        "Sau khi đăng, dán ID bài viết vào hệ thống để theo dõi trạng thái.",
    ],
    "tiktok": [
        "Mở TikTok Studio (https://www.tiktok.com/tiktokstudio) hoặc ứng dụng TikTok và đăng "
        "nhập đúng tài khoản.",
        "Chọn **Tải lên**, chọn tệp MP4 theo đường dẫn trong `video_path.txt`.",
        "Dán caption gồm tiêu đề (`title.txt`) và hashtag (`hashtags.txt`).",
        "Bật nhãn **Nội dung do AI tạo** (AI-generated content) theo `disclosure.txt`.",
        "Chọn ảnh bìa gần giống thumbnail trong gói này.",
        "Đăng ngay hoặc đặt lịch trong TikTok Studio theo thời gian bên dưới.",
        "Sau khi đăng, dán link video vào hệ thống để theo dõi trạng thái.",
    ],
    "zalo": [
        f"Mở Zalo OA Manager ({OA_MANAGER_DEEP_LINK}) và đăng nhập đúng Official Account.",
        "Chọn **Nội dung → Tạo bài viết/video**, tải lên tệp MP4 theo đường dẫn trong "
        "`video_path.txt`.",
        "Dán tiêu đề (`title.txt`) và nội dung mô tả (`description.txt`).",
        "Tải ảnh thumbnail trong gói này lên làm ảnh đại diện bài viết.",
        "Ghi rõ nội dung do AI tạo theo `disclosure.txt` trong phần mô tả.",
        "Đăng ngay hoặc đặt lịch trong OA Manager theo thời gian bên dưới.",
        "Sau khi đăng, dán ID bài viết vào hệ thống để theo dõi trạng thái.",
    ],
}

_GENERIC_STEPS = [
    "Tải lên tệp MP4 theo đường dẫn trong `video_path.txt` bằng công cụ đăng bài của nền tảng.",
    "Dán tiêu đề (`title.txt`), mô tả (`description.txt`) và hashtag (`hashtags.txt`).",
    "Khai báo nội dung do AI tạo theo `disclosure.txt`.",
    "Đăng ngay hoặc đặt lịch theo thời gian bên dưới, rồi cập nhật ID bài đăng vào hệ thống.",
]


def _yes_no(value: bool) -> str:
    return "Có" if value else "Không"


def _disclosure_text(ctx: PublishContext) -> str:
    lines = [
        "KHAI BÁO NỘI DUNG (bắt buộc đọc trước khi đăng)",
        f"- Nội dung do AI tạo (synthetic media): {_yes_no(ctx.disclosure.synthetic_media)}",
        f"- Dành cho trẻ em (made for kids): {_yes_no(ctx.disclosure.made_for_kids)}",
    ]
    if ctx.disclosure.content_risks:
        lines.append("- Rủi ro nội dung cần lưu ý:")
        lines.extend(f"  * {risk}" for risk in ctx.disclosure.content_risks)
    else:
        lines.append("- Rủi ro nội dung cần lưu ý: không có ghi nhận")
    lines.append(
        "Luôn bật nhãn 'nội dung do AI tạo' khi nền tảng hỗ trợ; không đăng nếu thiếu khai báo."
    )
    return "\n".join(lines) + "\n"


def _schedule_lines(ctx: PublishContext) -> list[str]:
    if ctx.scheduled_at is None:
        return ["- Thời gian đăng: đăng ngay khi hoàn tất các bước."]
    utc = ctx.scheduled_at.astimezone(UTC)
    local = utc.astimezone(VN_TZ)
    return [
        f"- Giờ đăng dự kiến (UTC): {utc.isoformat()}",
        f"- Giờ đăng dự kiến (giờ Việt Nam, UTC+7): {local.isoformat()}",
    ]


def _instructions_md(platform: str, ctx: PublishContext) -> str:
    display = PLATFORM_NAMES.get(platform, platform)
    steps = _PLATFORM_STEPS.get(platform, _GENERIC_STEPS)
    lines = [
        f"# Hướng dẫn đăng thủ công — {display}",
        "",
        f"Rendition: `{ctx.rendition_id}` — ngôn ngữ: `{ctx.locale}` — "
        f"quyền riêng tư đề xuất: `{ctx.privacy}`.",
        "",
        "## Thông tin lịch đăng",
        *_schedule_lines(ctx),
        "",
        "## Các bước thực hiện",
    ]
    lines.extend(f"{i}. {step}" for i, step in enumerate(steps, start=1))
    lines += [
        "",
        "## Sau khi đăng",
        "- Cập nhật trạng thái và remote post ID vào dashboard (mục Publish Targets).",
        "- Nếu nền tảng từ chối nội dung, ghi lại lý do vào hệ thống để xử lý policy.",
        "",
    ]
    return "\n".join(lines)


def build_bundle(
    ctx: PublishContext,
    platform: str,
    out_dir: str | Path,
    *,
    zip_output: bool = False,
    deep_link: str | None = None,
) -> dict[str, Any]:
    """Write a self-contained manual-posting bundle for one target.

    A manual bundle is useful only when it can be moved to another machine, so
    the video itself is mandatory and every path recorded in ``manifest.json``
    is relative to the bundle root.  ``bundle_dir`` and ``zip_path`` are added
    only to the returned build result; they are deliberately excluded from the
    portable manifest stored in the archive.
    """
    root = Path(out_dir)
    bundle_dir = root / f"{platform}_{ctx.rendition_id}_{ctx.locale}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}

    if not ctx.file_path:
        raise ValueError("manual bundle requires a local video file")
    source_video = Path(ctx.file_path)
    if not source_video.is_file():
        raise FileNotFoundError(f"manual bundle video not found: {source_video}")
    bundled_video = bundle_dir / f"video{source_video.suffix or '.mp4'}"
    shutil.copyfile(source_video, bundled_video)
    video_ref = bundled_video.name
    files["video"] = bundled_video.name

    video_ref_file = bundle_dir / "video_path.txt"
    video_ref_file.write_text(video_ref + "\n", encoding="utf-8")
    files["video_reference"] = video_ref_file.name

    if ctx.thumbnail_path and Path(ctx.thumbnail_path).is_file():
        src = Path(ctx.thumbnail_path)
        dest = bundle_dir / f"thumbnail{src.suffix or '.jpg'}"
        shutil.copyfile(src, dest)
        files["thumbnail"] = dest.name
    else:
        thumb_ref = bundle_dir / "thumbnail_path.txt"
        thumb_ref.write_text((ctx.thumbnail_path or "") + "\n", encoding="utf-8")
        files["thumbnail"] = thumb_ref.name

    title_file = bundle_dir / "title.txt"
    title_file.write_text(ctx.title + "\n", encoding="utf-8")
    files["title"] = title_file.name

    description_file = bundle_dir / "description.txt"
    description_file.write_text(ctx.description + "\n", encoding="utf-8")
    files["description"] = description_file.name

    hashtags_file = bundle_dir / "hashtags.txt"
    hashtags_file.write_text("\n".join(ctx.hashtags) + "\n", encoding="utf-8")
    files["hashtags"] = hashtags_file.name

    disclosure_file = bundle_dir / "disclosure.txt"
    disclosure_file.write_text(_disclosure_text(ctx), encoding="utf-8")
    files["disclosure"] = disclosure_file.name

    instructions_file = bundle_dir / f"instructions_{platform}.md"
    instructions_file.write_text(_instructions_md(platform, ctx), encoding="utf-8")
    files["instructions"] = instructions_file.name

    manifest_file = bundle_dir / "manifest.json"
    files["manifest"] = manifest_file.name
    with bundled_video.open("rb") as video_file:
        video_sha256 = hashlib.file_digest(video_file, "sha256").hexdigest()

    manifest: dict[str, Any] = {
        "bundle_format_version": 1,
        "platform": platform,
        "creative_id": ctx.creative_id,
        "rendition_id": ctx.rendition_id,
        "locale": ctx.locale,
        "title": ctx.title,
        "hashtags": list(ctx.hashtags),
        "privacy": ctx.privacy,
        "scheduled_at": ctx.scheduled_at.astimezone(UTC).isoformat() if ctx.scheduled_at else None,
        "video_path": video_ref,
        "video_sha256": video_sha256,
        "video_size_bytes": bundled_video.stat().st_size,
        "files": files,
        "deep_link": deep_link,
        "created_at": datetime.now(UTC).isoformat(),
    }

    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    zip_path: Path | None = None
    if zip_output:
        zip_path = root / f"{bundle_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(bundle_dir))

    return {
        **manifest,
        "bundle_dir": str(bundle_dir),
        "zip_path": str(zip_path) if zip_path is not None else None,
    }
