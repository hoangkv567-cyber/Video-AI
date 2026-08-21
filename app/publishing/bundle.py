"""Manual publish bundle builder — the mandatory fallback for every platform.

Used whenever a target's capability is MANUAL/BLOCKED or platform review has
not been granted. Writes a folder (optionally zipped) containing the
derivative MP4 path reference, thumbnail, title, description, hashtags,
disclosure and a step-by-step Vietnamese posting guide, and returns a manifest
dict that the dashboard stores on the publish target.
"""

from __future__ import annotations

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
    """Write the manual-posting bundle for one target and return its manifest."""
    root = Path(out_dir)
    bundle_dir = root / f"{platform}_{ctx.rendition_id}_{ctx.locale}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}

    video_ref = ctx.file_path or ctx.file_url or ""
    video_ref_file = bundle_dir / "video_path.txt"
    video_ref_file.write_text(video_ref + "\n", encoding="utf-8")
    files["video_reference"] = str(video_ref_file)

    if ctx.thumbnail_path and Path(ctx.thumbnail_path).is_file():
        src = Path(ctx.thumbnail_path)
        dest = bundle_dir / f"thumbnail{src.suffix or '.jpg'}"
        shutil.copyfile(src, dest)
        files["thumbnail"] = str(dest)
    else:
        thumb_ref = bundle_dir / "thumbnail_path.txt"
        thumb_ref.write_text((ctx.thumbnail_path or "") + "\n", encoding="utf-8")
        files["thumbnail"] = str(thumb_ref)

    title_file = bundle_dir / "title.txt"
    title_file.write_text(ctx.title + "\n", encoding="utf-8")
    files["title"] = str(title_file)

    description_file = bundle_dir / "description.txt"
    description_file.write_text(ctx.description + "\n", encoding="utf-8")
    files["description"] = str(description_file)

    hashtags_file = bundle_dir / "hashtags.txt"
    hashtags_file.write_text("\n".join(ctx.hashtags) + "\n", encoding="utf-8")
    files["hashtags"] = str(hashtags_file)

    disclosure_file = bundle_dir / "disclosure.txt"
    disclosure_file.write_text(_disclosure_text(ctx), encoding="utf-8")
    files["disclosure"] = str(disclosure_file)

    instructions_file = bundle_dir / f"instructions_{platform}.md"
    instructions_file.write_text(_instructions_md(platform, ctx), encoding="utf-8")
    files["instructions"] = str(instructions_file)

    manifest: dict[str, Any] = {
        "platform": platform,
        "creative_id": ctx.creative_id,
        "rendition_id": ctx.rendition_id,
        "locale": ctx.locale,
        "title": ctx.title,
        "hashtags": list(ctx.hashtags),
        "privacy": ctx.privacy,
        "scheduled_at": ctx.scheduled_at.astimezone(UTC).isoformat() if ctx.scheduled_at else None,
        "video_path": video_ref,
        "bundle_dir": str(bundle_dir),
        "files": files,
        "zip_path": None,
        "deep_link": deep_link,
        "created_at": datetime.now(UTC).isoformat(),
    }

    manifest_file = bundle_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files["manifest"] = str(manifest_file)

    if zip_output:
        zip_path = root / f"{bundle_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(bundle_dir))
        manifest["zip_path"] = str(zip_path)

    return manifest
