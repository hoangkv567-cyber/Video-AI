"""Offline 9:16 keyframe cards via FFmpeg. No API key, no paid quota."""

from __future__ import annotations

import re
import subprocess

from app.ai.base import ImageResult
from app.errors import UpstreamError

PROCEDURAL_MODEL_ID = "procedural_keyframe"
_SAFE_PROMPT = re.compile(r"[^A-Za-z0-9 ._-]+")


def _label(prompt: str) -> str:
    cleaned = _SAFE_PROMPT.sub(" ", prompt[:72]).strip()
    return (cleaned or "Video AI demo")[:48]


def procedural_keyframe_cmd(label: str) -> list[str]:
    lavfi = (
        "color=c=0x0f172a:s=1080x1920:d=1,format=rgba,"
        "drawbox=x=60:y=60:w=960:h=1800:color=0x38bdf8@0.7:t=8,"
        "drawbox=x=100:y=200:w=880:h=360:color=0x1e293b@0.9:t=fill,"
        f"drawtext=text='{label}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=340"
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        lavfi,
        "-vframes",
        "1",
        "-f",
        "image2",
        "-c:v",
        "png",
        "pipe:1",
    ]


class ProceduralImageProvider:
    """Free keyframe image provider: HD photo search, AI generation, and offline fallback."""

    def generate_image(
        self, prompt: str, *, model_id: str | None = None, negative_prompt: str = ""
    ) -> ImageResult:
        """Free image backends. ``model_id`` is accepted for interface
        compatibility but ignored: the result always reports the backend that
        actually produced the bytes so the cost ledger never mislabels a free
        image as a paid Gemini call."""
        del model_id, negative_prompt
        import urllib.parse

        import httpx

        # 1. Clean prompt of AI boilerplate
        prompt_clean = prompt.split("Continuity:")[0].split("Consistency:")[0]
        prompt_clean = re.sub(r"(?i)\b(vertical\s*9:?16|shot\s*of|cinematic|photorealistic|no\s*text|no\s*logos?|realistic\s*lighting|high\s*quality|4k|hd|8k|soft\s*focus)\b", "", prompt_clean)
        prompt_clean = re.sub(r"[^a-zA-Z0-9 ]", " ", prompt_clean)
        words = [w for w in prompt_clean.split() if len(w) > 2]
        search_query = " ".join(words[:6]) if words else "nature landscape"

        # 2. Search for real relevant HD photos matching the prompt
        try:
            r_ddg = httpx.get(
                "https://duckduckgo.com/?q=" + urllib.parse.quote(search_query) + "&iax=images&ia=images",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
                timeout=6.0,
            )
            vqd = re.search(r"vqd=([\d-]+)", r_ddg.text)
            if vqd:
                r_json = httpx.get(
                    "https://duckduckgo.com/i.js?q=" + urllib.parse.quote(search_query) + f"&o=json&vqd={vqd.group(1)}",
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=6.0,
                )
                for item in r_json.json().get("results", [])[:8]:
                    img_url = item.get("image")
                    w, h = item.get("width") or 0, item.get("height") or 0
                    # Portrait candidates only: landscape/square photos get
                    # center-cropped to the 9:16 frame and lose both sides.
                    if w and h and w / h > 0.85:
                        continue
                    if img_url and img_url.startswith(("http://", "https://")):
                        try:
                            img_resp = httpx.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8.0, follow_redirects=True)
                            if img_resp.status_code == 200 and len(img_resp.content) > 10000:
                                return ImageResult(
                                    image_bytes=img_resp.content,
                                    model_id="search_hd_photo",
                                    mime_type="image/jpeg",
                                )
                        except Exception:
                            continue
        except Exception:
            pass

        # 3. Try Pollinations AI image generator
        try:
            url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(search_query)}?width=720&height=1280&nologo=true&model=turbo"
            resp = httpx.get(url, timeout=10.0)
            if resp.status_code == 200 and resp.content and len(resp.content) > 1000:
                return ImageResult(
                    image_bytes=resp.content,
                    model_id="pollinations_turbo",
                    mime_type=resp.headers.get("content-type") or "image/jpeg",
                )
        except Exception:
            pass

        # 4. Try Lorem Picsum HD photo (1080x1920)
        try:
            picsum_resp = httpx.get("https://picsum.photos/1080/1920", follow_redirects=True, timeout=6.0)
            if picsum_resp.status_code == 200 and len(picsum_resp.content) > 5000:
                return ImageResult(
                    image_bytes=picsum_resp.content,
                    model_id="picsum_hd_photo",
                    mime_type="image/jpeg",
                )
        except Exception:
            pass

        # 5. Offline fail-safe canvas via FFmpeg
        proc = subprocess.run(
            procedural_keyframe_cmd(_label(prompt)),
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout and len(proc.stdout) > 500:
            return ImageResult(
                image_bytes=proc.stdout,
                model_id=PROCEDURAL_MODEL_ID,
                mime_type="image/png",
            )
        raise UpstreamError(
            "procedural keyframe ffmpeg failed",
            retryable=False,
            details={"stderr": (proc.stderr or b"")[-400:].decode("utf-8", "replace")},
        )
