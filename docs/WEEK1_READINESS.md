# Week 1 readiness

Last verified: 2026-08-21. This document records capability only; never place
API keys, OAuth tokens, passwords, or service-account JSON here.

## Local foundation

| Check | Status | Evidence |
|---|---|---|
| Git repository and Python 3.12 | Ready | `main` branch; project requires Python 3.12 |
| `uv` dependency lock | Ready | `uv.lock`; CI runs `uv sync --locked --all-extras` |
| FFmpeg/FFprobe | Ready | Real H.264/AAC round-trip test in `tests/test_runtime_smoke.py` |
| Free video path | Ready | Live 8 s keyframe-motion clip: 1080×1920, 30 fps, H.264/yuv420p with Edge VI voice as AAC 48 kHz |
| Docker Desktop, Compose, WSL2 | Ready | Docker Desktop 4.87, Linux engine, Compose v5 |
| Database migration | Ready | Initial Alembic revision; upgrade/downgrade smoke tested |
| Static analysis and unit tests | Ready | Ruff, mypy, and pytest are required CI gates |
| Local Compose stack | Ready | PostgreSQL, Redis, MinIO, web, 3 Celery workers, scheduler and Caddy started successfully |
| Runtime readiness | Ready | `/readyz` checks database, Redis, MinIO auth and both FFmpeg binaries without exposing error details |
| Local HTTP reverse proxy | Ready | Dashboard and API respond through Caddy at `http://localhost:8080`; host ports remain configurable |

## Live provider probes

Probe results below are capability-specific. A catalog result never implies that
a billed generation capability is available.

| Capability | Status | Evidence / action |
|---|---|---|
| Gemini model catalog | Ready | Text, image, Veo Lite and Veo Fast configured IDs all resolve |
| Gemini Search | Blocked quota | Live request reached Google but returned HTTP 429; wait for quota reset or enable an eligible paid tier |
| Gemini image | Skipped | No image request was made after the quota failure; adapter now requests 1K, image-only, 9:16 output |
| Veo | Skipped paid | Disabled by default; `keyframe_motion` remains the local video path |
| Groq GPT-OSS JSON | Ready | Live JSON-mode response parsed with low reasoning and a 4,096-token completion ceiling |
| Groq Orpheus English TTS | Blocked terms | Model resolves, but the organization admin must accept the Orpheus model terms in Groq Console |
| Edge TTS Vietnamese | Ready | Live MP3 generated with `vi-VN-HoaiMyNeural` |
| Edge TTS English fallback | Ready | Live MP3 generated with `en-US-AriaNeural` |
| Groq Whisper | Ready | Live Vietnamese MP3 produced word timestamps and a transcript |

Current reference pricing used by the ledger is USD 0.067 per Gemini 1K image,
USD 0.05/s for Veo Lite 720p and USD 0.10/s for Veo Fast 720p. Reverify before
enabling paid generation.

## External prerequisites

These items require account ownership or product decisions outside the source
repository. Keep the application capability at `MANUAL` or `BLOCKED` until its
probe succeeds.

| Provider/platform | Required input | Safe fallback |
|---|---|---|
| Gemini | Current quota or eligible billing for Search/Image/Veo | Groq script fallback, manual sources and keyframe-motion video |
| Groq Orpheus | Organization admin accepts the model-specific terms | Edge TTS for English and Vietnamese |
| YouTube | Google Cloud project, OAuth client, `youtube.upload`, API audit for public uploads | Private upload or manual bundle |
| Facebook Page Reels | Meta app, Page access, required Page scopes | Test Page or manual bundle |
| TikTok | Developer app and approved `video.upload` scope | Manual bundle; Direct Post is disabled for the internal MVP |
| Zalo OA | Zalo app, eligible OA, paid OpenAPI/content permission | Manual OA Manager bundle |
| Production HTTPS | Domain pointing to the VPS | Local `http://localhost` only |
| CI publishing | GitHub remote supplied by the project owner | Run the same commands locally |

## Before requesting app review

- Replace all sample domains and support contacts.
- Publish privacy policy, terms of use, and data-deletion instructions.
- Record an OAuth-to-publish screencast using test accounts.
- Verify callback URLs exactly match `PUBLIC_BASE_URL`.
- Confirm tokens are encrypted and absent from logs, screenshots, and Git.
- Record each connection as `LIVE`, `PRIVATE_TEST`, `MANUAL`, or `BLOCKED`.
