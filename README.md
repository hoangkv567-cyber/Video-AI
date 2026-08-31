# Video AI — MVP sinh video AI tự động

Hệ thống nội bộ sinh video Shorts dọc ~38,8s (5 cảnh × 8s, crossfade 0,3s) với hai bản VI/EN dùng chung visual master, đăng lên YouTube Shorts, Facebook Reels, TikTok và Zalo OA. Xem chi tiết trong [PLAN.md](PLAN.md).

## Kiến trúc

Modular monolith Python 3.12: FastAPI + Jinja2/JavaScript tối thiểu, PostgreSQL, Celery + Redis (queue `ai`, `render`, `publish`), FFmpeg trong container, MinIO cho asset, Caddy cho HTTPS.

## Chuẩn bị Windows

```powershell
winget install --id astral-sh.uv --exact
winget install --id Gyan.FFmpeg --exact
winget install --id Docker.DockerDesktop --exact
```

Mở lại terminal sau khi cài để nhận PATH mới. Docker Desktop cần WSL2 và có
thể yêu cầu xác nhận UAC ở lần cài/khởi động đầu tiên.

## Chạy dev (không cần Docker)

```powershell
uv sync --locked --all-extras
Copy-Item .env.example .env    # chỉ chạy nếu chưa có .env
uv run uvicorn app.main:app --reload
# http://localhost:8000/healthz
```

Chạy test:

```powershell
uv run pytest -q
uv run ruff check app tests
uv run mypy app
```

## Chạy production (Docker Compose)

```powershell
Copy-Item .env.production.example .env.production
docker compose --env-file .env.production -f compose.yaml -f compose.production.yaml config --quiet
docker compose --env-file .env.production -f compose.yaml -f compose.production.yaml up -d --build
```

Compose chạy `alembic upgrade head` qua service `migrate` trước khi khởi động
web/worker. Dashboard production ở `PUBLIC_BASE_URL`; file mẫu publish Caddy
trên cổng 80/443. Chạy local chỉ với `compose.yaml` thì dashboard ở
`http://localhost:8080`.

Kiểm tra liveness tại `/healthz` và hạ tầng nội bộ tại `/readyz`. Groq Orpheus
cần admin tài khoản chấp nhận điều khoản model trước khi dùng; Edge TTS là
đường thay thế đã được live-smoke cho cả VI và EN.

Services: `web`, `worker-ai`, `worker-render`, `worker-publish`, `scheduler`, `postgres`, `redis`, `minio`, `caddy`.

Checklist secret, HTTPS, backup, bootstrap admin và baseline database legacy nằm trong
[docs/PRODUCTION.md](docs/PRODUCTION.md). Không chạy production chỉ với file Compose cơ sở.

## Cấu trúc mã

- `app/config.py` — settings + bảng model/giá có version
- `app/readiness.py` — probe database, Redis, MinIO và FFmpeg/FFprobe
- `app/states.py` — state machine pipeline (DRAFT → … → PUBLISHED)
- `app/models.py` — bảng: user, campaign, creative, source, script version, scene, asset, rendition, cost event, connected account, job, publish target/attempt, audit
- `app/schemas/videoplan.py` — VideoPlan v1 + semantic validator + auto-mode gate
- `app/ai/` — Gemini research/scripting, keyframe (Nano Banana), Veo LRO, TTS
- `app/media/` — FFmpeg builders, QC/ffprobe, caption/SRT, loudness
- `app/publishing/` — publisher contract + YouTube/Facebook/TikTok/Zalo + manual bundle
- `app/api/` — REST API, OAuth, webhooks, idempotency
- `app/workers/` — Celery app, tasks, scheduler
- `app/web/` — dashboard Jinja2 + JavaScript `fetch` tối thiểu

## Nguyên tắc vận hành

- Mọi thời gian lưu UTC, hiển thị `Asia/Ho_Chi_Minh`.
- Chi phí AI mỗi creative bị chặn cứng ở 6 USD (admin override phải có lý do, ghi audit).
- Chỉ retry 408/429/5xx với backoff + jitter; lỗi quyền/policy → `NEEDS_ACTION`.
- Trước khi retry sau timeout phải query trạng thái remote — không blind retry.
- TikTok luôn cần consent thủ công trong MVP; mọi kênh đều có manual bundle fallback.
