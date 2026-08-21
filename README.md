# Video AI — MVP sinh video AI tự động

Hệ thống nội bộ sinh video Shorts dọc ~38,8s (5 cảnh × 8s, crossfade 0,3s) với hai bản VI/EN dùng chung visual master, đăng lên YouTube Shorts, Facebook Reels, TikTok và Zalo OA. Xem chi tiết trong [PLAN.md](PLAN.md).

## Kiến trúc

Modular monolith Python 3.12: FastAPI + Jinja2/HTMX, PostgreSQL, Celery + Redis (queue `ai`, `render`, `publish`), FFmpeg trong container, MinIO cho asset, Caddy cho HTTPS.

## Chạy dev (không cần Docker)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .[dev]
copy .env.example .env         # điền GEMINI_API_KEY, v.v.
uvicorn app.main:app --reload
# http://localhost:8000/healthz
```

Chạy test:

```bash
pytest -q
ruff check app tests
```

## Chạy production (Docker Compose)

```bash
cp .env.example .env   # điền secret thật
docker compose up -d --build
docker compose exec web alembic upgrade head
```

Services: `web`, `worker-ai`, `worker-render`, `worker-publish`, `scheduler`, `postgres`, `redis`, `minio`, `caddy`.

## Cấu trúc mã

- `app/config.py` — settings + bảng model/giá có version (probe lúc khởi động)
- `app/states.py` — state machine pipeline (DRAFT → … → PUBLISHED)
- `app/models.py` — bảng: user, campaign, creative, source, script version, scene, asset, rendition, cost event, connected account, job, publish target/attempt, audit
- `app/schemas/videoplan.py` — VideoPlan v1 + semantic validator + auto-mode gate
- `app/ai/` — Gemini research/scripting, keyframe (Nano Banana), Veo LRO, TTS
- `app/media/` — FFmpeg builders, QC/ffprobe, caption/SRT, loudness
- `app/publishing/` — publisher contract + YouTube/Facebook/TikTok/Zalo + manual bundle
- `app/api/` — REST API, OAuth, webhooks, idempotency
- `app/workers/` — Celery app, tasks, scheduler
- `app/web/` — dashboard Jinja2 + HTMX

## Nguyên tắc vận hành

- Mọi thời gian lưu UTC, hiển thị `Asia/Ho_Chi_Minh`.
- Chi phí AI mỗi creative bị chặn cứng ở 6 USD (admin override phải có lý do, ghi audit).
- Chỉ retry 408/429/5xx với backoff + jitter; lỗi quyền/policy → `NEEDS_ACTION`.
- Trước khi retry sau timeout phải query trạng thái remote — không blind retry.
- TikTok luôn cần consent thủ công trong MVP; mọi kênh đều có manual bundle fallback.
