FROM ghcr.io/astral-sh/uv:0.12.5 AS uv

FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/

# FFmpeg lives in the container per PLAN.md; fonts for caption burn-in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-noto-core fonts-noto-cjk curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/srv/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra google --extra free --no-install-project

COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv sync --locked --no-dev --extra google --extra free \
    && useradd --system --create-home --uid 10001 videoai

USER videoai

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
