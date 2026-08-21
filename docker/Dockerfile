FROM python:3.12-slim

# FFmpeg lives in the container per PLAN.md; fonts for caption burn-in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-noto-core fonts-noto-cjk curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY pyproject.toml ./
RUN pip install --no-cache-dir .[google] && pip uninstall -y video-ai || true

COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
