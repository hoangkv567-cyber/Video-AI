"""Asset storage: MinIO in production, a local directory in dev/test.

`AssetStore` is the minimal abstraction the workers need: content-addressed
puts, presigned time-limited GET URLs, byte reads and existence checks.
`store_asset` writes the file AND inserts the immutable `Asset` row (sha256,
model id, prompt hash, cost, ffprobe result) in one call; `find_asset` is the
resume-safety lookup workers run BEFORE calling any paid provider.

The minio SDK is imported lazily so dev/test never needs it installed or a
server running.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.errors import NotFound
from app.models import Asset

DEFAULT_URL_EXPIRES_SECONDS = 3600
LOCAL_ROOT_ENV = "ASSET_STORE_DIR"
BACKEND_ENV = "ASSET_STORE_BACKEND"
DEFAULT_LOCAL_ROOT = "./media_cache"

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_key(key: str) -> str:
    if not _KEY_RE.match(key) or ".." in key or key.startswith("/"):
        raise ValueError(f"invalid storage key: {key!r}")
    return key


@runtime_checkable
class AssetStore(Protocol):
    """Anything that can persist immutable blobs under string keys."""

    def put_bytes(self, key: str, data: bytes) -> tuple[str, int]:
        """Store bytes; returns (sha256_hex, size_bytes)."""
        ...

    def get_url(self, key: str, expires_seconds: int = DEFAULT_URL_EXPIRES_SECONDS) -> str:
        """Time-limited (presigned) HTTPS/file URL for the object."""
        ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class LocalDirBackend:
    """Filesystem-backed store for dev and tests (root from env or ./media_cache)."""

    def __init__(self, root: str | Path | None = None) -> None:
        raw = root if root is not None else os.environ.get(LOCAL_ROOT_ENV, DEFAULT_LOCAL_ROOT)
        self._root = Path(raw).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, key: str) -> Path:
        return self._root / _validate_key(key)

    def put_bytes(self, key: str, data: bytes) -> tuple[str, int]:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return sha256_hex(data), len(data)

    def get_url(self, key: str, expires_seconds: int = DEFAULT_URL_EXPIRES_SECONDS) -> str:
        # Dev-only presign analogue: a file URL carrying an explicit expiry stamp.
        expires_at = int(time.time()) + int(expires_seconds)
        return f"{self._path(key).as_uri()}?expires={expires_at}"

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise NotFound(f"asset object not found: {key}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()


class MinIOBackend:
    """MinIO/S3-backed store. The minio SDK and the client are created lazily."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        secure: bool | None = None,
        client: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self._endpoint = endpoint if endpoint is not None else cfg.minio_endpoint
        self._access_key = access_key if access_key is not None else cfg.minio_access_key
        self._secret_key = secret_key if secret_key is not None else cfg.minio_secret_key
        self._bucket = bucket if bucket is not None else cfg.minio_bucket
        self._secure = secure if secure is not None else cfg.minio_secure
        self._client = client  # injectable for tests; lazily built otherwise

    def _get_client(self) -> Any:
        if self._client is None:
            from minio import Minio  # lazy: not needed (or installed) in dev/test

            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
            )
        return self._client

    def _ensure_bucket(self, client: Any) -> None:
        if not client.bucket_exists(self._bucket):
            client.make_bucket(self._bucket)

    def put_bytes(self, key: str, data: bytes) -> tuple[str, int]:
        _validate_key(key)
        client = self._get_client()
        self._ensure_bucket(client)
        client.put_object(self._bucket, key, io.BytesIO(data), length=len(data))
        return sha256_hex(data), len(data)

    def get_url(self, key: str, expires_seconds: int = DEFAULT_URL_EXPIRES_SECONDS) -> str:
        client = self._get_client()
        return str(
            client.presigned_get_object(
                self._bucket, _validate_key(key), expires=timedelta(seconds=expires_seconds)
            )
        )

    def get_bytes(self, key: str) -> bytes:
        client = self._get_client()
        response = client.get_object(self._bucket, _validate_key(key))
        try:
            return bytes(response.read())
        finally:
            response.close()
            response.release_conn()

    def exists(self, key: str) -> bool:
        from minio.error import S3Error  # lazy

        client = self._get_client()
        try:
            client.stat_object(self._bucket, _validate_key(key))
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket"}:
                return False
            raise
        return True


def get_asset_store(settings: Settings | None = None) -> AssetStore:
    """Backend selection: env override first, else local for dev/test, MinIO otherwise."""
    cfg = settings or get_settings()
    backend = os.environ.get(BACKEND_ENV, "").strip().lower()
    if backend == "minio":
        return MinIOBackend(settings=cfg)
    if backend == "local" or cfg.app_env in {"dev", "test"}:
        return LocalDirBackend()
    return MinIOBackend(settings=cfg)


def asset_key(creative_id: str, kind: str, filename: str) -> str:
    return _validate_key(f"creatives/{creative_id}/{kind}/{filename}")


def store_asset(
    db: Session,
    store: AssetStore,
    *,
    creative_id: str,
    kind: str,
    data: bytes,
    filename: str,
    scene_id: str | None = None,
    locale: str | None = None,
    platform: str | None = None,
    model_id: str | None = None,
    prompt_hash: str | None = None,
    cost_usd: float = 0.0,
    ffprobe: dict[str, Any] | None = None,
) -> Asset:
    """Write the bytes to the store AND insert the immutable Asset row."""
    key = asset_key(creative_id, kind, filename)
    sha, size = store.put_bytes(key, data)
    row = Asset(
        creative_id=creative_id,
        scene_id=scene_id,
        kind=kind,
        locale=locale,
        platform=platform,
        storage_key=key,
        sha256=sha,
        size_bytes=size,
        model_id=model_id,
        prompt_hash=prompt_hash,
        cost_usd=cost_usd,
        ffprobe=ffprobe,
    )
    db.add(row)
    db.flush()
    return row


def find_asset(
    db: Session,
    creative_id: str,
    kind: str,
    *,
    scene_id: str | None = None,
    locale: str | None = None,
    prompt_hash: str | None = None,
    platform: str | None = None,
) -> Asset | None:
    """Resume-safety lookup: newest Asset matching every provided criterion, or None."""
    query = db.query(Asset).filter(Asset.creative_id == creative_id, Asset.kind == kind)
    if scene_id is not None:
        query = query.filter(Asset.scene_id == scene_id)
    if locale is not None:
        query = query.filter(Asset.locale == locale)
    if prompt_hash is not None:
        query = query.filter(Asset.prompt_hash == prompt_hash)
    if platform is not None:
        query = query.filter(Asset.platform == platform)
    return query.order_by(Asset.created_at.desc()).first()
