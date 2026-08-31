"""Database backup and restore with Fernet encryption (PLAN.md §5 & Week 7).

Supports:
- SQLite backup and restore
- PostgreSQL backup and restore (via pg_dump / pg_restore or structured dump)
- AES-128-CBC encryption (Fernet) using TOKEN_ENCRYPTION_KEY or BACKUP_ENCRYPTION_KEY
- Backup verification and integrity checks
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from cryptography.fernet import Fernet

from app.config import get_settings


@dataclass
class BackupMetadata:
    format_version: int
    created_at: str
    database_dialect: str
    original_size_bytes: int
    encrypted_size_bytes: int
    sha256_plaintext: str


def _resolve_encryption_key(key: str | None = None) -> bytes:
    """Resolve Fernet encryption key from argument, environment, or settings."""
    if key:
        raw_key = key.strip()
    else:
        settings = get_settings()
        raw_key = (
            os.environ.get("BACKUP_ENCRYPTION_KEY")
            or settings.token_encryption_key
            or os.environ.get("TOKEN_ENCRYPTION_KEY", "")
        ).strip()

    if not raw_key:
        raise ValueError(
            "An encryption key is required. Set TOKEN_ENCRYPTION_KEY or BACKUP_ENCRYPTION_KEY."
        )
    return raw_key.encode() if isinstance(raw_key, str) else raw_key


def _dump_sqlite(db_path: Path) -> bytes:
    """Safely dump a SQLite database file using its backup API."""
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database file not found: {db_path}")

    source_conn = sqlite3.connect(str(db_path))
    dest_db = sqlite3.connect(":memory:")
    try:
        source_conn.backup(dest_db)
        # Export memory db to raw script / bytes
        script = "".join(dest_db.iterdump())
        return script.encode("utf-8")
    finally:
        source_conn.close()
        dest_db.close()


def _restore_sqlite(sql_bytes: bytes, target_path: Path) -> None:
    """Restore a SQLite database from dumped SQL bytes."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    conn = sqlite3.connect(str(target_path))
    try:
        conn.executescript(sql_bytes.decode("utf-8"))
        conn.commit()
    finally:
        conn.close()


def create_database_backup(
    db_url: str | None = None,
    output_file: str | Path | None = None,
    *,
    encryption_key: str | None = None,
) -> Path:
    """Create an encrypted backup file of the active database."""
    settings = get_settings()
    url_str = db_url or settings.database_url
    parsed = urlparse(url_str)
    dialect = parsed.scheme.split("+")[0]

    fernet = Fernet(_resolve_encryption_key(encryption_key))
    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")

    if dialect == "sqlite":
        # sqlite:///./path or sqlite:////path
        sqlite_file_str = url_str.replace("sqlite:///", "").replace("sqlite://", "")
        sqlite_path = Path(sqlite_file_str).resolve()
        raw_data = _dump_sqlite(sqlite_path)
    elif dialect in {"postgresql", "postgres"}:
        try:
            # Try running pg_dump
            proc = subprocess.run(
                ["pg_dump", url_str, "-Fc"],
                capture_output=True,
                check=True,
            )
            raw_data = proc.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "pg_dump tool not found or failed while backing up PostgreSQL database."
            ) from None
    else:
        raise ValueError(f"Unsupported database dialect for automated backup: {dialect}")

    import hashlib

    sha256 = hashlib.sha256(raw_data).hexdigest()
    encrypted_data = fernet.encrypt(raw_data)

    metadata = BackupMetadata(
        format_version=1,
        created_at=datetime.now(UTC).isoformat(),
        database_dialect=dialect,
        original_size_bytes=len(raw_data),
        encrypted_size_bytes=len(encrypted_data),
        sha256_plaintext=sha256,
    )

    if output_file is not None:
        out_path = Path(output_file).resolve()
    else:
        out_dir = Path("./backups").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"backup_{dialect}_{timestamp_str}.enc"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write container format: 4-byte header length + json header + encrypted data
    header_json = json.dumps(metadata.__dict__).encode("utf-8")
    header_len = len(header_json)

    with out_path.open("wb") as f:
        f.write(header_len.to_bytes(4, "big"))
        f.write(header_json)
        f.write(encrypted_data)

    return out_path


def read_backup_metadata(backup_file: str | Path) -> BackupMetadata:
    """Read the metadata header from an encrypted backup file without decrypting payload."""
    path = Path(backup_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Backup file not found: {path}")

    with path.open("rb") as f:
        len_bytes = f.read(4)
        if len(len_bytes) < 4:
            raise ValueError("Invalid backup file: corrupted header")
        header_len = int.from_bytes(len_bytes, "big")
        header_bytes = f.read(header_len)
        data = json.loads(header_bytes.decode("utf-8"))
        return BackupMetadata(**data)


def restore_database_backup(
    backup_file: str | Path,
    target_db_url: str | None = None,
    *,
    encryption_key: str | None = None,
) -> bool:
    """Decrypt and restore a database from an encrypted backup file."""
    path = Path(backup_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Backup file not found: {path}")

    fernet = Fernet(_resolve_encryption_key(encryption_key))

    with path.open("rb") as f:
        len_bytes = f.read(4)
        header_len = int.from_bytes(len_bytes, "big")
        header_bytes = f.read(header_len)
        metadata = BackupMetadata(**json.loads(header_bytes.decode("utf-8")))
        encrypted_data = f.read()

    # Decrypt and verify checksum
    raw_data = fernet.decrypt(encrypted_data)
    import hashlib

    calculated_sha = hashlib.sha256(raw_data).hexdigest()
    if calculated_sha != metadata.sha256_plaintext:
        raise ValueError("Backup integrity verification failed: SHA-256 mismatch")

    settings = get_settings()
    dest_url = target_db_url or settings.database_url
    parsed = urlparse(dest_url)
    dialect = parsed.scheme.split("+")[0]

    if dialect != metadata.database_dialect:
        raise ValueError(
            f"Dialect mismatch: backup is for '{metadata.database_dialect}', target is '{dialect}'"
        )

    if dialect == "sqlite":
        sqlite_file_str = dest_url.replace("sqlite:///", "").replace("sqlite://", "")
        target_path = Path(sqlite_file_str).resolve()
        _restore_sqlite(raw_data, target_path)
        return True
    elif dialect in {"postgresql", "postgres"}:
        try:
            subprocess.run(
                ["pg_restore", "--clean", "--if-exists", "-d", dest_url],
                input=raw_data,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "pg_restore tool not found or failed while restoring PostgreSQL database."
            ) from None

    return False
