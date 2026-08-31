"""Test suite for maintenance features (retention, disk guard, backup/restore, CLI)."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.sqlite3")

import pytest
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.cli import main
from app.db import Base
from app.maintenance.backup import (
    create_database_backup,
    read_backup_metadata,
    restore_database_backup,
)
from app.maintenance.disk_guard import check_disk_space
from app.maintenance.retention import (
    DEFAULT_FINAL_RETENTION_DAYS,
    DEFAULT_RAW_RETENTION_DAYS,
    pin_asset,
    prune_expired_assets,
)
from app.models import Asset, Campaign, Creative, User
from app.storage import LocalDirBackend


@pytest.fixture
def clean_db(tmp_path: Path):
    db_file = tmp_path / "maint_test.sqlite3"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        campaign = Campaign(name="Maintenance Campaign")
        session.add(campaign)
        session.flush()
        creative = Creative(campaign_id=campaign.id)
        session.add(creative)
        session.flush()
        user = User(email="test@example.com", name="Test User", password_hash="hash")
        session.add(user)
        session.commit()
        yield session, db_url, db_file


class TestRetentionEngine:
    def test_prune_raw_and_final_assets_by_retention_window(
        self, clean_db: tuple[Session, str, Path], tmp_path: Path
    ) -> None:
        session, _, _ = clean_db
        store_root = tmp_path / "store"
        store = LocalDirBackend(store_root)

        now = datetime.now(UTC)
        creative = session.query(Creative).first()
        assert creative is not None

        # 1. Old raw asset (8 days old) -> should be pruned
        k1, s1 = store.put_bytes("creatives/c1/keyframe/old_kf.jpg", b"old keyframe data")
        a1 = Asset(
            creative_id=creative.id,
            kind="keyframe",
            storage_key="creatives/c1/keyframe/old_kf.jpg",
            sha256=k1,
            size_bytes=s1,
            created_at=now - timedelta(days=8),
            pinned=False,
        )

        # 2. Fresh raw asset (2 days old) -> should be kept
        k2, s2 = store.put_bytes("creatives/c1/keyframe/new_kf.jpg", b"new keyframe data")
        a2 = Asset(
            creative_id=creative.id,
            kind="keyframe",
            storage_key="creatives/c1/keyframe/new_kf.jpg",
            sha256=k2,
            size_bytes=s2,
            created_at=now - timedelta(days=2),
            pinned=False,
        )

        # 3. Old pinned raw asset (15 days old) -> should be kept because pinned
        k3, s3 = store.put_bytes("creatives/c1/keyframe/pinned_kf.jpg", b"pinned keyframe data")
        a3 = Asset(
            creative_id=creative.id,
            kind="keyframe",
            storage_key="creatives/c1/keyframe/pinned_kf.jpg",
            sha256=k3,
            size_bytes=s3,
            created_at=now - timedelta(days=15),
            pinned=True,
        )

        # 4. Final rendition asset (10 days old) -> should be kept (final retention is 30 days)
        k4, s4 = store.put_bytes("creatives/c1/master/master10.mp4", b"master 10 days")
        a4 = Asset(
            creative_id=creative.id,
            kind="master",
            storage_key="creatives/c1/master/master10.mp4",
            sha256=k4,
            size_bytes=s4,
            created_at=now - timedelta(days=10),
            pinned=False,
        )

        # 5. Old final rendition asset (35 days old) -> should be pruned
        k5, s5 = store.put_bytes("creatives/c1/master/master35.mp4", b"master 35 days")
        a5 = Asset(
            creative_id=creative.id,
            kind="master",
            storage_key="creatives/c1/master/master35.mp4",
            sha256=k5,
            size_bytes=s5,
            created_at=now - timedelta(days=35),
            pinned=False,
        )

        session.add_all([a1, a2, a3, a4, a5])
        session.commit()

        # Test Dry Run first
        report_dry = prune_expired_assets(
            session,
            store=store,
            raw_days=DEFAULT_RAW_RETENTION_DAYS,
            final_days=DEFAULT_FINAL_RETENTION_DAYS,
            dry_run=True,
            now=now,
        )
        assert report_dry.dry_run is True
        assert report_dry.pruned_count == 2
        assert set(report_dry.deleted_asset_ids) == {a1.id, a5.id}
        assert store.exists("creatives/c1/keyframe/old_kf.jpg")
        assert store.exists("creatives/c1/master/master35.mp4")

        # Now perform actual pruning
        report_actual = prune_expired_assets(
            session,
            store=store,
            raw_days=DEFAULT_RAW_RETENTION_DAYS,
            final_days=DEFAULT_FINAL_RETENTION_DAYS,
            dry_run=False,
            now=now,
        )
        assert report_actual.dry_run is False
        assert report_actual.pruned_count == 2
        assert report_actual.skipped_pinned_count == 1

        # Check DB deletions
        assert session.get(Asset, a1.id) is None
        assert session.get(Asset, a2.id) is not None
        assert session.get(Asset, a3.id) is not None
        assert session.get(Asset, a4.id) is not None
        assert session.get(Asset, a5.id) is None

        # Check Storage deletions
        assert not store.exists("creatives/c1/keyframe/old_kf.jpg")
        assert store.exists("creatives/c1/keyframe/new_kf.jpg")
        assert store.exists("creatives/c1/keyframe/pinned_kf.jpg")
        assert store.exists("creatives/c1/master/master10.mp4")
        assert not store.exists("creatives/c1/master/master35.mp4")

    def test_pin_asset_helper(self, clean_db: tuple[Session, str, Path]) -> None:
        session, _, _ = clean_db
        creative = session.query(Creative).first()
        assert creative is not None

        asset = Asset(
            creative_id=creative.id,
            kind="thumbnail",
            storage_key="test_thumb.jpg",
            sha256="abc",
            pinned=False,
        )
        session.add(asset)
        session.commit()

        updated = pin_asset(session, asset.id, pinned=True)
        assert updated is not None
        assert updated.pinned is True
        assert session.get(Asset, asset.id).pinned is True


class TestDiskGuard:
    def test_check_disk_space_returns_valid_metrics(self, tmp_path: Path) -> None:
        status = check_disk_space(tmp_path, min_free_gb=0.001, min_free_percent=0.1)
        assert status.total_gb > 0
        assert status.free_gb >= 0
        assert 0.0 <= status.free_percent <= 100.0
        assert status.is_low_space is False
        assert status.warning is None

    def test_check_disk_space_triggers_warning_on_unrealistic_threshold(
        self, tmp_path: Path
    ) -> None:
        status = check_disk_space(tmp_path, min_free_gb=999999.0, min_free_percent=100.0)
        assert status.is_low_space is True
        assert status.warning is not None
        assert "Disk space low" in status.warning


class TestBackupAndRestore:
    def test_backup_and_restore_sqlite_with_encryption(
        self, clean_db: tuple[Session, str, Path], tmp_path: Path
    ) -> None:
        session, db_url, _ = clean_db
        enc_key = Fernet.generate_key().decode()

        backup_file = tmp_path / "db_backup.enc"
        out_path = create_database_backup(
            db_url=db_url,
            output_file=backup_file,
            encryption_key=enc_key,
        )
        assert out_path.is_file()

        # Read and check metadata header
        meta = read_backup_metadata(out_path)
        assert meta.format_version == 1
        assert meta.database_dialect == "sqlite"
        assert meta.original_size_bytes > 0
        assert meta.encrypted_size_bytes > 0

        # Restore into a fresh target DB
        target_db_file = tmp_path / "restored.sqlite3"
        target_db_url = f"sqlite:///{target_db_file}"

        restored = restore_database_backup(
            backup_file=out_path,
            target_db_url=target_db_url,
            encryption_key=enc_key,
        )
        assert restored is True
        assert target_db_file.is_file()

        # Verify restored DB contains the same data
        target_engine = create_engine(target_db_url)
        with Session(target_engine) as restored_session:
            user = restored_session.query(User).filter_by(email="test@example.com").one()
            assert user.name == "Test User"
            campaign = (
                restored_session.query(Campaign).filter_by(name="Maintenance Campaign").one()
            )
            assert campaign is not None

    def test_restore_fails_with_invalid_key(
        self, clean_db: tuple[Session, str, Path], tmp_path: Path
    ) -> None:
        _, db_url, _ = clean_db
        enc_key = Fernet.generate_key().decode()
        wrong_key = Fernet.generate_key().decode()

        backup_file = tmp_path / "backup_wrong_key.enc"
        create_database_backup(db_url=db_url, output_file=backup_file, encryption_key=enc_key)

        target_db_file = tmp_path / "target_fail.sqlite3"
        with pytest.raises(InvalidToken):
            restore_database_backup(
                backup_file=backup_file,
                target_db_url=f"sqlite:///{target_db_file}",
                encryption_key=wrong_key,
            )


class TestCliMaintenanceCommands:
    def test_cli_disk_guard(self, tmp_path: Path) -> None:
        code = main(
            [
                "disk-guard",
                "--path",
                str(tmp_path),
                "--min-free-gb",
                "0.001",
                "--min-free-percent",
                "0.1",
            ]
        )
        assert code == 0

    def test_cli_prune_assets_dry_run(self) -> None:
        code = main(["prune-assets", "--dry-run", "--days-raw", "1", "--days-final", "1"])
        assert code == 0
