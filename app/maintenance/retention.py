"""Asset retention and lifecycle pruning engine (PLAN.md §2 & §5).

Rules:
- Raw / intermediate assets (keyframe, clip, voice, styleboard) are retained for 7 days.
- Final rendition assets (master, derivative, thumbnail, caption, srt) are retained for 30 days.
- Any asset explicitly marked as pinned (Asset.pinned == True) is never pruned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Asset
from app.storage import AssetStore, get_asset_store

RAW_ASSET_KINDS = frozenset({"keyframe", "clip", "voice", "styleboard"})
FINAL_ASSET_KINDS = frozenset({"master", "derivative", "thumbnail", "caption", "srt"})

DEFAULT_RAW_RETENTION_DAYS = 7
DEFAULT_FINAL_RETENTION_DAYS = 30


@dataclass
class RetentionReport:
    raw_retention_days: int
    final_retention_days: int
    dry_run: bool
    pruned_count: int = 0
    pruned_bytes: int = 0
    skipped_pinned_count: int = 0
    deleted_asset_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def pin_asset(db: Session, asset_id: str, *, pinned: bool = True) -> Asset | None:
    """Pin or unpin an asset to protect it from automated retention pruning."""
    asset = db.get(Asset, asset_id)
    if asset is not None:
        asset.pinned = pinned
        db.commit()
    return asset


def find_expired_assets(
    db: Session,
    *,
    raw_days: int = DEFAULT_RAW_RETENTION_DAYS,
    final_days: int = DEFAULT_FINAL_RETENTION_DAYS,
    now: datetime | None = None,
) -> Sequence[Asset]:
    """Find all unpinned assets that have exceeded their retention threshold."""
    current_time = now or datetime.now(UTC)
    raw_cutoff = current_time - timedelta(days=raw_days)
    final_cutoff = current_time - timedelta(days=final_days)

    stmt = (
        select(Asset)
        .where(
            Asset.pinned.is_(False),
            or_(
                Asset.kind.in_(RAW_ASSET_KINDS) & (Asset.created_at < raw_cutoff),
                Asset.kind.in_(FINAL_ASSET_KINDS) & (Asset.created_at < final_cutoff),
                # Any other unexpected kinds fallback to raw retention period
                (~Asset.kind.in_(RAW_ASSET_KINDS | FINAL_ASSET_KINDS))
                & (Asset.created_at < raw_cutoff),
            ),
        )
        .order_by(Asset.created_at.asc())
    )
    return db.execute(stmt).scalars().all()


def prune_expired_assets(
    db: Session,
    store: AssetStore | None = None,
    *,
    raw_days: int = DEFAULT_RAW_RETENTION_DAYS,
    final_days: int = DEFAULT_FINAL_RETENTION_DAYS,
    dry_run: bool = False,
    now: datetime | None = None,
) -> RetentionReport:
    """Prune expired unpinned assets from storage and database."""
    asset_store = store or get_asset_store()
    current_time = now or datetime.now(UTC)

    # Count pinned assets for metrics
    pinned_count_stmt = select(Asset.id).where(Asset.pinned.is_(True))
    total_pinned = len(db.execute(pinned_count_stmt).scalars().all())

    report = RetentionReport(
        raw_retention_days=raw_days,
        final_retention_days=final_days,
        dry_run=dry_run,
        skipped_pinned_count=total_pinned,
    )

    expired_assets = find_expired_assets(db, raw_days=raw_days, final_days=final_days, now=current_time)

    for asset in expired_assets:
        asset_id = asset.id
        size_bytes = asset.size_bytes or 0
        storage_key = asset.storage_key

        if dry_run:
            report.pruned_count += 1
            report.pruned_bytes += size_bytes
            report.deleted_asset_ids.append(asset_id)
            continue

        try:
            # Delete from object store / filesystem
            if storage_key:
                try:
                    asset_store.delete(storage_key)
                except Exception as exc:  # noqa: BLE001
                    report.errors.append(f"Storage delete failed for {storage_key}: {exc}")

            # Delete from database
            db.delete(asset)
            db.commit()

            report.pruned_count += 1
            report.pruned_bytes += size_bytes
            report.deleted_asset_ids.append(asset_id)
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            report.errors.append(f"DB delete failed for asset {asset_id}: {exc}")

    return report
