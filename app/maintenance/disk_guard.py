"""Disk guard monitor (PLAN.md §5 & Week 7).

Checks local volume / cache storage usage and alerts when free space
drops below safety thresholds (e.g. min 5.0 GB or 10% free space).
"""

from __future__ import annotations

import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiskStatus:
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    total_gb: float
    used_gb: float
    free_gb: float
    free_percent: float
    min_free_gb: float
    is_low_space: bool
    warning: str | None = None


def check_disk_space(
    target_path: str | Path | None = None,
    *,
    min_free_gb: float = 5.0,
    min_free_percent: float = 10.0,
) -> DiskStatus:
    """Check storage disk space on the specified path or the asset cache directory."""
    if target_path is not None:
        path = Path(target_path).resolve()
    else:
        cache_dir = os.environ.get("ASSET_STORE_DIR", "./media_cache")
        path = Path(cache_dir).resolve()

    with suppress(OSError):
        path.mkdir(parents=True, exist_ok=True)

    # Find the closest existing directory to inspect filesystem disk space
    check_dir = path
    while not check_dir.exists() and check_dir != check_dir.parent:
        check_dir = check_dir.parent

    usage = shutil.disk_usage(check_dir)

    total_gb = round(usage.total / (1024**3), 2)
    used_gb = round(usage.used / (1024**3), 2)
    free_gb = round(usage.free / (1024**3), 2)
    free_percent = round((usage.free / usage.total) * 100, 2) if usage.total > 0 else 0.0

    is_low_space = (free_gb < min_free_gb) or (free_percent < min_free_percent)

    warning = None
    if is_low_space:
        warning = (
            f"Disk space low on {path}: {free_gb:.2f} GB free ({free_percent:.1f}%), "
            f"threshold is {min_free_gb:.1f} GB / {min_free_percent:.1f}%"
        )

    return DiskStatus(
        path=str(path),
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        total_gb=total_gb,
        used_gb=used_gb,
        free_gb=free_gb,
        free_percent=free_percent,
        min_free_gb=min_free_gb,
        is_low_space=is_low_space,
        warning=warning,
    )
