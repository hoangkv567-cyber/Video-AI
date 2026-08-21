"""Jinja display filters: UTC storage -> Asia/Ho_Chi_Minh display."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from app.config import get_settings


def display_tz() -> ZoneInfo:
    return ZoneInfo(get_settings().timezone_display)


def to_display_tz(dt: datetime | None) -> datetime | None:
    """Convert a stored UTC datetime (aware or naive-UTC from SQLite) to display tz."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(display_tz())


def format_hcm_time(dt: datetime | None, default: str = "—") -> str:
    """Format for display as ``HH:MM dd/mm/yyyy`` in Asia/Ho_Chi_Minh."""
    local = to_display_tz(dt)
    return local.strftime("%H:%M %d/%m/%Y") if local else default


def hcm_datetime_input(dt: datetime | None) -> str:
    """Format for an ``<input type=datetime-local>`` value in Asia/Ho_Chi_Minh."""
    local = to_display_tz(dt)
    return local.strftime("%Y-%m-%dT%H:%M") if local else ""
