"""Small, non-secret-bearing operational commands for Video AI.

The admin bootstrap command deliberately never accepts a password as a command
line value: argv is commonly retained in shell history and process listings.
Use the interactive prompt, or ``--password-stdin`` when piping from a secret
manager.
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TextIO

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import AuditEvent, User
from app.states import Role
from app.web.auth import hash_password

MIN_ADMIN_PASSWORD_LENGTH = 14
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_ADMIN_BOOTSTRAP_LOCK_ID = 24_291_316_456_632_905


class BootstrapError(ValueError):
    """Safe operator-facing bootstrap failure (contains no secret material)."""


@dataclass(frozen=True)
class AdminBootstrapResult:
    user_id: str
    email: str
    created: bool


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if len(normalized) > 255 or not _EMAIL_RE.fullmatch(normalized):
        raise BootstrapError("provide a valid admin email address")
    return normalized


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized or len(normalized) > 255:
        raise BootstrapError("admin name must contain 1 to 255 characters")
    return normalized


def _validate_password(password: str) -> None:
    if len(password) < MIN_ADMIN_PASSWORD_LENGTH:
        raise BootstrapError(
            f"admin password must contain at least {MIN_ADMIN_PASSWORD_LENGTH} characters"
        )
    if not password.strip():
        raise BootstrapError("admin password cannot contain whitespace only")


def _lock_admin_bootstrap(db: Session) -> None:
    """Serialize first-admin creation on PostgreSQL; SQLite tests stay portable."""
    bind = db.get_bind()
    if bind.dialect.name == "postgresql":
        db.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": _ADMIN_BOOTSTRAP_LOCK_ID},
        )


def _bootstrap_status(db: Session, email: str) -> AdminBootstrapResult | None:
    existing = db.execute(
        select(User).where(func.lower(User.email) == email)
    ).scalar_one_or_none()
    if existing is not None:
        if existing.role != Role.ADMIN.value or not existing.is_active:
            raise BootstrapError(
                "an account with that email already exists but is not an active admin; "
                "refusing to elevate or reactivate it automatically"
            )
        return AdminBootstrapResult(user_id=existing.id, email=existing.email, created=False)

    another_admin = db.execute(
        select(User.id).where(
            User.role == Role.ADMIN.value,
            User.is_active.is_(True),
        )
    ).first()
    if another_admin is not None:
        raise BootstrapError(
            "an active admin already exists; bootstrap-admin only creates the first admin"
        )
    return None


def bootstrap_admin(
    db: Session, *, email: str, name: str, password: str
) -> AdminBootstrapResult:
    """Create the first admin idempotently without overwriting an existing account."""
    normalized_email = _normalize_email(email)
    _lock_admin_bootstrap(db)
    existing = _bootstrap_status(db, normalized_email)
    if existing is not None:
        return existing

    normalized_name = _normalize_name(name)
    _validate_password(password)
    user = User(
        email=normalized_email,
        name=normalized_name,
        role=Role.ADMIN.value,
        password_hash=hash_password(password),
        is_active=True,
    )
    db.add(user)
    try:
        db.flush()
        db.add(
            AuditEvent(
                actor_kind="system",
                action="admin_bootstrapped",
                entity_type="user",
                entity_id=user.id,
                data={"email": normalized_email},
            )
        )
        db.commit()
    except IntegrityError:
        # A concurrent bootstrap may have inserted the same email first. Never
        # overwrite its password or role; treat an active admin as the winner.
        db.rollback()
        winner = _bootstrap_status(db, normalized_email)
        if winner is not None:
            return winner
        raise BootstrapError("admin account could not be created safely") from None
    return AdminBootstrapResult(user_id=user.id, email=user.email, created=True)


def reset_password(db: Session, *, email: str, password: str) -> None:
    """Reset the password for an existing account."""
    normalized_email = _normalize_email(email)
    _validate_password(password)
    user = db.execute(
        select(User).where(func.lower(User.email) == normalized_email)
    ).scalar_one_or_none()
    if user is None:
        raise BootstrapError(f"account '{normalized_email}' not found")

    user.password_hash = hash_password(password)
    db.add(
        AuditEvent(
            actor_kind="system",
            action="password_reset",
            entity_type="user",
            entity_id=user.id,
            data={"email": normalized_email},
        )
    )
    db.commit()


def _read_password(*, password_stdin: bool, stdin: TextIO = sys.stdin, prompt_label: str = "Admin") -> str:
    if password_stdin:
        password = stdin.readline()
        if password == "":
            raise BootstrapError("no password was received on standard input")
        return password.rstrip("\r\n")

    if not stdin.isatty():
        raise BootstrapError(
            "interactive password entry requires a TTY; use --password-stdin "
            "with a trusted secret-manager pipe"
        )
    first = getpass.getpass(f"{prompt_label} password: ")
    second = getpass.getpass(f"Confirm {prompt_label.lower()} password: ")
    if first != second:
        raise BootstrapError("password confirmation does not match")
    return first


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    # bootstrap-admin
    bootstrap = commands.add_parser(
        "bootstrap-admin", description="Create the first administrator account safely."
    )
    bootstrap.add_argument("--email", required=True)
    bootstrap.add_argument("--name", default="Administrator")
    bootstrap.add_argument(
        "--password-stdin",
        action="store_true",
        help="read one password line from stdin; never place a password in argv",
    )

    # reset-password
    reset_pwd = commands.add_parser(
        "reset-password", description="Reset password for an existing account."
    )
    reset_pwd.add_argument("--email", required=True)
    reset_pwd.add_argument(
        "--password-stdin",
        action="store_true",
        help="read one password line from stdin; never place a password in argv",
    )

    # prune-assets
    prune = commands.add_parser(
        "prune-assets", description="Prune expired unpinned assets from storage and database."
    )
    prune.add_argument(
        "--days-raw",
        type=int,
        default=7,
        help="Retention period in days for raw/intermediate assets (default: 7)",
    )
    prune.add_argument(
        "--days-final",
        type=int,
        default=30,
        help="Retention period in days for final rendition assets (default: 30)",
    )
    prune.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate pruning without deleting files or DB records",
    )

    # disk-guard
    disk = commands.add_parser(
        "disk-guard", description="Check storage volume free space and alert if below threshold."
    )
    disk.add_argument("--path", default=None, help="Target directory path to inspect")
    disk.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="Minimum required free space in GB (default: 5.0)",
    )
    disk.add_argument(
        "--min-free-percent",
        type=float,
        default=10.0,
        help="Minimum required free space percentage (default: 10.0)",
    )

    # backup-db
    backup = commands.add_parser(
        "backup-db", description="Create an encrypted backup of the database."
    )
    backup.add_argument("--output", default=None, help="Output destination file path")
    backup.add_argument("--key", default=None, help="Fernet encryption key override")

    # restore-db
    restore = commands.add_parser(
        "restore-db", description="Decrypt and restore a database from an encrypted backup."
    )
    restore.add_argument("--input", required=True, help="Input backup file path")
    restore.add_argument("--target-url", default=None, help="Target database connection URL")
    restore.add_argument("--key", default=None, help="Fernet encryption key override")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "bootstrap-admin":
        try:
            normalized_email = _normalize_email(args.email)
            with SessionLocal() as db:
                existing = _bootstrap_status(db, normalized_email)
                if existing is not None:
                    print(f"Admin already exists: {existing.email}")
                    return 0
                password = _read_password(password_stdin=args.password_stdin)
                result = bootstrap_admin(
                    db,
                    email=normalized_email,
                    name=args.name,
                    password=password,
                )
        except (BootstrapError, EOFError, KeyboardInterrupt) as exc:
            message = str(exc) if isinstance(exc, BootstrapError) else "password entry cancelled"
            print(f"Bootstrap refused: {message}", file=sys.stderr)
            return 2
        except SQLAlchemyError:
            print(
                "Bootstrap failed: database unavailable or schema not migrated; "
                "run 'alembic upgrade head' and retry.",
                file=sys.stderr,
            )
            return 1

        action = "Created" if result.created else "Found"
        print(f"{action} active admin: {result.email}")
        return 0

    if args.command == "reset-password":
        try:
            password = _read_password(password_stdin=args.password_stdin, prompt_label="New")
            with SessionLocal() as db:
                reset_password(
                    db,
                    email=args.email,
                    password=password,
                )
            print(f"Password reset successfully for: {args.email.strip().lower()}")
            return 0
        except (BootstrapError, EOFError, KeyboardInterrupt) as exc:
            message = str(exc) if isinstance(exc, BootstrapError) else "password entry cancelled"
            print(f"Reset refused: {message}", file=sys.stderr)
            return 2
        except SQLAlchemyError:
            print("Database error during password reset.", file=sys.stderr)
            return 1

    if args.command == "prune-assets":
        from app.maintenance.retention import prune_expired_assets

        with SessionLocal() as db:
            report = prune_expired_assets(
                db,
                raw_days=args.days_raw,
                final_days=args.days_final,
                dry_run=args.dry_run,
            )
            mode_str = "DRY-RUN: Would prune" if report.dry_run else "Pruned"
            print(
                f"{mode_str} {report.pruned_count} asset(s) ({report.pruned_bytes / (1024*1024):.2f} MB), "
                f"skipped {report.skipped_pinned_count} pinned asset(s)."
            )
            if report.errors:
                print(f"Errors encountered ({len(report.errors)}):", file=sys.stderr)
                for err in report.errors:
                    print(f" - {err}", file=sys.stderr)
                return 1
        return 0

    if args.command == "disk-guard":
        from app.maintenance.disk_guard import check_disk_space

        status = check_disk_space(
            args.path,
            min_free_gb=args.min_free_gb,
            min_free_percent=args.min_free_percent,
        )
        print(
            f"Storage path: {status.path}\n"
            f"Total: {status.total_gb:.2f} GB | Used: {status.used_gb:.2f} GB | Free: {status.free_gb:.2f} GB ({status.free_percent:.1f}%)"
        )
        if status.is_low_space:
            print(f"WARNING: {status.warning}", file=sys.stderr)
            return 1
        print("OK: Disk space is within safety margins.")
        return 0

    if args.command == "backup-db":
        from app.maintenance.backup import create_database_backup

        try:
            out_path = create_database_backup(output_file=args.output, encryption_key=args.key)
            print(f"Encrypted database backup created successfully: {out_path}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Database backup failed: {exc}", file=sys.stderr)
            return 1

    if args.command == "restore-db":
        from app.maintenance.backup import restore_database_backup

        try:
            restore_database_backup(
                backup_file=args.input,
                target_db_url=args.target_url,
                encryption_key=args.key,
            )
            print(f"Database restored successfully from {args.input}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Database restoration failed: {exc}", file=sys.stderr)
            return 1

    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
