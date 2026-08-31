"""Safe, idempotent administrator bootstrap."""

import pytest
from sqlalchemy.orm import Session

from app.cli import BootstrapError, bootstrap_admin
from app.models import AuditEvent, User
from app.states import Role
from app.web.auth import hash_password, verify_password


def test_bootstrap_admin_creates_hashed_active_admin_and_audit(db_session: Session) -> None:
    result = bootstrap_admin(
        db_session,
        email="  Owner@Example.com ",
        name="Video AI Owner",
        password="correct horse battery staple",
    )

    assert result.created is True
    assert result.email == "owner@example.com"
    user = db_session.get(User, result.user_id)
    assert user is not None
    assert user.role == Role.ADMIN.value
    assert user.is_active is True
    assert user.password_hash != "correct horse battery staple"
    assert verify_password("correct horse battery staple", user.password_hash)
    audit = db_session.query(AuditEvent).filter_by(action="admin_bootstrapped").one()
    assert audit.entity_id == user.id
    assert "correct horse" not in repr(audit.data)


def test_bootstrap_admin_is_idempotent_and_never_resets_password(db_session: Session) -> None:
    first = bootstrap_admin(
        db_session,
        email="admin@example.com",
        name="First name",
        password="first secure password",
    )
    original_hash = db_session.get(User, first.user_id).password_hash

    replay = bootstrap_admin(
        db_session,
        email="ADMIN@example.com",
        name="Replacement name",
        password="different secure password",
    )

    assert replay.created is False
    assert replay.user_id == first.user_id
    db_session.expire_all()
    user = db_session.get(User, first.user_id)
    assert user.password_hash == original_hash
    assert verify_password("first secure password", user.password_hash)
    assert not verify_password("different secure password", user.password_hash)
    assert db_session.query(AuditEvent).filter_by(action="admin_bootstrapped").count() == 1


def test_bootstrap_refuses_to_elevate_existing_non_admin(db_session: Session) -> None:
    editor = User(
        email="editor@example.com",
        name="Editor",
        role=Role.EDITOR.value,
        password_hash=hash_password("existing secure password"),
    )
    db_session.add(editor)
    db_session.commit()

    with pytest.raises(BootstrapError, match="refusing to elevate"):
        bootstrap_admin(
            db_session,
            email=editor.email,
            name="Admin now",
            password="new secure password",
        )
    db_session.refresh(editor)
    assert editor.role == Role.EDITOR.value


def test_bootstrap_refuses_a_second_admin_email(db_session: Session) -> None:
    first = bootstrap_admin(
        db_session,
        email="first@example.com",
        name="First Admin",
        password="first secure password",
    )

    with pytest.raises(BootstrapError, match="only creates the first admin"):
        bootstrap_admin(
            db_session,
            email="second@example.com",
            name="Second Admin",
            password="second secure password",
        )
    assert db_session.query(User).filter_by(role=Role.ADMIN.value).count() == 1
    assert db_session.get(User, first.user_id) is not None


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("not-an-email", "correct horse battery staple"),
        ("owner@example.com", "too-short"),
        ("owner@example.com", " " * 16),
    ],
)
def test_bootstrap_rejects_invalid_input(
    db_session: Session, email: str, password: str
) -> None:
    with pytest.raises(BootstrapError):
        bootstrap_admin(
            db_session,
            email=email,
            name="Owner",
            password=password,
        )
    assert db_session.query(User).count() == 0


def test_reset_password_updates_hash_and_audit(db_session: Session) -> None:
    from app.cli import reset_password

    admin = bootstrap_admin(
        db_session,
        email="reset_target@example.com",
        name="Reset User",
        password="old secure password 123",
    )
    old_hash = db_session.get(User, admin.user_id).password_hash

    reset_password(
        db_session,
        email="reset_target@example.com",
        password="new secure password 456",
    )

    db_session.expire_all()
    user = db_session.get(User, admin.user_id)
    assert user.password_hash != old_hash
    assert verify_password("new secure password 456", user.password_hash)
    assert not verify_password("old secure password 123", user.password_hash)

    audit = db_session.query(AuditEvent).filter_by(action="password_reset").one()
    assert audit.entity_id == user.id


def test_reset_password_unknown_email(db_session: Session) -> None:
    from app.cli import reset_password

    with pytest.raises(BootstrapError, match="not found"):
        reset_password(
            db_session,
            email="nonexistent@example.com",
            password="new secure password 456",
        )
