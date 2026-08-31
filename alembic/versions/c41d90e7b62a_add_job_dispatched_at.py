"""add job dispatch recovery timestamp

Revision ID: c41d90e7b62a
Revises: a89e43c131e3
Create Date: 2026-08-22 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

revision = "c41d90e7b62a"
down_revision = "a89e43c131e3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(op.f("ix_jobs_dispatched_at"), "jobs", ["dispatched_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_jobs_dispatched_at"), table_name="jobs")
    op.drop_column("jobs", "dispatched_at")
