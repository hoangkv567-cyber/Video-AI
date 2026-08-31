"""guard one active Veo operation per scene

Revision ID: d82a7f4e9c10
Revises: c41d90e7b62a
Create Date: 2026-08-22 00:30:00.000000

"""

import sqlalchemy as sa

from alembic import op

revision = "d82a7f4e9c10"
down_revision = "c41d90e7b62a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "uq_jobs_active_veo_operation",
        "jobs",
        ["idempotency_key"],
        unique=True,
        postgresql_where=sa.text("kind = 'veo_operation' AND status = 'RUNNING'"),
        sqlite_where=sa.text("kind = 'veo_operation' AND status = 'RUNNING'"),
    )


def downgrade() -> None:
    op.drop_index("uq_jobs_active_veo_operation", table_name="jobs")
