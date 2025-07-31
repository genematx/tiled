"""Add node_id index in data_sources table

Revision ID: 67f24bca1c6b
Revises: e05e918092c3
Create Date: 2025-07-28 09:44:55.809681

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "67f24bca1c6b"
down_revision = "e05e918092c3"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("data_sources", schema=None) as batch_op:
        batch_op.create_index("idx_data_sources_node_id", ["node_id"])


def downgrade():
    with op.batch_alter_table("data_sources", schema=None) as batch_op:
        batch_op.drop_index("idx_data_sources_node_id")
