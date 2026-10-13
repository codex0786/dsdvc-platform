"""Alembic environment: uses the application's metadata; works with a passed-in connection."""
from alembic import context
from sqlalchemy import engine_from_config, pool

from dsdvc.database.models import Base

config = context.config
target_metadata = Base.metadata


def run_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata,
                      literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online():
    conn = config.attributes.get("connection")
    if conn is None:
        eng = engine_from_config(config.get_section(config.config_ini_section) or {}, prefix="sqlalchemy.",
                                 poolclass=pool.NullPool)
        with eng.connect() as c:
            _run(c)
    else:
        _run(conn)


def _run(conn):
    context.configure(connection=conn, target_metadata=target_metadata,
                      render_as_batch=conn.dialect.name == "sqlite")
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
