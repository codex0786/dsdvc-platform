"""Schema migrations (Alembic) with a safe path for databases created before migrations existed."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from dsdvc.database.models import Base

MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    if url:
        cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _additive_sync(conn) -> list[str]:
    """For a pre-migration database: create missing tables and add missing *nullable* columns. Never drops
    or alters existing columns, so no data can be lost."""
    done = []
    Base.metadata.create_all(conn)
    insp = inspect(conn)
    for table in Base.metadata.sorted_tables:
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in have:
                if not col.nullable and col.default is None and col.server_default is None:
                    raise RuntimeError(f"cannot add NOT NULL column {table.name}.{col.name} automatically; run Alembic")
                ddl = col.type.compile(dialect=conn.dialect)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'))
                done.append(f"{table.name}.{col.name}")
    return done


_PROCESS_LOCK = threading.Lock()
_DONE: set[str] = set()


@contextmanager
def _migration_lock(engine):
    """Serialise migrations across threads and processes: a PostgreSQL advisory lock, or an exclusive
    file lock next to a SQLite database."""
    with _PROCESS_LOCK:
        if engine.dialect.name == "postgresql":
            with engine.connect() as c:
                c.execute(text("SELECT pg_advisory_lock(727274)"))
                try:
                    yield
                finally:
                    c.execute(text("SELECT pg_advisory_unlock(727274)"))
            return
        db_path = engine.url.database
        if engine.dialect.name == "sqlite" and db_path and db_path != ":memory:":
            import fcntl
            with open(f"{db_path}.migrate.lock", "w") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            return
        yield


def upgrade(engine) -> dict:
    key = str(engine.url)
    if key in _DONE:
        return {"mode": "already-current-in-process"}
    with _migration_lock(engine):
        if key in _DONE:
            return {"mode": "already-current-in-process"}
        res = _upgrade(engine)
        _DONE.add(key)
        return res


def _upgrade(engine) -> dict:
    with engine.begin() as conn:
        tables = set(inspect(conn).get_table_names())
        cfg = alembic_config()
        cfg.attributes["connection"] = conn
        if tables and "alembic_version" not in tables:
            added = _additive_sync(conn)
            command.stamp(cfg, "head")
            return {"mode": "legacy-sync+stamp", "added_columns": added}
        command.upgrade(cfg, "head")
        return {"mode": "alembic-upgrade"}
