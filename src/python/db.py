"""
Single place that decides WHICH SQL database the pipeline uses.

    POSTGRES_USER and POSTGRES_PASSWORD both set in .env  ->  PostgreSQL (as before)
    either one missing/empty                              ->  SQLite  (zero-setup fallback)

Override with DB_BACKEND=sqlite | postgres in .env (default: auto).

SQLite fallback layout - one file per layer, attached under the same schema names that the
Postgres setup uses, so existing SQL such as ``bronze.pii_key_vault`` or ``gold.credit_decisions``
works unchanged:

    data/sqlite/credit_risk.db   (main)
    data/sqlite/bronze.db  silver.db  gold.db  audit.db

Usage everywhere:  ``from src.python.db import get_engine``  ->  ``engine = get_engine()``
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime
from functools import lru_cache

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine

from src.python.config import DATA_DIR, settings

logger = logging.getLogger(__name__)

SQLITE_DIR = DATA_DIR / "sqlite"
SQLITE_SCHEMAS = ("bronze", "silver", "gold", "audit")

# Store datetimes the same way SQLAlchemy does ("YYYY-MM-DD HH:MM:SS.ffffff") so comparisons work
# and Python 3.12's deprecated default adapters are not used.
sqlite3.register_adapter(datetime, lambda d: d.strftime("%Y-%m-%d %H:%M:%S.%f"))
sqlite3.register_adapter(date, lambda d: d.isoformat())


# ----------------------------------------------------------------------------- backend choice
def postgres_configured() -> bool:
    return bool(settings.postgres_user.strip() and settings.postgres_password.strip())


def use_sqlite() -> bool:
    backend = settings.db_backend.strip().lower()
    if backend == "sqlite":
        return True
    if backend == "postgres":
        return False
    return not postgres_configured()


def backend_name() -> str:
    return "sqlite" if use_sqlite() else "postgresql"


# ----------------------------------------------------------------------------- engine
def _sqlite_engine() -> Engine:
    SQLITE_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{(SQLITE_DIR / 'credit_risk.db').as_posix()}")

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):
        dbapi_conn.isolation_level = None  # we emit BEGIN ourselves (needed for SAVEPOINT support)
        dbapi_conn.create_function("NOW", 0, lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        for schema in SQLITE_SCHEMAS:  # emulate Postgres schemas with attached files
            dbapi_conn.execute(f"ATTACH DATABASE '{(SQLITE_DIR / (schema + '.db')).as_posix()}' AS {schema}")

    @event.listens_for(engine, "begin")
    def _on_begin(conn):
        conn.exec_driver_sql("BEGIN")

    return engine


# Tables Postgres gets from sql/init_postgres.sql (docker). Same shape, SQLite syntax.
_SQLITE_INIT = [
    """CREATE TABLE IF NOT EXISTS bronze.pii_key_vault (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        member_id_hash VARCHAR(64) UNIQUE NOT NULL,
        encrypted_name BLOB,
        encrypted_contact BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS audit.data_lineage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pipeline_stage VARCHAR(50) NOT NULL,
        source_file VARCHAR(255),
        target_path VARCHAR(255),
        record_count INTEGER,
        ingest_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status VARCHAR(20) DEFAULT 'SUCCESS',
        notes TEXT)""",
    "CREATE INDEX IF NOT EXISTS audit.idx_lineage_stage ON data_lineage(pipeline_stage)",
    """CREATE TABLE IF NOT EXISTS audit.erasure_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        member_id_hash VARCHAR(64) NOT NULL,
        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        sla_deadline TIMESTAMP,
        processed_at TIMESTAMP,
        status VARCHAR(20) DEFAULT 'PENDING',
        notes TEXT)""",
    "CREATE INDEX IF NOT EXISTS audit.idx_erasure_hash ON erasure_audit_log(member_id_hash)",
    "CREATE INDEX IF NOT EXISTS audit.idx_erasure_status ON erasure_audit_log(status)",
]


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine for the configured backend."""
    if use_sqlite():
        engine = _sqlite_engine()
        with engine.begin() as conn:
            for ddl in _SQLITE_INIT:
                conn.execute(text(ddl))
        logger.info("Database backend: SQLite (%s) - no PostgreSQL credentials in .env", SQLITE_DIR)
        return engine
    logger.info("Database backend: PostgreSQL (%s:%s/%s)", settings.postgres_host, settings.postgres_port, settings.postgres_db)
    return create_engine(settings.postgres_dsn)


# ----------------------------------------------------------------------------- portable SQL helpers
def serial_pk() -> str:
    """Auto-increment primary key column definition for the active backend."""
    return "INTEGER PRIMARY KEY AUTOINCREMENT" if use_sqlite() else "SERIAL PRIMARY KEY"


def add_column_if_missing(conn, schema: str, table: str, column: str, ddl_type: str) -> None:
    """Portable replacement for ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` (SQLite lacks it)."""
    cols = {c["name"] for c in inspect(conn).get_columns(table, schema=schema)}
    if column not in cols:
        conn.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN {column} {ddl_type}"))
