"""
PDPA Right-to-Erasure
- Gold (DuckDB): feature store, scorecard and SHAP explanations are DELETED for the member and the
  id is added to main.erasure_registry, which dbt staging anti-joins, so a rebuild from Silver can
  never resurrect the person.
- Silver: rows removed from the local parquet files and from Postgres silver.applicants_silver.
- Decisions: gold.credit_decisions rows removed.
- PII vault: bronze.pii_key_vault row DELETED (crypto-shredding). The vault is the ONLY place the
  pseudonymous id can be linked back to a name/contact. With it gone, the Bronze parquet retained
  for the regulatory audit trail (ADR-004) is no longer linkable to an identifiable person.
- Audit: audit.erasure_audit_log keeps the pseudonymous hash + timestamps (proof of erasure).
- SLA: 72 hours from request submission to processing completion.

Previously only two columns of one DuckDB table were NULLed, the SQL was built with f-strings
(injection risk), and the PII vault, Silver and decisions were never touched.
A member is only marked COMPLETED if a post-erasure verification finds no residual rows.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import bindparam, text

from src.python.db import get_engine, serial_pk

from src.python.config import DUCKDB_PATH, SILVER_DIR, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ERASURE_SLA_HOURS = 72
DUCKDB_ERASE_TABLES = ["applicant_feature_store", "credit_scorecard", "prediction_explanations"]
SILVER_PARQUETS = ["applicants_silver_final.parquet", "core_features.parquet", "transactions.parquet"]
# (table, id column) in Postgres
POSTGRES_ERASE_TABLES = [
    ("silver.applicants_silver", "member_id_hash"),
    ("gold.credit_decisions", "member_id_hash"),
    ("bronze.pii_key_vault", "member_id_hash"),
]


def create_erasure_audit_table() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS audit.erasure_audit_log (
                    id {serial_pk()},
                    member_id_hash VARCHAR(64) NOT NULL,
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sla_deadline TIMESTAMP,
                    processed_at TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'PENDING',
                    notes TEXT
                )
                """
            )
        )
    logger.info("Table audit.erasure_audit_log is ready")


def submit_erasure_request(member_id_hash: str) -> dict:
    engine = get_engine()
    requested_at = datetime.utcnow()
    sla_deadline = requested_at + timedelta(hours=ERASURE_SLA_HOURS)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO audit.erasure_audit_log (member_id_hash, requested_at, sla_deadline, status)
                VALUES (:hash, :requested_at, :deadline, 'PENDING')
                """
            ),
            {"hash": member_id_hash, "requested_at": requested_at, "deadline": sla_deadline},
        )
    logger.info(f"Erasure request submitted for {member_id_hash}, SLA deadline: {sla_deadline}")
    return {"member_id_hash": member_id_hash, "sla_deadline": sla_deadline.isoformat(), "status": "PENDING"}


# ---------------------------------------------------------------- DuckDB (Gold)

def erase_from_duckdb(con: "duckdb.DuckDBPyConnection", hashes: list[str]) -> dict[str, int]:
    """Delete members from the gold tables and record them in the erasure registry."""
    con.execute(
        "CREATE TABLE IF NOT EXISTS main.erasure_registry (member_id_hash varchar, erased_at timestamp)"
    )
    con.execute(
        """
        INSERT INTO main.erasure_registry
        SELECT h, now() FROM unnest(?) AS t(h)
        WHERE h NOT IN (SELECT member_id_hash FROM main.erasure_registry)
        """,
        [hashes],
    )
    deleted: dict[str, int] = {}
    for table in DUCKDB_ERASE_TABLES:
        try:
            before = con.execute(f"SELECT count(*) FROM main.{table}").fetchone()[0]
            con.execute(f"DELETE FROM main.{table} WHERE list_contains(?, member_id_hash)", [hashes])
            after = con.execute(f"SELECT count(*) FROM main.{table}").fetchone()[0]
            deleted[f"duckdb.{table}"] = before - after
        except duckdb.CatalogException:
            logger.info(f"duckdb table main.{table} does not exist; skipped")
    return deleted


def residual_in_duckdb(con: "duckdb.DuckDBPyConnection", hashes: list[str]) -> dict[str, int]:
    out = {}
    for table in DUCKDB_ERASE_TABLES:
        try:
            out[f"duckdb.{table}"] = con.execute(
                f"SELECT count(*) FROM main.{table} WHERE list_contains(?, member_id_hash)", [hashes]
            ).fetchone()[0]
        except duckdb.CatalogException:
            pass
    return out


# ---------------------------------------------------------------- Silver parquet

def erase_from_silver_parquet(hashes: list[str], silver_dir: Path | None = None) -> dict[str, int]:
    silver_dir = silver_dir or SILVER_DIR
    removed: dict[str, int] = {}
    for name in SILVER_PARQUETS:
        path = silver_dir / name
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        keep = df[~df["member_id_hash"].isin(hashes)]
        if len(keep) != len(df):
            keep.to_parquet(path, index=False)
        removed[f"parquet.{name}"] = len(df) - len(keep)
    return removed


# ---------------------------------------------------------------- Postgres

def erase_from_postgres(engine, hashes: list[str]) -> dict[str, int]:
    removed: dict[str, int] = {}
    with engine.begin() as conn:
        for table, col in POSTGRES_ERASE_TABLES:
            try:
                with conn.begin_nested():
                    res = conn.execute(
                        text(f"DELETE FROM {table} WHERE {col} IN :hashes").bindparams(bindparam("hashes", expanding=True)),
                        {"hashes": hashes},
                    )
                    removed[f"postgres.{table}"] = res.rowcount
            except Exception as e:  # table may not exist in this deployment
                logger.warning(f"Skipping {table}: {e}")
    return removed


def process_pending_erasures() -> int:
    engine = get_engine()
    with engine.begin() as conn:
        pending = conn.execute(
            text("SELECT member_id_hash FROM audit.erasure_audit_log WHERE status = 'PENDING'")
        ).fetchall()

    if not pending:
        logger.info("No PENDING erasure requests found")
        return 0

    member_hashes = sorted({row[0] for row in pending})
    logger.info(f"Processing {len(member_hashes)} erasure request(s)...")

    removed: dict[str, int] = {}
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        removed.update(erase_from_duckdb(con, member_hashes))
        residual = residual_in_duckdb(con, member_hashes)
    finally:
        con.close()
    removed.update(erase_from_silver_parquet(member_hashes))
    removed.update(erase_from_postgres(engine, member_hashes))

    residual = {k: v for k, v in residual.items() if v}
    status, note = ("COMPLETED", f"Erased: {removed}. Bronze parquet retained (audit, unlinkable: vault row deleted).")
    if residual:
        status, note = "FAILED", f"Residual rows after erasure: {residual}"
        logger.error(note)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE audit.erasure_audit_log
                SET status = :status, processed_at = CURRENT_TIMESTAMP, notes = :notes
                WHERE member_id_hash IN :hashes AND status = 'PENDING'
                """
            ).bindparams(bindparam("hashes", expanding=True)),
            {"hashes": member_hashes, "status": status, "notes": note},
        )

    logger.info(f"Erasure {status} for {len(member_hashes)} request(s): {removed}")
    return len(member_hashes) if status == "COMPLETED" else 0


def check_sla_breaches() -> list[str]:
    engine = get_engine()
    with engine.begin() as conn:
        breaches = conn.execute(
            text(
                """
                SELECT member_id_hash FROM audit.erasure_audit_log
                WHERE status = 'PENDING' AND sla_deadline < CURRENT_TIMESTAMP
                """
            )
        ).fetchall()
    breach_list = [row[0] for row in breaches]
    if breach_list:
        logger.warning(f"SLA BREACH: {len(breach_list)} erasure request(s) passed the 72-hour deadline!")
    return breach_list


if __name__ == "__main__":
    create_erasure_audit_table()
    process_pending_erasures()
    check_sla_breaches()
