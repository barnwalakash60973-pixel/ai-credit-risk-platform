"""PDPA erasure: SLA logic + the real DuckDB / parquet erasure paths (no Postgres needed)."""
from datetime import datetime, timedelta

import duckdb
import pandas as pd
import pytest

from src.python.compliance.pdpa_erasure import (
    DUCKDB_ERASE_TABLES,
    ERASURE_SLA_HOURS,
    erase_from_duckdb,
    erase_from_silver_parquet,
    residual_in_duckdb,
)


def test_sla_hours_constant_is_72():
    assert ERASURE_SLA_HOURS == 72


def test_sla_deadline_calculation():
    requested_at = datetime(2026, 1, 1, 12, 0, 0)
    assert requested_at + timedelta(hours=ERASURE_SLA_HOURS) == datetime(2026, 1, 4, 12, 0, 0)


@pytest.fixture
def con():
    c = duckdb.connect(":memory:")
    for t in DUCKDB_ERASE_TABLES:
        c.execute(f"CREATE TABLE main.{t} AS SELECT * FROM (VALUES ('h1', 1), ('h2', 2), ('h3', 3)) v(member_id_hash, x)")
    yield c
    c.close()


def test_erase_from_duckdb_removes_member_from_every_gold_table(con):
    deleted = erase_from_duckdb(con, ["h1", "h3"])
    assert all(deleted[f"duckdb.{t}"] == 2 for t in DUCKDB_ERASE_TABLES)
    assert all(v == 0 for v in residual_in_duckdb(con, ["h1", "h3"]).values())
    assert con.execute("SELECT member_id_hash FROM main.applicant_feature_store").fetchall() == [("h2",)]


def test_erasure_registry_records_ids_once_and_is_idempotent(con):
    erase_from_duckdb(con, ["h1"])
    erase_from_duckdb(con, ["h1"])
    assert con.execute("SELECT count(*) FROM main.erasure_registry WHERE member_id_hash='h1'").fetchone()[0] == 1


def test_hostile_ids_are_parameterised_not_interpolated(con):
    """The old implementation built SQL with f-strings."""
    erase_from_duckdb(con, ["x') OR 1=1; --"])
    assert con.execute("SELECT count(*) FROM main.applicant_feature_store").fetchone()[0] == 3


def test_missing_gold_table_is_skipped_not_fatal():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE main.applicant_feature_store AS SELECT 'h1' AS member_id_hash")
    assert erase_from_duckdb(c, ["h1"])["duckdb.applicant_feature_store"] == 1


def test_erase_from_silver_parquet(tmp_path):
    for name in ("applicants_silver_final.parquet", "core_features.parquet", "transactions.parquet"):
        pd.DataFrame({"member_id_hash": ["h1", "h2", "h2"], "v": [1, 2, 3]}).to_parquet(tmp_path / name)
    removed = erase_from_silver_parquet(["h2"], silver_dir=tmp_path)
    assert removed["parquet.applicants_silver_final.parquet"] == 2
    left = pd.read_parquet(tmp_path / "transactions.parquet")
    assert list(left["member_id_hash"]) == ["h1"]
