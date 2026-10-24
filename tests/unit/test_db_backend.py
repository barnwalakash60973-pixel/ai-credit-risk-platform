"""Backend selection: PostgreSQL only when BOTH user and password are set, else SQLite."""
import pytest

from src.python import db
from src.python.config import settings


@pytest.mark.parametrize(
    "backend,user,pw,expect_sqlite",
    [
        ("auto", "", "", True),
        ("auto", "credit_admin", "", True),
        ("auto", "", "secret", True),
        ("auto", "credit_admin", "secret", False),
        ("sqlite", "credit_admin", "secret", True),   # forced
        ("postgres", "", "", False),                    # forced
    ],
)
def test_backend_choice(monkeypatch, backend, user, pw, expect_sqlite):
    monkeypatch.setattr(settings, "db_backend", backend)
    monkeypatch.setattr(settings, "postgres_user", user)
    monkeypatch.setattr(settings, "postgres_password", pw)
    assert db.use_sqlite() is expect_sqlite


def test_sqlite_schemas_work(monkeypatch, tmp_path):
    """Schema-qualified names (bronze./silver./gold./audit.) must work on the SQLite fallback."""
    from sqlalchemy import text

    monkeypatch.setattr(settings, "db_backend", "sqlite")
    monkeypatch.setattr(db, "SQLITE_DIR", tmp_path)
    db.get_engine.cache_clear()
    engine = db.get_engine()
    with engine.begin() as c:
        c.execute(text("INSERT INTO bronze.pii_key_vault (member_id_hash) VALUES ('x')"))
        c.execute(text("INSERT INTO audit.data_lineage (pipeline_stage) VALUES ('bronze')"))
        assert c.execute(text("SELECT count(*) FROM bronze.pii_key_vault")).scalar() == 1
    db.get_engine.cache_clear()
