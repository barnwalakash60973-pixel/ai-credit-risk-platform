"""Unit tests for PII pseudonymization."""
import pytest
from cryptography.fernet import Fernet

from src.python.config import settings
from src.python.loaders.bronze_ingest import (
    add_simulated_pii,
    encrypt_field,
    hash_member_id,
    drop_erased,
    pseudonymize,
)


@pytest.fixture(autouse=True)
def hash_key(monkeypatch):
    monkeypatch.setattr(settings, "pii_hash_key", "unit-test-secret-key")


def test_hash_member_id_is_deterministic():
    assert hash_member_id("abc-123") == hash_member_id("abc-123")


def test_hash_member_id_produces_sha256_length():
    assert len(hash_member_id("some-uuid")) == 64


def test_hash_is_keyed_not_plain_sha256():
    import hashlib

    assert hash_member_id("123456") != hashlib.sha256(b"123456").hexdigest()
    assert hash_member_id("123456", key=b"k1") != hash_member_id("123456", key=b"k2")


def test_missing_hash_key_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "pii_hash_key", "")
    with pytest.raises(RuntimeError, match="PII_HASH_KEY"):
        hash_member_id("x")


def test_encrypt_field_roundtrip():
    fernet = Fernet(Fernet.generate_key())
    encrypted = encrypt_field("John Doe", fernet)
    assert fernet.decrypt(encrypted).decode() == "John Doe"
    assert encrypted != b"John Doe"


def test_add_simulated_pii_adds_expected_columns(sample_credit_df):
    df = add_simulated_pii(sample_credit_df)
    for col in ("member_id", "name", "contact"):
        assert col in df.columns
    assert len(df) == len(sample_credit_df)


def test_member_ids_are_deterministic_so_reingest_is_idempotent(sample_credit_df):
    a = add_simulated_pii(sample_credit_df)
    b = add_simulated_pii(sample_credit_df)
    assert list(a["member_id"]) == list(b["member_id"])
    assert a["member_id"].is_unique


def test_pseudonymize_removes_raw_pii_from_bronze_df(sample_credit_df):
    df = add_simulated_pii(sample_credit_df)
    bronze_df, vault_df = pseudonymize(df, Fernet(Fernet.generate_key()))
    for col in ("member_id", "name", "contact"):
        assert col not in bronze_df.columns
    assert "member_id_hash" in bronze_df.columns
    assert len(vault_df) == len(df)
    assert set(vault_df.columns) == {"member_id_hash", "encrypted_name", "encrypted_contact"}


def test_reingest_does_not_resurrect_erased_members(sample_credit_df):
    df = add_simulated_pii(sample_credit_df)
    bronze_df, vault_df = pseudonymize(df, Fernet(Fernet.generate_key()))
    gone = {bronze_df["member_id_hash"].iloc[0]}
    b, v = drop_erased(bronze_df, vault_df, gone)
    assert gone.isdisjoint(set(b["member_id_hash"])) and gone.isdisjoint(set(v["member_id_hash"]))
    assert len(b) == len(bronze_df) - 1 and len(v) == len(vault_df) - 1
