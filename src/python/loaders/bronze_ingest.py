"""
Bronze Layer Ingestion
- Read raw CSV (Kaggle) + add simulated PII columns (member_id, name, contact)
- Keyed HMAC-SHA256 of member_id (one-way; unlike a bare SHA-256 it cannot be brute-forced
  from guessable ids such as sequential account numbers without the secret key)
- Fernet encryption for name/contact (reversible, key stored in PII key vault)
- Save mapping to PostgreSQL bronze.pii_key_vault
- Save output (without raw PII) to MinIO as parquet, partitioned by ingest_date
- Log to audit.data_lineage
"""
import hashlib
import hmac
import logging
import uuid
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from cryptography.fernet import Fernet
from faker import Faker
from sqlalchemy import text

from src.python.db import get_engine

from src.python.config import RAW_DIR, BRONZE_DIR, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

fake = Faker()


def get_or_create_encryption_key() -> bytes:
    """Retrieve Fernet key from .env, or generate a new one if missing (dev only)."""
    if settings.pii_encryption_key:
        return settings.pii_encryption_key.encode()
    key = Fernet.generate_key()
    logger.warning(
        "PII_ENCRYPTION_KEY not found in .env — generating new key (dev only). "
        f"Save this to .env: PII_ENCRYPTION_KEY={key.decode()}"
    )
    return key


def get_hash_key() -> bytes:
    """Secret key for HMAC pseudonymisation. Required: there is deliberately no silent default,
    because an unkeyed (or default-keyed) hash of a guessable id is trivially reversible."""
    if not settings.pii_hash_key:
        raise RuntimeError(
            "PII_HASH_KEY is not set. Generate one with `python -c \"import secrets;print(secrets.token_hex(32))\"` "
            "and add it to .env."
        )
    return settings.pii_hash_key.encode()


def add_simulated_pii(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add synthetic PII columns: member_id, name, contact (see ADR-002).

    member_id is derived deterministically from the source row index, so re-ingesting the same
    file yields the same ids (idempotent; no duplicate vault rows). Deterministic ids are
    guessable, which is exactly why hash_member_id is keyed."""
    Faker.seed(seed)
    df = df.copy()
    df["member_id"] = [str(uuid.uuid5(uuid.NAMESPACE_OID, f"member-{i}")) for i in df.index]
    df["name"] = [fake.name() for _ in range(len(df))]
    df["contact"] = [fake.email() for _ in range(len(df))]
    return df


def hash_member_id(member_id: str, key: bytes | None = None) -> str:
    """Keyed HMAC-SHA256 (64 hex chars), one-way pseudonymous identifier."""
    key = key if key is not None else get_hash_key()
    return hmac.new(key, member_id.encode(), hashlib.sha256).hexdigest()


def encrypt_field(value: str, fernet: Fernet) -> bytes:
    return fernet.encrypt(value.encode())


def pseudonymize(df: pd.DataFrame, fernet: Fernet) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
        bronze_df: dataframe WITHOUT raw PII (ready for MinIO)
        vault_df: hash -> encrypted PII mapping (ready for Postgres pii_key_vault)
    """
    df = df.copy()
    key = get_hash_key()
    df["member_id_hash"] = df["member_id"].apply(lambda m: hash_member_id(m, key))

    vault_records = []
    for _, row in df.iterrows():
        vault_records.append(
            {
                "member_id_hash": row["member_id_hash"],
                "encrypted_name": encrypt_field(row["name"], fernet),
                "encrypted_contact": encrypt_field(row["contact"], fernet),
            }
        )
    vault_df = pd.DataFrame(vault_records)

    bronze_df = df.drop(columns=["member_id", "name", "contact"])
    return bronze_df, vault_df


def get_erased_hashes() -> set[str]:
    """Members whose erasure request is COMPLETED. Re-ingesting the source file must never
    bring them back. If the audit table does not exist yet (first run) nobody has been erased."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT member_id_hash FROM audit.erasure_audit_log WHERE status = 'COMPLETED'")
            ).fetchall()
        return {r[0] for r in rows}
    except Exception as e:
        logger.warning(f"Could not read erasure log (assuming no erasures yet): {e}")
        return set()


def drop_erased(bronze_df: pd.DataFrame, vault_df: pd.DataFrame, erased: set[str]):
    """Remove erased members from both outputs of pseudonymize()."""
    if not erased:
        return bronze_df, vault_df
    b = bronze_df[~bronze_df["member_id_hash"].isin(erased)].reset_index(drop=True)
    v = vault_df[~vault_df["member_id_hash"].isin(erased)].reset_index(drop=True)
    logger.info(f"Skipped {len(bronze_df) - len(b)} previously erased member(s)")
    return b, v


def save_vault_to_postgres(vault_df: pd.DataFrame) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for _, row in vault_df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO bronze.pii_key_vault (member_id_hash, encrypted_name, encrypted_contact)
                    VALUES (:hash, :name, :contact)
                    ON CONFLICT (member_id_hash) DO NOTHING
                    """
                ),
                {
                    "hash": row["member_id_hash"],
                    "name": bytes(row["encrypted_name"]),
                    "contact": bytes(row["encrypted_contact"]),
                },
            )
    logger.info(f"Vault: {len(vault_df)} rows saved to bronze.pii_key_vault")


def log_lineage(stage: str, source_file: str, target_path: str, record_count: int, notes: str = "") -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO audit.data_lineage (pipeline_stage, source_file, target_path, record_count, notes)
                VALUES (:stage, :source, :target, :count, :notes)
                """
            ),
            {
                "stage": stage,
                "source": source_file,
                "target": target_path,
                "count": record_count,
                "notes": notes,
            },
        )
    logger.info(f"Lineage logged: {stage} | {record_count} records")


def save_bronze_parquet(bronze_df: pd.DataFrame, ingest_date: date) -> Path:
    """Save as local parquet, partition folder by ingest_date (ready for MinIO upload)."""
    partition_dir = BRONZE_DIR / f"ingest_date={ingest_date.isoformat()}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    out_path = partition_dir / "credit_applications.parquet"
    bronze_df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info(f"Bronze parquet saved: {out_path} ({len(bronze_df)} rows)")
    return out_path


def upload_to_minio(local_path: Path, ingest_date: date) -> str:
    """Upload parquet to MinIO bronze bucket."""
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_root_user,
        aws_secret_access_key=settings.minio_root_password,
    )
    object_key = f"ingest_date={ingest_date.isoformat()}/credit_applications.parquet"
    s3.upload_file(str(local_path), settings.minio_bucket_bronze, object_key)
    s3_uri = f"s3://{settings.minio_bucket_bronze}/{object_key}"
    logger.info(f"Uploaded to MinIO: {s3_uri}")
    return s3_uri


def run_bronze_ingest(csv_filename: str = "cs-training.csv") -> None:
    ingest_date = date.today()
    csv_path = RAW_DIR / csv_filename

    logger.info(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, index_col=0)

    logger.info("Adding simulated PII columns ...")
    df = add_simulated_pii(df)

    fernet = Fernet(get_or_create_encryption_key())

    logger.info("Pseudonymizing PII (hash + encrypt) ...")
    bronze_df, vault_df = pseudonymize(df, fernet)
    bronze_df, vault_df = drop_erased(bronze_df, vault_df, get_erased_hashes())

    logger.info("Saving vault to the SQL database ...")
    save_vault_to_postgres(vault_df)

    logger.info("Saving local bronze parquet ...")
    local_path = save_bronze_parquet(bronze_df, ingest_date)

    try:
        s3_uri = upload_to_minio(local_path, ingest_date)
        target_path = s3_uri
    except Exception as e:
        logger.warning(f"MinIO upload failed ({e}), using local path as target")
        target_path = str(local_path)

    log_lineage(
        stage="bronze_ingest",
        source_file=str(csv_path),
        target_path=target_path,
        record_count=len(bronze_df),
        notes="Initial Kaggle dataset ingest with PII pseudonymization",
    )

    logger.info("Bronze ingest completed.")


if __name__ == "__main__":
    run_bronze_ingest()