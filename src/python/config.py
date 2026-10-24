"""Central configuration menggunakan pydantic-settings, load dari .env"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Kaggle
    kaggle_username: str = ""
    kaggle_key: str = ""

    # Database. If POSTGRES_USER or POSTGRES_PASSWORD is empty, the pipeline falls back to SQLite
    # (see src/python/db.py). DB_BACKEND=auto|sqlite|postgres forces a choice.
    db_backend: str = "auto"
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_db: str = "credit_risk_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # MinIO
    minio_root_user: str = "minioadmin"
    minio_root_password: str = ""
    minio_endpoint: str = "localhost:9000"
    minio_bucket_bronze: str = "bronze-layer"
    minio_bucket_silver: str = "silver-layer"
    minio_bucket_gold: str = "gold-layer"

    # PII
    pii_encryption_key: str = ""
    # Secret key for HMAC pseudonymisation of member ids (keep separate from the Fernet key).
    pii_hash_key: str = ""

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Absolute paths so scripts behave the same regardless of the working directory.
DUCKDB_PATH = BASE_DIR / "dbt" / "local_dev.duckdb"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

for d in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR):
    d.mkdir(parents=True, exist_ok=True)

settings = Settings()