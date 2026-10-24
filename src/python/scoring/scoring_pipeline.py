"""
Scoring pipeline.
Input: applicant features -> model feature matrix (same contract as training) -> champion model
       -> PostgreSQL gold.credit_decisions.

Two entry points:
  * score_and_save_all():   re-score the whole gold feature store (each row tagged with its
                            train/test split so in-sample scores are never mistaken for out-of-sample).
  * score_new_applications() + save_decisions(): daily new-application scoring (used by scoring_dag).
"""
import logging
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy import bindparam, text

from src.python.db import add_column_if_missing, get_engine, serial_pk

from src.python.config import DUCKDB_PATH, settings
from src.python.loaders.bronze_ingest import get_hash_key, hash_member_id
from src.python.models.features import (
    ID_COL,
    SPLIT_COL,
    build_feature_matrix,
    raw_to_model_features,
)
from src.python.models.model_registry import load_champion
from src.python.scoring.decision_rules import decision_from_tier, risk_tier_from_proba

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def create_credit_decisions_table() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS gold.credit_decisions (
                    id {serial_pk()},
                    member_id_hash VARCHAR(64),
                    scoring_date DATE DEFAULT CURRENT_DATE,
                    predicted_proba DOUBLE PRECISION,
                    risk_tier VARCHAR(20),
                    decision VARCHAR(20),
                    data_split VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        add_column_if_missing(conn, "gold", "credit_decisions", "data_split", "VARCHAR(10)")
    logger.info("Table gold.credit_decisions is ready")


def score_dataframe(df: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    """Score a frame that already has the model feature columns. Raises on missing features."""
    X = build_feature_matrix(df, feature_cols)
    proba = model.predict_proba(X.values)[:, 1]
    out = pd.DataFrame({ID_COL: df[ID_COL].values, "scoring_date": date.today(), "predicted_proba": proba.round(4)})
    out["risk_tier"] = out["predicted_proba"].apply(risk_tier_from_proba)
    out["decision"] = out["risk_tier"].apply(decision_from_tier)
    out["data_split"] = df[SPLIT_COL].values if SPLIT_COL in df.columns else "new"
    return out


def score_new_applications(csv_path: Path, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Score a CSV of NEW applications in the raw Kaggle schema (+ member_id)."""
    model, model_features = load_champion()
    feature_cols = feature_cols or model_features
    raw = pd.read_csv(csv_path)
    key = get_hash_key()
    raw[ID_COL] = raw["member_id"].apply(lambda m: hash_member_id(m, key))
    feats = raw_to_model_features(raw)
    return score_dataframe(feats, model, feature_cols)


def save_decisions(decisions: pd.DataFrame, replace_today: bool = True) -> None:
    """Append decisions; by default replace only today's rows OF THE SAME KIND (same data_split
    values), so a daily new-application run never wipes the full-portfolio re-score and vice versa.
    History from earlier days is preserved (the original implementation TRUNCATEd the table)."""
    create_credit_decisions_table()
    engine = get_engine()
    with engine.begin() as conn:
        if replace_today:
            splits = sorted(decisions["data_split"].dropna().unique().tolist())
            conn.execute(
                text("DELETE FROM gold.credit_decisions WHERE scoring_date = CURRENT_DATE AND data_split IN :splits")
                .bindparams(bindparam("splits", expanding=True)),
                {"splits": splits},
            )
    decisions.to_sql("credit_decisions", engine, schema="gold", if_exists="append", index=False, chunksize=5000)
    logger.info(f"SUCCESS: {len(decisions)} credit decisions saved to gold.credit_decisions")


def score_and_save_all() -> None:
    logger.info("Loading champion model from MLflow...")
    model, feature_cols = load_champion()

    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(f"DuckDB file not found at {DUCKDB_PATH}")
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute("SELECT * FROM applicant_feature_store").fetchdf()
    finally:
        con.close()
    if df.empty:
        logger.warning("No data found in DuckDB applicant_feature_store!")
        return

    logger.info(f"Scoring {len(df)} applicants with the champion model ({len(feature_cols)} features)...")
    save_decisions(score_dataframe(df, model, feature_cols))


if __name__ == "__main__":
    score_and_save_all()
