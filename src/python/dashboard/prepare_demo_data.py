"""
Prepare representative demo data for video recording —
ensure there are LOW, MEDIUM, HIGH, and VERY_HIGH risk tier applicant examples
so the waterfall & narrative demo shows variety.
"""
import logging
import os
from pathlib import Path
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "dbt/local_dev.duckdb")

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_demo_applicant_ids() -> dict:
    """Return one member_id_hash example per risk tier for the demo."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df = con.execute(
        """
        SELECT member_id_hash, predicted_proba,
            predicted_proba AS p
        FROM prediction_explanations
        """
    ).fetchdf()
    con.close()
    from src.python.scoring.decision_rules import risk_tier_from_proba

    df["risk_tier"] = df["p"].apply(risk_tier_from_proba)

    demo_ids = {}
    for tier in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]:
        subset = df[df["risk_tier"] == tier]
        if not subset.empty:
            demo_ids[tier] = subset.iloc[0]["member_id_hash"]

    logger.info(f"Demo applicant IDs ready: {demo_ids}")
    return demo_ids


if __name__ == "__main__":
    ids = get_demo_applicant_ids()
    for tier, mid in ids.items():
        print(f"{tier}: {mid}")