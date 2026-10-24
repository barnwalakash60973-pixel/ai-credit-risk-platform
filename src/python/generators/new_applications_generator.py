"""
New Applications Generator
Generate 1,000 new credit applications per day using Faker,
for daily scoring pipeline simulation (reused in Day 3 & Day 5).
"""
import logging
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from src.python.config import RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

fake = Faker()

# Columns according to Give Me Some Credit schema
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]


def generate_new_applications(n: int = 1000, seed: int | None = None) -> pd.DataFrame:
    """Generate n new credit applications with realistic distribution (no target label,
    since these are NEW applications with unknown outcomes -> to be scored by the model)."""
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    n_rows = n
    data = {
        "member_id": [str(uuid.uuid4()) for _ in range(n_rows)],
        "name": [fake.name() for _ in range(n_rows)],
        "contact": [fake.email() for _ in range(n_rows)],
        "RevolvingUtilizationOfUnsecuredLines": rng.beta(2, 5, n_rows).round(4),
        "age": rng.integers(21, 75, n_rows),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.4, n_rows),
        "DebtRatio": rng.gamma(2, 0.3, n_rows).round(4),
        "MonthlyIncome": rng.lognormal(mean=8.2, sigma=0.6, size=n_rows).round(2),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n_rows),
        "NumberOfTimes90DaysLate": rng.poisson(0.15, n_rows),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, n_rows),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.15, n_rows),
        "NumberOfDependents": rng.integers(0, 5, n_rows),
    }
    df = pd.DataFrame(data)
    return df


def save_new_applications(df: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or (RAW_DIR / "new_applications")
    out_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()
    out_path = out_dir / f"new_applications_{today_str}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"{len(df)} new applications saved to {out_path}")
    return out_path


if __name__ == "__main__":
    df = generate_new_applications(n=1000, seed=None)
    save_new_applications(df)