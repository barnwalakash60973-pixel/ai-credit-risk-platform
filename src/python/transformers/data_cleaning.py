"""
PySpark Data Cleaning
- Read all Bronze parquet partitions
- Replace the 96/98 sentinel codes in the past-due columns with NULL
- Flag missing income, and null DebtRatio where income is missing (see below)
- Median-impute NumberOfDependents (label-independent; XGBoost handles remaining NULLs natively)
- Winsorize heavy-tailed columns at the 1st/99th percentile
- Data quality validation
- Save to intermediate parquet (data/silver/intermediate/)

Data-quality facts about Give Me Some Credit that this module handles explicitly:
  * The three past-due count columns contain the codes 96 and 98 (about 270 rows). These are
    not real counts; left in, they inflate the correlation between the three columns to ~0.99.
  * When MonthlyIncome is missing (~20% of rows) DebtRatio is not a ratio but an absolute debt
    amount (median in the thousands). Treating it as a ratio corrupts the feature.
"""
import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.python.config import BRONZE_DIR, SILVER_DIR
from src.python.transformers.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_COLS_TO_IMPUTE = ["NumberOfDependents"]
WINSORIZE_COLS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
PAST_DUE_SENTINELS = [96, 98]


def load_bronze(spark) -> DataFrame:
    path = str(BRONZE_DIR / "ingest_date=*" / "*.parquet")
    logger.info(f"Reading Bronze parquet from {path}")
    df = spark.read.parquet(path)
    logger.info(f"Bronze loaded: {df.count()} rows, {len(df.columns)} columns")
    return df


def replace_past_due_sentinels(df: DataFrame) -> DataFrame:
    """96/98 are placeholder codes, not counts -> NULL."""
    for c in PAST_DUE_COLS:
        df = df.withColumn(c, F.when(F.col(c).isin(PAST_DUE_SENTINELS), None).otherwise(F.col(c)))
    return df


def handle_missing_income(df: DataFrame) -> DataFrame:
    """Add income_missing flag; DebtRatio is not a ratio when income is missing -> NULL."""
    df = df.withColumn("income_missing", F.col("MonthlyIncome").isNull().cast("int"))
    df = df.withColumn(
        "DebtRatio",
        F.when(F.col("MonthlyIncome").isNull(), None).otherwise(F.col("DebtRatio")),
    )
    return df


def impute_missing(df: DataFrame) -> DataFrame:
    """Median imputation for NumberOfDependents (statistic does not use the label)."""
    for col_name in NUMERIC_COLS_TO_IMPUTE:
        median_val = df.approxQuantile(col_name, [0.5], 0.001)[0]
        n_missing = df.filter(F.col(col_name).isNull()).count()
        df = df.fillna({col_name: median_val})
        logger.info(f"Imputed {col_name}: {n_missing} missing values -> median={median_val:.2f}")
    return df


def winsorize_outliers(df: DataFrame, cols: list[str] = None) -> DataFrame:
    """Cap outliers at 1st & 99th percentiles per column (NULLs are ignored)."""
    cols = cols or WINSORIZE_COLS
    for col_name in cols:
        p1, p99 = df.approxQuantile(col_name, [0.01, 0.99], 0.001)
        n_capped = df.filter((F.col(col_name) < p1) | (F.col(col_name) > p99)).count()
        df = df.withColumn(
            col_name,
            F.when(F.col(col_name) < p1, p1)
            .when(F.col(col_name) > p99, p99)
            .otherwise(F.col(col_name)),
        )
        logger.info(f"Winsorized {col_name}: capped {n_capped} rows to [{p1:.4f}, {p99:.4f}]")
    return df


def filter_invalid_age(df: DataFrame) -> DataFrame:
    """Data quality: drop rows with invalid age (0 or negative, original dataset has 1 row with age=0)."""
    before = df.count()
    df = df.filter(F.col("age") > 0)
    after = df.count()
    logger.info(f"Filtered invalid age rows: {before - after} dropped")
    return df


def validate_data_quality(df: DataFrame) -> None:
    """Fail loudly if the cleaning contract is violated."""
    for col_name in NUMERIC_COLS_TO_IMPUTE + ["age"]:
        n_null = df.filter(F.col(col_name).isNull()).count()
        assert n_null == 0, f"Validation FAILED: {col_name} still has {n_null} missing values"
    assert df.filter(F.col("age") <= 0).count() == 0, "Validation FAILED: age <= 0 still present"
    for c in PAST_DUE_COLS:
        n_bad = df.filter(F.col(c).isin(PAST_DUE_SENTINELS)).count()
        assert n_bad == 0, f"Validation FAILED: sentinel codes still present in {c}"
    n_inconsistent = df.filter(
        (F.col("income_missing") == 1) & F.col("DebtRatio").isNotNull()
    ).count()
    assert n_inconsistent == 0, "Validation FAILED: DebtRatio present where income is missing"
    logger.info("Data quality validation PASSED")


def run_cleaning_pipeline() -> str:
    spark = get_spark_session("bronze-to-silver-cleaning")
    try:
        df = load_bronze(spark)
        df = filter_invalid_age(df)
        df = replace_past_due_sentinels(df)
        df = handle_missing_income(df)
        df = impute_missing(df)
        df = winsorize_outliers(df)
        validate_data_quality(df)

        out_path = str(SILVER_DIR / "intermediate")
        df.coalesce(4).write.mode("overwrite").parquet(out_path)
        logger.info(f"Cleaned dataset saved to {out_path} ({df.count()} rows)")
        return out_path
    finally:
        spark.stop()


if __name__ == "__main__":
    run_cleaning_pipeline()
