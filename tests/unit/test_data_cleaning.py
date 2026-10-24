"""Unit tests for cleaning functions. Uses a lightweight local Spark session."""
import pytest

from src.python.transformers.spark_session import get_spark_session


@pytest.fixture(scope="session")
def spark():
    spark_session = get_spark_session("pytest-data-cleaning")
    yield spark_session
    spark_session.stop()


def test_impute_missing_fills_dependents_only(spark):
    from src.python.transformers.data_cleaning import impute_missing

    df = spark.createDataFrame(
        [(1000.0, 2), (None, 1), (3000.0, None)], ["MonthlyIncome", "NumberOfDependents"]
    )
    result = impute_missing(df)
    assert result.filter(result.NumberOfDependents.isNull()).count() == 0
    # income is NOT imputed any more: unknown income is information, handled via income_missing
    assert result.filter(result.MonthlyIncome.isNull()).count() == 1


def test_replace_past_due_sentinels(spark):
    from src.python.transformers.data_cleaning import PAST_DUE_COLS, replace_past_due_sentinels

    df = spark.createDataFrame([(0, 96, 98), (2, 1, 0)], PAST_DUE_COLS)
    rows = replace_past_due_sentinels(df).collect()
    assert rows[0][PAST_DUE_COLS[1]] is None and rows[0][PAST_DUE_COLS[2]] is None
    assert rows[0][PAST_DUE_COLS[0]] == 0
    assert [rows[1][c] for c in PAST_DUE_COLS] == [2, 1, 0]


def test_missing_income_is_flagged_and_debt_ratio_nulled(spark):
    from src.python.transformers.data_cleaning import handle_missing_income

    df = spark.createDataFrame([(5000.0, 0.3), (None, 4200.0)], ["MonthlyIncome", "DebtRatio"])
    rows = handle_missing_income(df).collect()
    assert (rows[0]["income_missing"], rows[0]["DebtRatio"]) == (0, 0.3)
    assert rows[1]["income_missing"] == 1 and rows[1]["DebtRatio"] is None  # not a ratio when income unknown


def test_winsorize_caps_outliers(spark):
    from src.python.transformers.data_cleaning import winsorize_outliers

    data = [(0.1,) for _ in range(95)] + [(0.5,) for _ in range(4)] + [(100.0,)]
    df = spark.createDataFrame(data, ["DebtRatio"])
    result = winsorize_outliers(df, cols=["DebtRatio"])
    assert result.agg({"DebtRatio": "max"}).collect()[0][0] < 100.0


def test_winsorize_ignores_nulls(spark):
    from src.python.transformers.data_cleaning import winsorize_outliers

    df = spark.createDataFrame([(float(i),) for i in range(100)] + [(None,)], ["DebtRatio"])
    result = winsorize_outliers(df, cols=["DebtRatio"])
    assert result.filter(result.DebtRatio.isNull()).count() == 1


def test_filter_invalid_age_drops_zero(spark):
    from src.python.transformers.data_cleaning import filter_invalid_age

    df = spark.createDataFrame([(0,), (25,), (40,)], ["age"])
    assert filter_invalid_age(df).count() == 2
