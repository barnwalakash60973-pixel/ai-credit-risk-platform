"""Form / POST /predict path: must agree with batch scoring and validate input."""
import numpy as np
import pandas as pd
import pytest

from src.python.models.features import MODEL_FEATURES
from src.python.scoring import live_predict as lp

BASE = dict(revolving_utilization=0.10, age=52, past_due_30_59=0, past_due_60_89=0, past_due_90=0, debt_ratio=0.25,
            monthly_income=7000.0, open_credit_lines=8, real_estate_loans=1, number_of_dependents=0)


def test_build_feature_row_has_all_model_features_and_derives_totals():
    df = lp.build_feature_row(BASE | dict(past_due_30_59=1, past_due_60_89=2, past_due_90=3))
    assert set(MODEL_FEATURES) <= set(df.columns)
    assert df.loc[0, "total_past_due"] == 6
    assert df.loc[0, "income_missing"] == 0


def test_income_not_reported_blanks_income_and_debt_ratio():
    df = lp.build_feature_row(BASE | dict(monthly_income=None, debt_ratio=None))
    assert df.loc[0, "income_missing"] == 1
    assert np.isnan(df.loc[0, "monthly_income"]) and np.isnan(df.loc[0, "debt_ratio"])
    # a debt ratio typed without income is ignored, like in training
    assert np.isnan(lp.build_feature_row(BASE | dict(monthly_income=None, debt_ratio=0.9)).loc[0, "debt_ratio"])


@pytest.mark.parametrize(
    "bad",
    [dict(age=10), dict(age=200), dict(revolving_utilization=-0.1), dict(past_due_90=-1), dict(debt_ratio=None),
     dict(open_credit_lines=None), dict(age="abc")],
)
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        lp.build_feature_row(BASE | bad)


def test_field_definitions_cover_the_inputs():
    assert {f["key"] for f in lp.INPUT_FIELDS} == set(BASE)
    for f in lp.INPUT_FIELDS:
        assert f["min"] <= f["default"] <= f["max"]


# ---- needs the trained champion (models/champion or MLflow) + the gold DuckDB: skipped on a fresh checkout
@pytest.fixture(scope="module")
def champion():
    try:
        lp._champion()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no trained model available: {e}")


def test_risky_applicant_is_rejected_and_clean_one_approved(champion):
    good = lp.predict_application(BASE)
    bad = lp.predict_application(BASE | dict(revolving_utilization=1.0, past_due_30_59=3, past_due_60_89=2, past_due_90=2,
                                             debt_ratio=0.8, monthly_income=2500.0))
    assert good["decision"] == "APPROVED" and bad["decision"] == "REJECTED"
    assert bad["predicted_proba"] > good["predicted_proba"]
    assert len(good["top_shap_features"]) == 5 and len(good["all_factors"]) == len(MODEL_FEATURES)


def test_matches_batch_scoring_for_real_applicants(champion):
    """Typing a real applicant's features into the form must reproduce the stored batch prediction."""
    duckdb = pytest.importorskip("duckdb")
    from src.python.config import DUCKDB_PATH

    try:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        df = con.execute("SELECT f.*, e.predicted_proba AS stored FROM applicant_feature_store f "
                         "JOIN prediction_explanations e USING (member_id_hash) USING SAMPLE 50").fetchdf()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"gold tables not available: {e}")
    keys = [f["key"] for f in lp.INPUT_FIELDS]
    for _, r in df.iterrows():
        inp = {k: (None if pd.isna(r[k]) else float(r[k])) for k in keys}
        # Both sides are rounded to 4 decimals; batch vs single-row XGBoost can differ in the last float bits,
        # which flips the rounding of a value sitting on a boundary -> allow 2 units of the 4th decimal.
        assert lp.predict_application(inp)["predicted_proba"] == pytest.approx(round(float(r["stored"]), 4), abs=2e-4)
