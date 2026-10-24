"""Train -> score -> explain parity on a small synthetic frame (fast, no MLflow/DB needed)."""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from src.python.explainability.shap_explainer import (
    build_prediction_explanations,
    compute_shap_values,
    save_global_importance,
)
from src.python.models.features import ID_COL, MODEL_FEATURES, SPLIT_COL, TARGET_COL, build_feature_matrix
from src.python.models.train_xgboost import calculate_ks_stat, calibration_table, prepare_train_data
from src.python.scoring.scoring_pipeline import score_dataframe


def _gold_like(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) for c in MODEL_FEATURES})
    df["income_missing"] = (rng.random(n) < 0.2).astype(int)
    df.loc[df["income_missing"] == 1, "monthly_income"] = np.nan
    logit = -3 + 3 * df["revolving_utilization"] + 2 * df["total_past_due"]
    df[TARGET_COL] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df[ID_COL] = [f"id{i}" for i in range(n)]
    df[SPLIT_COL] = np.where(np.arange(n) % 5 == 0, "test", "train")
    return df


def test_prepare_train_data_uses_dbt_split_and_disjoint_sets():
    df = _gold_like()
    train, test, cols = prepare_train_data(df)
    assert set(train[ID_COL]).isdisjoint(test[ID_COL])
    assert len(train) + len(test) == len(df)
    assert cols == MODEL_FEATURES


def test_scoring_and_shap_use_identical_features_and_real_names(tmp_path):
    df = _gold_like()
    train, test, cols = prepare_train_data(df)
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3).fit(
        build_feature_matrix(train, cols).values, train[TARGET_COL]
    )

    scored = score_dataframe(df, model, cols)
    assert len(scored) == len(df) and scored["predicted_proba"].between(0, 1).all()
    assert set(scored["risk_tier"]) <= {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}

    X = build_feature_matrix(df, cols)
    sv = compute_shap_values(model, X)
    assert sv.shape == (len(df), len(cols))
    expl = build_prediction_explanations(df, sv, cols, model.predict_proba(X.values)[:, 1])
    names = {f["feature"] for js in expl["top_shap_features_json"] for f in json.loads(js)}
    assert names <= set(MODEL_FEATURES)
    assert not any("padded" in n for n in names)  # the old pipeline emitted padded_feature_N

    imp = save_global_importance(sv, cols, out_path=tmp_path / "imp.csv")
    assert list(imp["feature"].head(2)) != []
    assert set(imp["feature"]) == set(MODEL_FEATURES)


def test_ks_and_calibration_helpers():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert calculate_ks_stat(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    t = calibration_table((rng.random(5000) < p).astype(int), p)
    assert (abs(t["mean_pred"] - t["observed"]) < 0.08).all()  # perfectly calibrated by construction
