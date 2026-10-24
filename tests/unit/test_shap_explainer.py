import numpy as np

from src.python.explainability.shap_explainer import top_n_shap_features


def test_top_n_shap_features_returns_correct_count():
    shap_row = np.array([0.1, -0.5, 0.05, 0.3, -0.02, 0.9])
    features = ["a", "b", "c", "d", "e", "f"]
    result = top_n_shap_features(shap_row, features, n=3)
    assert len(result) == 3
    assert result[0]["feature"] == "f"  # |0.9| terbesar


def test_top_n_shap_features_direction_labels():
    shap_row = np.array([0.5, -0.5])
    features = ["increases", "decreases"]
    result = top_n_shap_features(shap_row, features, n=2)
    directions = {r["feature"]: r["direction"] for r in result}
    assert directions["increases"] == "increases_risk"
    assert directions["decreases"] == "decreases_risk"