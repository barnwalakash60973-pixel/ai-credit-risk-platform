"""
Guards the *champion model's logged metrics* against the two failure modes found in review:
too-good-to-be-true results (label leakage) and miscalibrated probabilities.
Skips when no champion exists (fresh checkout).
"""
import pytest

pytest.importorskip("mlflow")

from src.python.models.model_registry import get_champion_version, load_champion  # noqa: E402


@pytest.fixture(scope="module")
def champion():
    c = get_champion_version()
    if c is None:
        pytest.skip("No champion model registered yet")
    import mlflow

    return c, mlflow.get_run(c.run_id).data.metrics


def test_auc_is_plausible_for_give_me_some_credit(champion):
    _, m = champion
    # Published best-in-class on this dataset is ~0.87. Above 0.90 means leakage, not a better model.
    assert 0.80 <= m["test_auc"] <= 0.90, f"test_auc={m['test_auc']:.4f} is outside the plausible range"


def test_probabilities_are_calibrated_on_average(champion):
    _, m = champion
    assert abs(m["test_mean_pred"] - m["test_base_rate"]) < 0.01


def test_champion_ships_feature_list_without_forbidden_columns():
    from src.python.models.train_xgboost import FORBIDDEN_FEATURES

    if get_champion_version() is None:
        pytest.skip("No champion model registered yet")
    _, feats = load_champion()
    assert feats and not FORBIDDEN_FEATURES.intersection(feats)
    assert not [f for f in feats if "padded" in f or "target_enc" in f]
