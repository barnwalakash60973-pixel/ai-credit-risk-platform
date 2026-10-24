import numpy as np

from src.python.monitoring.drift_monitor import calculate_psi


def test_identical_distributions_have_near_zero_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    assert calculate_psi(x[:10000], x[10000:]) < 0.02


def test_shifted_continuous_feature_is_flagged():
    rng = np.random.default_rng(0)
    assert calculate_psi(rng.normal(size=10000), rng.normal(2, 1, size=10000)) > 0.2


def test_values_outside_training_range_are_counted_not_dropped():
    base = np.random.default_rng(0).uniform(0, 1, 10000)
    assert calculate_psi(base, np.full(1000, 50.0)) > 0.2


def test_zero_inflated_count_feature_shift_is_detected():
    """Regression: past_due-style features (92% zeros) used to collapse to one bin -> PSI always 0."""
    rng = np.random.default_rng(0)
    base = (rng.random(20000) < 0.08).astype(int) * rng.integers(1, 6, 20000)
    shifted = (rng.random(20000) < 0.30).astype(int) * rng.integers(1, 6, 20000)
    assert calculate_psi(base, shifted) > 0.2
    assert calculate_psi(base, base.copy()) < 0.01
