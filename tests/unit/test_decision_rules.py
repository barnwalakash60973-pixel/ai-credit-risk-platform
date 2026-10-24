import pytest

from src.python.scoring.decision_rules import decision_from_tier, risk_tier_from_proba


@pytest.mark.parametrize(
    "p,tier",
    [(0.0, "LOW"), (0.0999, "LOW"), (0.10, "MEDIUM"), (0.2999, "MEDIUM"), (0.30, "HIGH"),
     (0.5999, "HIGH"), (0.60, "VERY_HIGH"), (1.0, "VERY_HIGH")],
)
def test_tier_boundaries(p, tier):
    assert risk_tier_from_proba(p) == tier


def test_decision_mapping():
    assert decision_from_tier("LOW") == decision_from_tier("MEDIUM") == "APPROVED"
    assert decision_from_tier("HIGH") == decision_from_tier("VERY_HIGH") == "REJECTED"


def test_explainer_uses_the_same_rules():
    pytest.importorskip("chromadb")
    pytest.importorskip("requests")
    import src.python.rag.credit_decision_explainer as ex
    import src.python.scoring.decision_rules as rules

    assert ex.risk_tier_from_proba is rules.risk_tier_from_proba
