# ADR-003: SHAP over LIME for Explainability

## Status

Accepted

## Context

Credit scoring requires an explainability framework that can be audited by financial regulators and clearly understood by applicants (specifically regarding credit denial reasons).

## Decision

Use SHAP (TreeExplainer) instead of LIME.

## Justification

* **Consistency Property**: SHAP is grounded in theoretical Shapley values, guaranteeing mathematically consistent feature contributions unlike LIME, which relies on local surrogate sampling.
* **Computational Efficiency**: TreeExplainer provides exact, non-approximated calculations rapidly for tree-based models like XGBoost.
* **Additivity**: The sum of all SHAP values equals the difference between the final predicted probability and the baseline rate ($\text{predicted\_proba} - \text{base\_rate}$), making verification straightforward.
* **Regulatory Compliance**: Offers a far more defensible audit trail compared to LIME, which can produce unstable results across runs due to random perturbations.