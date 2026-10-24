# ADR-005: XGBoost over Neural Networks

## Status

Accepted

## Context

Credit scoring models require high interpretability to fulfill regulatory "right to explanation" requirements—both applicants and regulatory bodies have the right to understand the precise factors driving credit rejection decisions.

## Decision

Use XGBoost (gradient boosted decision trees) instead of neural networks / deep learning architecture.

## Justification

* **Tree-Explainer Compatibility**: Tree-based models seamlessly integrate with SHAP TreeExplainer, providing exact, fast, and deterministic explainability calculations.
* **Auditability**: Neural networks require approximate explainers (e.g., DeepSHAP or Integrated Gradients), which are harder to audit and present to non-technical stakeholders (such as risk officers and regulators).
* **Tabular Data Efficiency**: On tabular datasets (~150K rows, 15–20 features), XGBoost performs optimally without requiring deep learning. Neural networks yield marginal accuracy gains while adding significant infrastructure overhead and black-box operational risks incompatible with regulated lending.
* **Industry Precedent**: Production credit scoring systems in regulated financial institutions predominantly use gradient boosting or logistic regression over deep learning specifically to preserve compliance and interpretability standards.