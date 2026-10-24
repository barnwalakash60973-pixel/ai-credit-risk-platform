# ADR-006: Synthetic Categorical Enrichment

## Status

Accepted

## Context

The public *Give Me Some Credit* dataset contains only numeric credit features (utilization, DTI, delinquency counts, etc.) — there are no categorical columns such as `loan_purpose`, `employment_type`, `occupation_category`, or monthly transaction data. Data of this nature is sensitive and typically omitted from open-source datasets.

## Decision

Add synthetic categorical columns generated **deterministically** (seeded from `member_id_hash`, rather than purely random) to ensure reproducibility:

* `loan_purpose`, `employment_type` — to demonstrate one-hot encoding
* `occupation_category` — to demonstrate target encoding (leave-one-out)
* `credit_grade` — derived from a risk proxy (utilization + DTI + delinquency), **NOT** from the target label, ensuring safe usage as a model feature without target leakage
* Synthetic monthly transaction snapshot — to demonstrate `payment_velocity_3m`

## Justification

A strong portfolio project requires realistic categorical features to showcase production-style encoding pipelines during technical interviews (covering one-hot, ordinal, and leave-one-out target encoding). All enrichment logic is transparently documented and deterministic, allowing for honest framing to interviewers as a "generic dataset enriched to demonstrate production-grade feature engineering pipelines."

## Trade-offs

These synthetic features do NOT carry genuine predictive power toward the target variable (`credit_grade` is exempt from this issue as it is derived from real risk proxies). The primary model's AUC is evaluated using the original dataset features and their direct derivations (`dti_ratio`, `credit_utilization`, `delinquency_count_12m`), rather than relying on these synthetic categorical columns.

---

## Amendment (post-review): synthetic columns are not model features

`loan_purpose`, `employment_type`, `employment_years`, `income_consistency`, `income_stability_score`, `payment_velocity_3m` and the synthetic transactions are generated from a hash of the member id (or from columns the model already sees). They contain no independent information about default. They remain in the Silver/Gold layers for the demo (scorecard, dashboard), but `features.MODEL_FEATURES` excludes them and a unit test enforces it. `occupation_category` and its leave-one-out target encoding were removed entirely (see ADR-007).
