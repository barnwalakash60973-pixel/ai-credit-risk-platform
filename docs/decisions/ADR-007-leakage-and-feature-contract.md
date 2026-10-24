# ADR-007: Leak-free features and a single feature contract

## Status

Accepted

## Context

Review of the first version found:

1. `occupation_category` was generated from a hash of the member id (pure noise) and then **leave-one-out target-encoded**. Leave-one-out encoding leaks the row's own label: within a category, a row's encoded value is lower when its label is 1. The model used it as its top feature (mean |SHAP| 1.52 vs 0.69 for the next) and reported test AUC 0.942, well above anything achievable on Give Me Some Credit (~0.87). A synthetic demonstration (random category, random labels) reaches AUC 0.90 the same way.
2. The Gold feature store dropped the real credit-bureau features (utilization, debt ratio, past-due counts), so the model trained mostly on synthetic columns.
3. Training used "all columns in the table", scoring a hardcoded list that zero-filled missing columns, and SHAP padded the matrix with `padded_feature_N` columns, so explanation names did not match the model's inputs.
4. Data quality: the past-due counts contain 96/98 sentinel codes; `DebtRatio` is an absolute amount, not a ratio, when income is missing.
5. Class re-weighting (`scale_pos_weight`) distorted probabilities that are then compared against fixed tier thresholds.

## Decision

- Remove the leaky encoding. Keep an out-of-fold `target_encode_oof` for any future real categorical, plus a canary test that shows naive LOO leaks and OOF does not.
- `src/python/models/features.py` is the single source of truth for model inputs. The feature list is stored with the MLflow model; training, scoring and SHAP all use it; a missing feature raises instead of being zero-filled.
- Clean sentinels to NULL, add `income_missing`, NULL `DebtRatio` where income is missing; leave NULLs to XGBoost (no 0-fill).
- Train without class re-weighting; log Brier score and calibration by decile; assert mean predicted probability ≈ base rate.
- Train/test split is a deterministic hash in dbt (`data_split`), shared by training, scoring and reporting. Hyperparameters are tuned by CV on the training split only; the test split is touched once.
- The champion alias moves only if a new model beats the current one on held-out AUC.
- A quality test fails if logged test AUC is outside 0.80-0.90 (too good to be true) or mean prediction deviates from the base rate.

## Consequences

Reported AUC drops from a leaked 0.94 to an honest ~0.86; the result is defensible. Decisions in `gold.credit_decisions` carry `data_split` so in-sample scores are never presented as out-of-sample.
