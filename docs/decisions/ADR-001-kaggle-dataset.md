# ADR-001: Kaggle Dataset over Synthetic Data

## Status
Accepted

## Context
A realistic credit risk dataset is needed to train a model with performance that can
be defended statistically in an interview, rather than fabricated numbers or numbers
from an overly clean (fully synthetic) dataset.

## Decision
Use the public Kaggle dataset "Give Me Some Credit" (~150,000 rows, real label
distribution from the 2011 Kaggle competition) as the primary numeric data basis, and
add synthetic-deterministic categorical columns (see ADR-006) to demonstrate pipeline
features that are not available in any public dataset.

## Justification
- Real label distribution → AUC-ROC 0.85-0.92 is a number that can be empirically
  defended in an interview, not a fabricated number.
- This dataset is a widely recognized benchmark in the data science community, so
  model results can be compared against public leaderboards for validation.
- The imbalance ratio (~7% positive class) reflects real-world credit risk conditions,
  providing an opportunity to demonstrate imbalanced data handling techniques
  (stratified k-fold, scale_pos_weight) relevant to a data engineer/scientist role
  in fintech lending.

## Trade-off
This dataset is from 2011 (somewhat old) and does not specifically represent the
Singapore/Indonesia market — this is transparently documented as a portfolio
demonstration, not a production-ready model for actual deployment.