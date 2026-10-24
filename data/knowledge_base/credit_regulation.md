# Simulated Consumer Credit Regulation (MAS Notice / OJK style)
> This document is a SIMULATION for portfolio project demonstration purposes.
> It is not an official regulatory document, but is structured following the
> general principles and structure of MAS (Monetary Authority of Singapore)
> and OJK credit assessment regulations.

## Section 1: General Credit Assessment Principles

### 1.1 Duty of Responsible Assessment
Every credit-granting institution must conduct a thorough creditworthiness
assessment before approving an application. The assessment must consider the
customer's ability to repay, not merely the value of collateral or credit score alone.

### 1.2 Non-Discrimination Principle
Credit decisions must not be based on race, religion, gender, or marital status.
Permitted factors as a basis for decisions include: payment history, debt-to-income
ratio (DTI), income stability, and historical credit record.

### 1.3 Debt-to-Income (DTI) Ratio Threshold
Customers with a DTI ratio exceeding 45% are considered high-risk for default and
must go through an additional approval process or be rejected, unless significant
mitigating factors exist (e.g., additional collateral, a co-signer with good credit,
or a large verified savings account).

## Section 2: Credit Utilization Requirements

### 2.1 Revolving Credit Utilization Threshold
Revolving credit utilization above 80% of the available limit indicates potential
over-leverage and short-term liquidity pressure. Applications with utilization above
this threshold require additional risk review.

### 2.2 Combination of High DTI and High Utilization
A combination of DTI above 45% AND credit utilization above 80% simultaneously is a
strong indicator of default risk. Lending institutions may reject applications with
this combination unless clearly documented mitigating factors exist.

## Section 3: Payment Delinquency History

### 3.1 Delinquency Classification
- **30-59 days**: Minor delinquency, recorded but does not automatically cause rejection.
- **60-89 days**: Moderate delinquency, adds significant risk weight to scoring.
- **90+ days**: Serious delinquency, a strong indicator of default risk and the
  primary basis for the "serious delinquent" classification in the scoring system.

### 3.2 Delinquency Frequency in the Last 12 Months
Customers with 2 or more delinquencies (any category) in the last 12 months are
classified as high-risk and require additional mitigating factors for approval.

## Section 4: Income and Employment Stability

### 4.1 Minimum Employment Tenure
Employment tenure of less than 1 year at the current job is considered an additional
risk factor, especially when combined with other risk indicators.

### 4.2 Income Consistency
Significant fluctuation in monthly income (income inconsistency) indicates financial
instability. An income_consistency score below 0.5 requires additional review of
income sources.

## Section 5: Risk Tier Classification

### 5.1 Risk Tier Definitions
- **LOW**: Predicted probability of default < 10%. Generally auto-approved.
- **MEDIUM**: Probability 10-30%. Approved with additional conditions (higher interest
  rate, lower limit).
- **HIGH**: Probability 30-60%. Requires manual review, generally rejected unless
  strong mitigating factors exist.
- **VERY_HIGH**: Probability > 60%. Rejected, except in exceptional cases with written
  justification from the risk committee.

## Section 6: Customer Rights (Right to Explanation)

### 6.1 Duty to Explain Decisions
Every credit rejection decision must be accompanied by a written explanation listing
at least 3 specific factors underlying the decision, along with each factor's relative
contribution to the final score.

### 6.2 Right to Appeal
Customers have the right to appeal a rejection decision within 30 calendar days from
the date the decision was issued, by submitting additional supporting documents.

### 6.3 Right to Access Data
Customers have the right to request a copy of the data used in the credit decision
process concerning them, including the factors considered by the scoring system.

## Section 7: Personal Data Protection (PDPA Compliance)

### 7.1 Data Minimization Principle
Personal data collected for credit assessment must be limited to data that is
relevant and necessary for the purpose of assessing creditworthiness.

### 7.2 Right to Erasure
Customers have the right to submit a request for the erasure of their personal data.
The institution must process this request within a maximum of 72 hours, while still
retaining the minimum audit trail data required by anti-money laundering regulations
for the applicable retention period (generally 5-10 years), but excluding that data
from active analytics and decision-making processes.

### 7.3 Audit Data Retention
Audit trail data (access logs, decision logs, aggregated transaction history) may be
retained beyond the active erasure period for the purpose of proving regulatory
compliance, with access restricted solely to audit and compliance investigation purposes.

## Section 8: Model Governance and Monitoring

### 8.1 Explainability Requirement
Automated scoring models (including machine learning models) must be able to explain
the individual factors contributing to each decision, and must not operate as a pure
"black box" in the context of decisions that significantly impact customers.

### 8.2 Model Drift Monitoring
Institutions must periodically monitor model performance stability (population
stability index or similar metrics) and perform retraining when a significant data
distribution shift is detected, to ensure decisions remain fair and accurate over time.