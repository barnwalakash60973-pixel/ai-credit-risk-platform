# Data Dictionary — Give Me Some Credit Dataset

## Data Source

Kaggle Competition: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

Row count: ~150,000 (training set) — project documentation mentions ~250K as it includes simulated new applications (see Day 1 Block 4).

## Original Columns (from Kaggle)

| Column | Type | Description | PII? |
| --- | --- | --- | --- |
| SeriousDlqin2yrs | int (0/1) | Target — experienced serious delinquency of 90+ days in 2 years | No |
| RevolvingUtilizationOfUnsecuredLines | float | Total balance on credit cards and personal lines of credit divided by the sum of credit limits | No |
| age | int | Age of borrower (years) | No (quasi-identifier) |
| NumberOfTime30-59DaysPastDueNotWorse | int | Number of times borrower has been 30-59 days past due in the last 2 years | No |
| DebtRatio | float | Monthly debt payments, alimony, living costs divided by monthly gross income | No |
| MonthlyIncome | float | Monthly income | No (financially sensitive) |
| NumberOfOpenCreditLinesAndLoans | int | Number of open loans (credit cards, mortgage, etc.) | No |
| NumberOfTimes90DaysLate | int | Number of times borrower has been 90+ days past due | No |
| NumberRealEstateLoansOrLines | int | Number of mortgage/real estate loans | No |
| NumberOfTime60-89DaysPastDueNotWorse | int | Number of times borrower has been 60-89 days past due | No |
| NumberOfDependents | int | Number of dependents (excluding self) | No |

## Simulated PII Columns (added in Bronze layer, Day 1 Block 3)

| Column | Type | Description | Handling |
| --- | --- | --- | --- |
| member_id | string | Unique customer ID (generated with Faker/UUID) | SHA-256 hash (one-way) |
| name | string | Full name of customer (simulated, Faker) | Fernet encryption (reversible, key in vault) |
| contact | string | Email/phone number of customer (simulated, Faker) | Fernet encryption (reversible, key in vault) |

## Target Distribution

* Majority class (SeriousDlqin2yrs = 0): ~93%
* Minority class (SeriousDlqin2yrs = 1): ~7% → **imbalanced dataset**, needs handling during modeling (Day 3): stratified k-fold, class weighting, or resampling.

## Data Quality Notes

* `MonthlyIncome` & `NumberOfDependents` have significant missing values → require imputation (Day 2 Block 1).
* `RevolvingUtilizationOfUnsecuredLines` & `DebtRatio` contain extreme outliers (>1 despite theoretically being a ratio ≤1-2) → require winsorizing (Day 2 Block 1).
* `age` = 0 in several rows → data quality issue, needs filtering/imputation.