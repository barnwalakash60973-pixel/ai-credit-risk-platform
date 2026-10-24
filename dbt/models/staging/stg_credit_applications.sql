{{
    config(
        materialized='view'
    )
}}

-- Staging layer on top of the final output from the Silver pipeline.
-- Purpose of staging: consistent renaming, data type casting, filtering invalid rows and
-- already-erased members (PDPA), without business transformations.
--
-- Columns are split in two groups:
--   * REAL features (revolving_utilization ... number_of_dependents): used by the model.
--   * DEMO-ONLY synthetic columns (payment_velocity_3m, employment_years, ...): derived from a
--     hash of the member id (ADR-006). Kept for the scorecard/dashboard demo, NOT used for training.

select
    member_id_hash,
    cast("SeriousDlqin2yrs" as integer)                    as is_serious_delinquent,

    -- real features
    cast("RevolvingUtilizationOfUnsecuredLines" as double) as revolving_utilization,
    cast(age as integer)                                   as age,
    cast("NumberOfTime30-59DaysPastDueNotWorse" as double) as past_due_30_59,
    cast("NumberOfTime60-89DaysPastDueNotWorse" as double) as past_due_60_89,
    cast("NumberOfTimes90DaysLate" as double)              as past_due_90,
    cast(total_past_due as double)                         as total_past_due,
    cast("DebtRatio" as double)                            as debt_ratio,
    cast("MonthlyIncome" as double)                        as monthly_income,
    cast(income_missing as integer)                        as income_missing,
    cast("NumberOfOpenCreditLinesAndLoans" as integer)     as open_credit_lines,
    cast("NumberRealEstateLoansOrLines" as integer)        as real_estate_loans,
    cast("NumberOfDependents" as integer)                  as number_of_dependents,

    -- heuristic scorecard value (function of the real features; not a model input)
    credit_grade,
    cast(credit_grade_ordinal as integer)                  as credit_grade_ordinal,

    -- demo-only synthetic columns (ADR-006)
    cast(payment_velocity_3m as double)                    as payment_velocity_3m,
    cast(income_stability_score as double)                 as income_stability_score,
    cast(employment_years as double)                       as employment_years,
    cast(income_consistency as double)                     as income_consistency

from read_parquet('{{ env_var("SILVER_APPLICANTS_PATH", "../data/silver/applicants_silver_final.parquet") }}')

where age > 0
  and member_id_hash not in (select member_id_hash from main.erasure_registry)
