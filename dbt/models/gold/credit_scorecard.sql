{{
  config(
    materialized='table'
  )
}}

with features as (
    select * from {{ ref('applicant_feature_store') }}
)

select
    member_id_hash,
    is_serious_delinquent,
    age,
    payment_velocity_3m_verified as payment_velocity_3m,
    income_stability_score,
    credit_grade,
    credit_grade_ordinal,
    case
        when credit_grade in ('A', 'B') then 'LOW'
        when credit_grade = 'C' then 'MEDIUM'
        when credit_grade = 'D' then 'HIGH'
        else 'VERY_HIGH'
    end as risk_tier_proxy,
    monthly_income,
    number_of_dependents,
    feature_store_updated_at
from features