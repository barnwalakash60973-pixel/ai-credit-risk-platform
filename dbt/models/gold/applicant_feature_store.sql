{{
  config(
    materialized='table'
  )
}}

with applications as (
    select * from {{ ref('stg_credit_applications') }}
),

transaction_velocity as (
    select
        member_id_hash,
        avg(delinquency_flag) as recomputed_payment_velocity_3m,
        count(*) as transaction_snapshot_count
    from {{ ref('stg_transactions') }}
    group by member_id_hash
),

final as (
    select
        a.member_id_hash,
        a.is_serious_delinquent,

        -- deterministic train/test assignment (~80/20) shared by training, scoring and reporting
        case when hash(a.member_id_hash) % 10 < 2 then 'test' else 'train' end as data_split,

        -- real model features
        a.revolving_utilization,
        a.age,
        a.past_due_30_59,
        a.past_due_60_89,
        a.past_due_90,
        a.total_past_due,
        a.debt_ratio,
        a.monthly_income,
        a.income_missing,
        a.open_credit_lines,
        a.real_estate_loans,
        a.number_of_dependents,

        -- scorecard value (not a model input)
        a.credit_grade,
        a.credit_grade_ordinal,

        -- demo-only synthetic columns (ADR-006), NOT model features
        a.payment_velocity_3m,
        coalesce(t.recomputed_payment_velocity_3m, a.payment_velocity_3m) as payment_velocity_3m_verified,
        a.income_stability_score,
        a.employment_years,
        a.income_consistency,

        current_timestamp as feature_store_updated_at
    from applications a
    left join transaction_velocity t
        on a.member_id_hash = t.member_id_hash
)

select * from final
