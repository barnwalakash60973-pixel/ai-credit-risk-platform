{% snapshot applicants_gold_snapshot %}

{{
    config(
        target_schema='snapshots',
        unique_key='member_id_hash',
        strategy='timestamp',
        updated_at='feature_store_updated_at',
        invalidate_hard_deletes=False
    )
}}

-- This snapshot serves as the foundation for PDPA right-to-erasure (Block 4):
-- erasure_requested & erasure_requested_at columns are updated via post-hook or manual update.
-- When an erasure request is received, dbt snapshot records the changes as a new row (SCD Type 2),
-- ensuring the "before and after erasure" history remains auditable without exposing raw data.

select
    member_id_hash,
    is_serious_delinquent,
    credit_grade,
    risk_tier_proxy,
    feature_store_updated_at::timestamp as feature_store_updated_at,
    false as erasure_requested,
    cast(null as timestamp) as erasure_requested_at
from {{ ref('credit_scorecard') }}

{% endsnapshot %}