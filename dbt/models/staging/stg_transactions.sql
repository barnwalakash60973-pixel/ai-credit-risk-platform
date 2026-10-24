{{
  config(
    materialized='view'
  )
}}

select
    member_id_hash,
    cast(month_offset as integer)      as month_offset,
    cast(snapshot_income as double)     as snapshot_income,
    cast(delinquency_flag as integer)   as delinquency_flag

from read_parquet('{{ env_var("SILVER_TRANSACTIONS_PATH", "../data/silver/transactions.parquet") }}')