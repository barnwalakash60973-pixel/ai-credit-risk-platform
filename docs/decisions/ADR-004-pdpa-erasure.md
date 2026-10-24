# ADR-004: dbt Snapshots for PDPA Right-to-Erasure

## Status

Accepted

## Context

PDPA (and similar regulations) mandate a customer's right to request the erasure of personal data. However, financial institutions are simultaneously obligated to retain audit trails (typically for 5–10 years) for anti-money laundering regulations and credit dispute resolution.

## Decision

* **Gold layer**: Soft-delete — mask sensitive columns (`income`, `dti_ratio`, etc.) when an erasure request is processed, while keeping the row intact for referential integrity and decision history.
* **Bronze layer**: RETAIN in full (including PII encrypted via key vault) for audit trails, avoiding physical deletion during the mandated retention period.
* **Processing SLA**: 72 hours from request receipt.
* **dbt Snapshot**: `applicants_gold_snapshot` tracks before/after erasure history as SCD Type 2, providing verifiable evidence to regulators showing what was masked and precisely when.

## Justification

This balances individual privacy rights against regulatory data retention obligations: soft-deleting in Gold (used for analytics/modeling) satisfies user-facing erasure rights, while the Bronze audit trail fulfills organizational compliance requirements.


---

## Amendment (post-review): what erasure actually removes

The first implementation NULLed two columns of one DuckDB table and left everything else intact, which does not satisfy a right-to-erasure request. Current behaviour:

| Layer | Action | Why |
|---|---|---|
| `bronze.pii_key_vault` | **Row deleted** (crypto-shredding) | It is the only place a pseudonymous id can be linked back to a name/contact. |
| Gold (DuckDB): feature store, scorecard, SHAP explanations | Rows **deleted**; id added to `main.erasure_registry` | dbt staging anti-joins the registry, so a rebuild from Silver cannot resurrect the member. |
| Silver parquet + `silver.applicants_silver` | Rows deleted | Same data, earlier layer. |
| `gold.credit_decisions` | Rows deleted | Per-person decisions. |
| Bronze parquet | **Retained** (audit/retention) | Now unlinkable: with the vault row gone, the hashed record cannot be tied to a person. |
| `audit.erasure_audit_log` | Retained | Proof of request, SLA and outcome; contains only the pseudonymous hash. |

A request is marked `COMPLETED` only after verification finds no residual gold rows; otherwise `FAILED`.

**Open items for legal/DPO review (not decided by engineering):** (1) whether unlinkable-but-retained Bronze rows count as erased under the applicable regime; (2) the dbt snapshot (`applicants_gold_snapshot`) keeps historical rows of grade/label for erased members as an SCD2 audit trail. Remove them if the DPO does not accept that as audit data.

Further notes from end-to-end testing: Bronze ingest skips members whose erasure is `COMPLETED` (otherwise re-ingesting the source CSV would re-create them in the vault). Pseudonymous ids also remain in `audit.decision_request_log` (stores a hash of the letter, not its text) and in MLflow LLMOps run params; with the vault row deleted they are unlinkable, but confirm this with the DPO.
