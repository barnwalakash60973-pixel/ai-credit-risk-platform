# ADR-002: PII Pseudonymization in the Bronze Layer

## Status

Accepted

## Context

The public *Give Me Some Credit* dataset does not contain original PII columns (e.g., names, contact info).
To demonstrate a privacy-by-design pipeline relevant for production lending systems, synthetic PII columns (`member_id`, `name`, `contact`) are generated during Bronze ingestion using Faker and immediately pseudonymized.

## Decision

* `member_id` → SHA-256 hash (one-way, non-reversible)
* `name`, `contact` → Fernet encryption (reversible, encryption key stored separately in `bronze.pii_key_vault`)
* Raw PII columns are NEVER saved to Bronze/Silver/Gold layers — only hashes and encrypted values are persisted.

## Justification

Privacy by design: sensitive data must be protected before entering the lakehouse rather than being masked after the fact. This reflects compliance requirements for Singapore's PDPA and similar regulations (such as GDPR and Indonesia's UU PDP).