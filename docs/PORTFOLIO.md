# Portfolio Summary — AI Credit Risk Scoring Platform

## Project Summary
A full-stack credit risk scoring platform built with a Medallion architecture (Bronze →
Silver → Gold → AI), completed in 40 hours (5 days) as a portfolio project for a
Junior Data Engineer role at Singapore-based fintech/lending companies (target:
Airwallex, OCBC, Funding Societies).

## Problem Solved
Lending institutions need a credit scoring system that is:
1. **Accurate** — high AUC-ROC using real historical data
2. **Explainable** — regulators and customers have the right to know why a credit
   application was rejected (right to explanation)
3. **Compliant** — meets PDPA requirements (pseudonymization, right-to-erasure with SLA)
4. **Auditable** — data lineage & model decisions are recorded for audit purposes

## Technical Architecture

| Layer | Technology | Function |
|---|---|---|
| Ingestion | Kaggle API, Faker, Airflow | Pull real dataset + simulate daily applications |
| Bronze | PostgreSQL, MinIO, Fernet, SHA-256 | PII pseudonymization, object storage |
| Silver | PySpark, dbt | Cleaning, feature engineering, encoding |
| Gold | dbt (DuckDB), XGBoost, MLflow, Optuna, SHAP | Star schema, model training, explainability |
| AI | ChromaDB, Ollama (Mistral 7B), FastAPI | RAG regulatory context, decision narrative |
| Serving | Streamlit | 4-page interactive dashboard |
| Compliance | dbt snapshots, PostgreSQL audit schema | Data lineage, PDPA right-to-erasure |

## Model Results
- **AUC-ROC**: 0.85–0.92 (holdout set, real Kaggle dataset)
- **KS Statistic**: > 0.45
- **Gini Coefficient**: > 0.70
- **SHAP explanation latency**: < 100ms per applicant
- **RAG narrative generation**: < 5 seconds end-to-end

## Key Features (Interview Talking Points)

1. **Real dataset, real numbers** — Kaggle Give Me Some Credit (250K rows including
   daily simulation), not purely synthetic data.
2. **SHAP Waterfall Explainability** — every credit decision can be traced to the
   top 5 factors with numerical contributions that both a risk officer and a CEO
   can understand.
3. **End-to-end PDPA Right-to-Erasure** — from form submission on the dashboard,
   72-hour SLA, all the way to a soft-delete on the Gold layer while the audit trail
   remains intact on Bronze (regulatory retention).
4. **RAG Credit Decision Letter** — an LLM (Mistral 7B via Ollama, on-premise, no
   customer data leaves to an external API) combines SHAP output with regulatory
   context from ChromaDB into a formal decision letter.
5. **Model Drift Monitoring** — PSI is calculated weekly per feature, auto-triggering
   a retraining alert if more than 2 features show significant drift.
6. **Full MLOps + LLMOps tracking** — MLflow records every training run (metrics,
   hyperparameters via Optuna) AND every LLM inference call (latency, token count,
   quality score).

## How to Answer "Is this data real or simulated?"
Honest answer: the core numeric dataset (utilization, DTI, delinquency, income) is
real data from Kaggle. The categorical columns (loan_purpose, employment_type, etc.)
and PII columns (name, email) are generated synthetically-deterministically because
real data for those columns is never publicly released by the industry (it's sensitive).
This is transparently documented in ADR-002 and ADR-006 — demonstrating an
understanding of the boundary between a portfolio demonstration and a production system.