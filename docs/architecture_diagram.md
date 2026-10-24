# Architecture Diagram — AI Credit Risk Scoring Platform

```
┌─────────────────┐     ┌──────────────────────────────────────────────┐
│   Kaggle API     │     │              BRONZE LAYER                    │
│ (Give Me Some    │────▶│  • PII Pseudonymization (SHA-256 + Fernet)   │
│  Credit dataset) │     │  • PostgreSQL pii_key_vault                  │
└─────────────────┘     │  • MinIO parquet (partitioned by ingest_date)│
        │                │  • Airflow DAG: ingest_bronze_dag            │
┌─────────────────┐     └──────────────────────┬───────────────────────┘
│  Faker Generator │                            │
│ (1000 apps/day)  │────────────────────────────┘
└─────────────────┘                             ▼
                          ┌──────────────────────────────────────────────┐
                          │              SILVER LAYER                    │
                          │  • PySpark: cleaning, imputation, winsorize  │
                          │  • Feature Eng: dti_ratio, utilization,      │
                          │    payment_velocity, delinquency_count       │
                          │  • Encoding: one-hot, ordinal, target (LOO)  │
                          │  • dbt staging models + tests                │
                          └──────────────────────┬───────────────────────┘
                                                  ▼
                          ┌──────────────────────────────────────────────┐
                          │               GOLD LAYER                     │
                          │  • dbt star schema (DuckDB)                  │
                          │    - applicant_feature_store                 │
                          │    - credit_scorecard                        │
                          │  • XGBoost + Optuna + MLflow (AUC/KS/Gini)   │
                          │  • SHAP TreeExplainer (top-5 per applicant)  │
                          │  • PDPA snapshot (dbt SCD Type 2)            │
                          │  • PSI drift monitoring (weekly)             │
                          └──────────────────────┬───────────────────────┘
                                                  ▼
                          ┌──────────────────────────────────────────────┐
                          │                AI LAYER                      │
                          │  • ChromaDB regulation knowledge base        │
                          │    (all-MiniLM-L6-v2 embeddings)             │
                          │  • RAG: SHAP + regulation → Ollama Mistral7B │
                          │  • FastAPI /credit-decision/{id}             │
                          │  • Audit log + LLMOps (MLflow)               │
                          └──────────────────────┬───────────────────────┘
                                                  ▼
                          ┌──────────────────────────────────────────────┐
                          │           SERVING / PRESENTATION             │
                          │  • Streamlit Dashboard (4 pages)              │
                          │  • PDPA erasure request UI                    │
                          └────────────────────────────────────────────────┘

Orchestration: Apache Airflow (ingest_bronze_dag, scoring_dag,
                erasure_request_dag, drift_monitor_dag)
```