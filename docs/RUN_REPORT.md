# Credit Risk Scoring Platform - Pipeline Run Report

Run date: 2026-10-24 | Dataset: Kaggle *Give Me Some Credit* (`cs-training.csv`, 150,000 applicants) | Environment: 1 CPU, 3 GB RAM, Python 3.12, no PostgreSQL (SQLite fallback), no Ollama, no Hugging Face access.

## 1. Headline result

| Metric (held-out test, 30,290 applicants) | Value |
|---|---|
| AUC-ROC | **0.8594** (5-fold CV on train: 0.8677 +/- 0.0019) |
| KS statistic | 0.5681 |
| Gini | 0.7189 |
| Brier score | 0.0491 |
| Mean predicted vs observed default rate | 6.64% vs 6.65% |

The CV and test AUCs are close and the result is in the range expected for this public dataset, so there is no sign of label leakage (the project's own leakage guard also passed). The test split is the hash-assigned 20% (`hash(member_id_hash) % 10 < 2`); it depends on `PII_HASH_KEY`, so another installation will see about 0.86, not identical digits.

## 2. Pipeline steps executed

| Step | Outcome |
|---|---|
| Bronze ingest | 150,000 rows; member id HMAC-pseudonymized, name/contact Fernet-encrypted into `bronze.pii_key_vault`; MinIO upload skipped (no server), local parquet used |
| Silver cleaning (PySpark) | 149,999 rows (1 invalid age removed); 3,924 missing `NumberOfDependents` imputed (median 0); winsorized utilization, debt ratio, income; data-quality validation passed |
| Feature engineering / encoding | 22 -> 33 columns; correlated columns `credit_utilization`, `delinquency_count_12m` dropped (model inputs protected) |
| dbt (DuckDB) | `dbt run` 4 models OK; `dbt test` 28/28 passed; `dbt snapshot` OK |
| Training | Optuna 30 trials x 5-fold CV on train split only; registered as `credit_risk_xgboost` v1 and promoted to `champion` |
| SHAP | TreeExplainer over all 149,999 applicants; top-5 factors per applicant stored; global importance exported |
| Scoring | 149,999 decisions written to `gold.credit_decisions` |
| Drift monitoring | Run against both comparisons (section 6) |
| PDPA erasure | Job executed; no pending requests |
| API | `/health` 200; decision endpoint 200 with 100% SHAP coverage; unknown applicant 404; audit rows written (LLM call stubbed) |
| Dashboard | All 7 pages rendered without exceptions |
| Tests | Unit: 81 passed. Integration: 6 passed, 4 skipped (need sentence-transformers / Ollama) |

## 3. Model

Final hyperparameters: n_estimators=200, max_depth=4, learning_rate=0.0561, subsample=0.787, colsample_bytree=0.664, min_child_weight=9, reg_alpha=6.762, reg_lambda=0.853.
Features (12): revolving_utilization, age, past_due_30_59, past_due_60_89, past_due_90, total_past_due, debt_ratio, monthly_income, income_missing, open_credit_lines, real_estate_loans, number_of_dependents.

### Decision policy and confusion matrix (reject at predicted probability >= 0.30)

| | Predicted approve | Predicted reject |
|---|---|---|
| Actually fine | 27,362 | 913 |
| Actually defaulted | 1,221 | 794 |

Precision of rejections 46.5%, recall of defaulters 39.4%, share of applicants rejected 5.6%. Tiers: LOW < 0.10, MEDIUM < 0.30 (approved), HIGH < 0.60, VERY_HIGH (rejected). Roughly 6 in 10 defaulters are still approved at this threshold; the threshold is a business choice, not a model limit.

### Calibration (test set, by predicted-risk decile)

| Decile | n | Mean predicted | Observed default rate |
|---|---|---|---|
| 1 | 3029 | 0.48% | 0.30% |
| 2 | 3029 | 0.68% | 0.76% |
| 3 | 3029 | 0.90% | 0.96% |
| 4 | 3029 | 1.20% | 1.55% |
| 5 | 3029 | 1.69% | 1.68% |
| 6 | 3029 | 2.54% | 2.94% |
| 7 | 3029 | 3.97% | 4.16% |
| 8 | 3029 | 6.53% | 6.24% |
| 9 | 3029 | 11.66% | 11.72% |
| 10 | 3029 | 36.78% | 36.22% |

Probabilities can be read as real default rates, which is what makes the fixed tier thresholds meaningful.

### Global feature importance (mean |SHAP|, log-odds)

| feature | mean_abs_shap |
|---|---|
| revolving_utilization | 0.7371 |
| total_past_due | 0.6545 |
| age | 0.2132 |
| debt_ratio | 0.1487 |
| open_credit_lines | 0.1412 |
| past_due_90 | 0.1245 |
| real_estate_loans | 0.1059 |
| monthly_income | 0.0841 |
| past_due_60_89 | 0.0659 |
| past_due_30_59 | 0.0359 |
| number_of_dependents | 0.0209 |
| income_missing | 0.0011 |

Utilization and past-due history dominate; `income_missing` is effectively unused.

## 4. Scored portfolio (gold.credit_decisions)

| Tier | Decision | Applicants (train+test) |
|---|---|---|
| LOW | APPROVED | 124,853 |
| MEDIUM | APPROVED | 16,361 |
| HIGH | REJECTED | 7,227 |
| VERY_HIGH | REJECTED | 1,558 |

Scores on training rows are in-sample; each row carries its `data_split` so they are never mistaken for out-of-sample.

## 5. Problems found and fixed

1. `drift_monitor.py` had a no-op `__main__`; added `run_drift_report()` (absolute output path), DAG now delegates to it.
2. Integration test used a relative DuckDB path and hard-failed without the embedding backend; now uses the config path and skips cleanly.
3. `pyproject.toml` forced `apache-airflow` as a hard dependency (contradicting `requirements.txt`); moved to optional extra `airflow`.
4. README contained a stale orphan results table (AUC 0.88, "RAG < 5 s") and no reproducibility note; table removed, note added. **A stray `---|` line remains above the "Cost Estimate" heading and should be deleted.**
5. MLflow stores absolute artifact paths, so the registry breaks when the project is moved. Added `export_champion()` (`python -m src.python.models.model_registry --export`) and a fallback in `load_champion()` to `models/champion/`. Verified: with MLflow disabled the exported model reproduces test AUC 0.8594.

## 6. Drift monitoring

**A. Train vs held-out test (sanity check):** max PSI 0.0008, 0 alerts - the monitor does not raise false alarms.

| feature | psi | drift_alert |
|---|---|---|
| debt_ratio | 0.0008 | no |
| open_credit_lines | 0.0007 | no |
| monthly_income | 0.0006 | no |
| age | 0.0005 | no |
| total_past_due | 0.0005 | no |
| past_due_30_59 | 0.0003 | no |
| past_due_60_89 | 0.0003 | no |
| number_of_dependents | 0.0002 | no |
| revolving_utilization | 0.0001 | no |
| past_due_90 | 0.0001 | no |
| income_missing | 0.0000 | no |
| real_estate_loans | 0.0000 | no |

**B. Train vs 1,000 generated "new applications":** 9 of 12 features above 0.2. This is an artifact of the demo generator, which draws simple synthetic distributions (e.g. no missing incomes) unlike the real data. It shows the alerting works, not that the model is drifting.

| feature | psi | drift_alert |
|---|---|---|
| revolving_utilization | 2.5100 | YES |
| income_missing | 1.5491 | YES |
| number_of_dependents | 0.9487 | YES |
| debt_ratio | 0.6635 | YES |
| total_past_due | 0.6170 | YES |
| monthly_income | 0.4577 | YES |
| real_estate_loans | 0.4527 | YES |
| open_credit_lines | 0.3389 | YES |
| past_due_30_59 | 0.2487 | YES |
| age | 0.1701 | no |
| past_due_90 | 0.1659 | no |
| past_due_60_89 | 0.1342 | no |

## 7. Not verified / limitations

- Live RAG: `all-MiniLM-L6-v2` (Hugging Face) and Ollama `llama3.2` were unreachable, so real embeddings and real decision letters were not produced. Retrieval and prompt logic were validated with a stand-in embedder and a stubbed LLM.
- Knowledge base: the official RBI PDF is not in `data/knowledge_base/pdfs/`, so the simulated markdown fallback (19 chunks) is used; citations are to a demo document.
- Not tested: Docker, PostgreSQL, MinIO, Airflow, `dbt deps` from hub.getdbt.com (`dbt_utils` 1.4.1 was fetched from GitHub).
- Only one training run; the 30-trial search is modest. No fairness/bias analysis was performed (age is a model feature; review before any real lending use).
- The `age` feature and the 0.30 reject threshold are policy-sensitive; this is a portfolio/demo system, not a validated production credit model.

## 8. Using the results without re-running

- Model only: `models/champion/` (`model.json`, `feature_columns.json`, `metadata.json`, `predict_example.py`; needs only `xgboost` and `pandas`).
- Whole project with stored data, registry, DuckDB, SQLite and SHAP outputs: unzip `credit-risk-project-with-results.zip`, `pip install -r requirements.txt`, then e.g. `uvicorn src.python.api.main:app` or `streamlit run src/python/dashboard/app.py`. Dashboard metrics come from MLflow; if artifact loading fails after moving folders, the model still loads from `models/champion/`.
- `.env` (generated PII keys) is included in the project zip so hashes and the encrypted vault stay consistent. The data is synthetic, but do not publish the zip.
