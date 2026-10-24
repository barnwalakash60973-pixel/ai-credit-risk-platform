# 🏦 AI Credit Risk Scoring Platform

**Explainable credit risk scoring on 150,000 real applicants: XGBoost + SHAP decisions, regulation-grounded decision letters (local RAG), and a working PDPA right-to-erasure flow, served through a Streamlit dashboard and a FastAPI service.**

> Everything in this folder is **already computed**: trained model, scores, SHAP explanations, drift report and databases. To use the website you only install the requirements and start it. No pipeline re-run.

---

## 🚀 Quick start (website only)

Requires **Python 3.11+**.

```bash
pip install -r requirements.txt
streamlit run src/python/dashboard/app.py
```

Or run `run_website.bat` (Windows) / `./run_website.sh` (macOS/Linux). Open <http://localhost:8501>.

**Optional extras (not needed for pages 1-4):**

| Feature | What it needs |
|---|---|
| Page 5 *Regulation Search*, page 6 *Loan Advisor Chat*, decision-letter retrieval | Internet on **first use only**: the embedding model `all-MiniLM-L6-v2` (~90 MB) is downloaded and the search indexes are **built automatically** |
| AI-written decision letters and chat answers | [Ollama](https://ollama.com) running locally: `ollama pull llama3.2` then `ollama serve`. Without it these buttons show an error; everything else still works |
| REST API | `uvicorn src.python.api.main:app --port 8000` then <http://localhost:8000/docs> (already installed by `requirements.txt`) |

---

## 📊 What's in the website

| Page | Shows |
|---|---|
| 1 · Overview | Applications, approval rate, risk-tier mix, probability distribution |
| 2 · Model Performance | AUC / KS / Gini, confusion matrix, calibration, best hyperparameters, SHAP importance, PSI drift |
| 3 · Decision Viewer | Search an applicant, SHAP waterfall of the top-5 drivers, generate the formal decision letter |
| 4 · PDPA Compliance | Data lineage, erasure queue with 72 h SLA, live right-to-erasure form |
| 5 · Regulation Search | Semantic search over the regulation knowledge base, with citations |
| 6 · Loan Advisor Chat | Post-rejection Q&A grounded in the RBI FAME financial-literacy booklet |

The full numbers of the stored run are in **`docs/RUN_REPORT.md`**.

---

## 🎯 Results (held-out test set)

Hash-assigned 20% split (30,290 applicants, 6.65% default rate). Hyperparameters were tuned by 5-fold CV on the other 80% only (Optuna, 30 trials); the test set was scored once.

| Metric | Value |
|---|---|
| AUC-ROC | **0.859** (CV 0.868) |
| KS statistic | 0.568 |
| Gini | 0.719 |
| Brier score | 0.049 (mean predicted ≈ observed default rate) |
| Reject rule | predicted default probability ≥ 0.30 → precision 46.5%, recall 39.4%, 5.6% of applicants rejected |

Predicted risk tracks observed default rate closely in every decile (well calibrated). An earlier version reported AUC 0.94; that was label leakage and was removed (ADR-007). A test fails the build if the logged AUC looks implausibly high.

> **Reproducibility:** the split is `hash(member_id_hash) % 10` and `member_id_hash` is a keyed HMAC, so metrics are deterministic *for a given `PII_HASH_KEY`* and differ slightly between installations (expect AUC ≈ 0.86 ± 0.005).

### Use the model on its own

`models/champion/` holds the champion as plain files. It needs only `xgboost` and `pandas`:

```bash
python models/champion/predict_example.py
```

Files: `model.json`, `feature_columns.json`, `metadata.json` (run id, metrics, hyperparameters), `predict_example.py`. The project loads this copy automatically if MLflow is unavailable.

---

## 🏗 Architecture

```
Kaggle "Give Me Some Credit" + Faker new applications
        │
   BRONZE   HMAC-SHA256 pseudonym id, Fernet-encrypted name/contact, PII key vault, lineage log
        │
   SILVER   PySpark cleaning / imputation / winsorizing → feature engineering → encoding
        │
   GOLD     dbt star schema (DuckDB) · XGBoost + Optuna + MLflow · SHAP top-5 per applicant
            PSI drift monitor · PDPA snapshot (SCD Type 2)
        │
   AI       ChromaDB regulation index · SHAP + regulation → local Ollama letter · audit log
        │
   FastAPI  /credit-decision/{id}   →   Streamlit dashboard (6 pages)
```

PII is protected at the point of ingest (Bronze), not retrofitted later. Erasure soft-deletes Gold data and removes the encryption link while Bronze keeps the unlinkable audit trail regulators require (ADR-004).

| Layer | Technology |
|---|---|
| Orchestration / storage | Airflow, PostgreSQL *(or built-in SQLite)*, MinIO, DuckDB |
| Processing / transform | PySpark, dbt |
| ML / explainability | XGBoost, Optuna, SHAP, MLflow |
| RAG / LLM | ChromaDB, sentence-transformers, Ollama (fully local, no data leaves the machine) |
| Serving | FastAPI, Streamlit, Plotly |

Cost to run locally: **$0**. An illustrative managed-cloud equivalent is roughly $125/month.

---

## 📁 Project structure

```
requirements.txt            website dependencies        run_website.bat / .sh
requirements-pipeline.txt   full pipeline dependencies  mlflow.db, mlruns/   stored experiments
models/champion/            portable model + predict_example.py
data/sqlite/                bronze / silver / gold / audit databases (SQLite backend)
dbt/local_dev.duckdb        gold feature store          docs/   report, drift & SHAP CSVs, 7 ADRs
src/python/
  extractors/ loaders/ generators/   ingest + Faker new applications
  transformers/                      PySpark cleaning, features, encoding
  models/ explainability/ scoring/   training, SHAP, batch scoring, decision rules
  monitoring/ compliance/            PSI drift, PDPA erasure
  rag/ api/ dashboard/               regulation + FAME RAG, FastAPI, Streamlit
dbt/  dags/  docker/  sql/  tests/   models+snapshots, Airflow DAGs, infra, unit+integration tests
```

---

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/credit-decision/{applicant_id}` | Score, risk tier, decision, letter, top SHAP factors, regulation references, quality check |
| `POST` | `/credit-decision/{applicant_id}/chat` | Post-rejection follow-up question (FAME-grounded) |

```bash
curl -X POST "http://localhost:8000/credit-decision/<member_id_hash>?requested_by=risk_team"
```

Every request is hashed and written to `audit.decision_request_log`; LLM latency, tokens and SHAP-coverage score are logged to MLflow.

---

## 🔁 Re-running the whole pipeline (optional)

Only needed if you want to rebuild from raw data. Needs **Java 17+** (PySpark), the Kaggle CSV saved as `data/raw/cs-training.csv` (or Kaggle keys in `.env`), and internet for `dbt deps`.

```bash
pip install -r requirements-pipeline.txt
python -m src.python.setup_env                          # creates .env + encryption keys, checks Java/dataset/Ollama
python -m src.python.loaders.bronze_ingest
python -m src.python.transformers.data_cleaning
python -m src.python.transformers.feature_engineering
python -m src.python.transformers.encoding
cd dbt && dbt deps && dbt run && dbt test && dbt snapshot && cd ..
python -m src.python.models.train_xgboost               # weak laptop: --trials 10 --cv-folds 3
python -m src.python.models.model_registry --export     # refresh models/champion/
python -m src.python.explainability.shap_explainer      # weak laptop: --sample 20000
python -m src.python.scoring.scoring_pipeline
python -m src.python.monitoring.drift_monitor           # writes docs/psi_drift_report.csv
python -m src.python.compliance.pdpa_erasure
```

**Database backend:** SQLite (`data/sqlite/`) is used automatically unless both `POSTGRES_USER` and `POSTGRES_PASSWORD` are set in `.env`; then PostgreSQL via `docker compose up -d` (host port 5433). Force a choice with `DB_BACKEND=sqlite|postgres`. A configured-but-unreachable PostgreSQL fails loudly rather than silently switching.

**Tests:** `pytest tests/unit` (81 tests), `pytest tests/integration` (model-quality guards; RAG tests skip when Ollama / the embedding model are unavailable), `cd dbt && dbt test` (28 data tests).

---

## 📜 Architecture decision records

All in `docs/decisions/`: **001** real Kaggle data over synthetic · **002** PII pseudonymization in Bronze · **003** SHAP over LIME · **004** dbt snapshots for PDPA erasure · **005** XGBoost over a neural net (interpretability) · **006** deterministic synthetic categorical enrichment · **007** leak-free features and one train/score/explain feature contract.

---

## ⚠️ Known limitations

- **Synthetic columns are not model features.** `loan_purpose`, `employment_type`, `employment_years`, `income_consistency`, `income_stability_score`, `payment_velocity_3m`, name and contact do not exist in the public data; they are generated deterministically (ADR-006) for the scorecard/demo only. `models/features.py` is the model's feature contract (12 features).
- **In-sample scores.** Rows used for training carry `data_split = 'train'`; only `test` rows are out-of-sample.
- **Drift demo.** The monitor compares the training split with the latest generated applications (or the test split if none). The generator uses simple synthetic distributions, so it shows large PSI by design. A train-vs-test check gives max PSI 0.0008, confirming no false alarms.
- **RAG latency target not met:** about 120 s per letter on CPU with `llama3.2`. Smaller local models may also omit a SHAP factor, which is why a coverage score is logged per request.
- **Regulation text is simulated** (MAS/OJK style, labelled as such) unless you place the official RBI PDF in `data/knowledge_base/pdfs/` and run `python -m src.python.rag.chroma_store` to rebuild the index.
- **2011 data**, not the Singapore/Indonesia market; no fairness/bias analysis (age is a model feature). A portfolio demonstration, **not** a validated production credit model.
- Erasure scope for retained Bronze rows and snapshot history needs DPO sign-off.
- `.env` holds generated keys that tie the stored hashes to the encrypted vault. The data is synthetic, but do not publish this folder.

**Toward production:** managed warehouse and object store, access-controlled hosted LLM with PII redaction, real intake pipeline and load testing, human-approved automated retraining, versioned legal text with legal sign-off, and role-based access control on the API and dashboard.

---

## 🪟 Environment notes

- PySpark needs a JDK; `spark_session.py` auto-detects JDK 11/17/21 on Windows and converts `JAVA_HOME` to a short path. Writing files from Spark on Windows needs `winutils.exe` + `hadoop.dll` in `%HADOOP_HOME%\bin`.
- PowerShell 5.1 has no `&&`; use separate lines, PowerShell 7 or Git Bash. Activate a venv with `.venv\Scripts\Activate.ps1`.
- The default LLM is `llama3.2` (2 GB); change the model name in `.env` for a larger one.
- Pages insert the project root into `sys.path` themselves, so run Streamlit from the project root.
- If you move this folder, nothing breaks: MLflow's stored absolute paths are rewritten automatically the first time a dashboard page loads.

---

## 📄 License

MIT (add a `LICENSE` file before publishing).