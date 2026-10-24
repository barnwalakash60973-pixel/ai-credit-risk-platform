"""
Airflow DAG - Bronze Ingest Pipeline
Runs daily, generates new applications, then ingests into Bronze layer with PII pseudonymization.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "rio",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _task_generate_new_applications(**context):
    from src.python.generators.new_applications_generator import (
        generate_new_applications,
        save_new_applications,
    )

    df = generate_new_applications(n=1000)
    out_path = save_new_applications(df)
    context["ti"].xcom_push(key="new_applications_path", value=str(out_path))


def _task_bronze_ingest_initial(**context):
    """Ingest initial Kaggle dataset (cs-training.csv) - only needs to run once,
    but kept as a separate task so it can be manually triggered."""
    from src.python.loaders.bronze_ingest import run_bronze_ingest

    run_bronze_ingest(csv_filename="cs-training.csv")


with DAG(
    dag_id="ingest_bronze_dag",
    description="Daily ingestion of new credit applications into Bronze layer with PII pseudonymization",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit-risk", "bronze", "pii"],
) as dag:

    generate_applications_task = PythonOperator(
        task_id="generate_new_applications",
        python_callable=_task_generate_new_applications,
    )

    bronze_ingest_initial_task = PythonOperator(
        task_id="bronze_ingest_initial_dataset",
        python_callable=_task_bronze_ingest_initial,
    )

    generate_applications_task >> bronze_ingest_initial_task