"""PDPA erasure processing DAG - runs hourly, processes PENDING requests & checks SLA breaches."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "rio", "retries": 1, "retry_delay": timedelta(minutes=5)}


def _task_process_erasures(**context):
    from src.python.compliance.pdpa_erasure import check_sla_breaches, process_pending_erasures

    process_pending_erasures()
    check_sla_breaches()


with DAG(
    dag_id="erasure_request_dag",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit-risk", "pdpa", "compliance"],
) as dag:
    PythonOperator(task_id="process_pending_erasures", python_callable=_task_process_erasures)