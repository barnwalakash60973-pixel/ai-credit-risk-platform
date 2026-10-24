"""Weekly PSI drift monitoring DAG."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "rio", "retries": 1, "retry_delay": timedelta(minutes=10)}


def _task_run_drift_check(**context):
    """Delegates to drift_monitor.run_drift_report (single implementation, absolute output path)."""
    from src.python.monitoring.drift_monitor import run_drift_report

    run_drift_report()


with DAG(
    dag_id="drift_monitor_dag",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit-risk", "monitoring", "mlops"],
) as dag:
    PythonOperator(task_id="run_drift_check", python_callable=_task_run_drift_check)