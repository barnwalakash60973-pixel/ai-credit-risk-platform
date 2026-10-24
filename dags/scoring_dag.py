"""Daily scoring DAG - score new applications using the production model."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "rio", "retries": 2, "retry_delay": timedelta(minutes=5)}


def _task_score_applications(**context):
    from datetime import date

    from src.python.config import RAW_DIR
    from src.python.scoring.scoring_pipeline import save_decisions, score_new_applications

    csv_path = RAW_DIR / "new_applications" / f"new_applications_{date.today().isoformat()}.csv"
    # Feature list comes from the champion model itself (stored with it at training time).
    decisions = score_new_applications(csv_path)
    save_decisions(decisions)


with DAG(
    dag_id="scoring_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit-risk", "scoring"],
) as dag:
    PythonOperator(task_id="score_applications", python_callable=_task_score_applications)