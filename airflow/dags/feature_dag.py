"""
AlphaForge — Feature Engineering DAG
Computes alpha features for all assets after ingestion.
Scheduled after ingest_dag completes.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "alphaforge",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


def compute_features(**context) -> dict:
    from src.features.pipeline import run_feature_pipeline
    results = run_feature_pipeline()
    context["task_instance"].xcom_push(key="feature_results", value=results)
    return results


def validate_features(**context) -> None:
    """Check that features were written successfully."""
    import logging
    from src.data.storage import get_engine
    from sqlalchemy import text

    with get_engine().connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM features WHERE time >= NOW() - INTERVAL '2 days'")
        ).scalar()

    if count == 0:
        raise ValueError("No features written in last 2 days — feature pipeline failed")

    logging.info(f"Feature validation passed. Recent feature rows: {count}")


def check_drift(**context) -> None:
    """Run drift detection after each feature update."""
    from src.evaluation.drift import run_drift_detection
    summary = run_drift_detection()
    import logging
    if summary.get("any_drift"):
        logging.warning(f"DRIFT DETECTED: {summary}")
    else:
        logging.info("No drift detected")


with DAG(
    dag_id="alphaforge_features",
    default_args=default_args,
    description="Feature engineering pipeline — runs after ingestion",
    schedule="30 1 * * *",     # 1:30 AM UTC — 30 min after ingestion
    catchup=False,
    max_active_runs=1,
    tags=["alphaforge", "features"],
) as dag:

    dag.doc_md = """
    ## AlphaForge Feature Engineering DAG

    **Schedule:** Daily at 01:30 UTC (after ingestion)

    **Pipeline:**
    1. Wait for ingestion DAG to complete (sensor)
    2. Compute technical indicators (RSI, MACD, BB, ATR, OBV, ADX)
    3. Compute momentum features (multi-horizon, cross-sectional rank)
    4. Compute volatility + microstructure features
    5. Compute forward return labels
    6. Validate feature quality
    7. Run drift detection
    """

    # Wait for ingestion to finish
    wait_for_ingestion = ExternalTaskSensor(
        task_id="wait_for_ingestion",
        external_dag_id="alphaforge_ingest",
        external_task_id="end",
        timeout=3600,
        mode="reschedule",
        poke_interval=60,
    )

    task_compute = PythonOperator(
        task_id="compute_features",
        python_callable=compute_features,
    )

    task_validate = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
    )

    task_drift = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    end = EmptyOperator(task_id="end")

    wait_for_ingestion >> task_compute >> task_validate >> task_drift >> end
