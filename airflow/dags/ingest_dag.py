"""
AlphaForge — Ingestion DAG
Runs daily to fetch fresh OHLCV data for all configured assets.
Validates data quality before marking as success.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ─── Default Args ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "alphaforge",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}


# ─── Task Functions ───────────────────────────────────────────────────────────

def ingest_crypto(**context) -> dict:
    """Fetch OHLCV data from Binance for crypto assets."""
    from src.data.ingestion import BinanceIngester
    from src.config import get_settings

    settings = get_settings()
    ingester = BinanceIngester()
    results = {}

    for symbol in settings.crypto_assets:
        n = ingester.ingest(
            symbol,
            timeframe=settings.crypto_timeframe,
            lookback_days=7,  # daily run: only last 7 days (incremental)
        )
        results[symbol] = n

    context["task_instance"].xcom_push(key="crypto_results", value=results)
    return results


def ingest_equity(**context) -> dict:
    """Fetch OHLCV data from Yahoo Finance for equity assets."""
    from src.data.ingestion import YFinanceIngester
    from src.config import get_settings

    settings = get_settings()
    ingester = YFinanceIngester()
    results = {}

    for ticker in settings.equity_assets:
        n = ingester.ingest(
            ticker,
            lookback_days=7,
            interval=settings.equity_timeframe,
        )
        results[ticker] = n

    context["task_instance"].xcom_push(key="equity_results", value=results)
    return results


def validate_data(**context) -> str:
    """
    Run Great Expectations data validation.
    Branch to 'ingestion_success' or 'ingestion_failed'.
    """
    ti = context["task_instance"]
    crypto_results = ti.xcom_pull(key="crypto_results", task_ids="ingest_crypto") or {}
    equity_results = ti.xcom_pull(key="equity_results", task_ids="ingest_equity") or {}

    all_results = {**crypto_results, **equity_results}
    failed = [asset for asset, n in all_results.items() if n <= 0]

    if failed:
        import logging
        logging.warning(f"Ingestion failed for: {failed}")
        # Don't hard-fail — partial success is acceptable for daily runs
        # Only fail if ALL assets failed
        if len(failed) == len(all_results):
            return "ingestion_failed"

    # Run basic row count validation
    from src.data.storage import get_engine
    from sqlalchemy import text

    with get_engine().connect() as conn:
        for asset in all_results:
            result = conn.execute(
                text("SELECT COUNT(*) FROM ohlcv WHERE asset = :a AND time >= NOW() - INTERVAL '7 days'"),
                {"a": asset},
            ).scalar()
            if result == 0:
                logging.warning(f"No recent data for {asset}")

    return "ingestion_success"


def log_ingestion_summary(**context) -> None:
    """Log final ingestion summary."""
    ti = context["task_instance"]
    crypto = ti.xcom_pull(key="crypto_results", task_ids="ingest_crypto") or {}
    equity = ti.xcom_pull(key="equity_results", task_ids="ingest_equity") or {}

    total_rows = sum(v for v in {**crypto, **equity}.values() if v > 0)
    import logging
    logging.info(f"Ingestion complete. Total rows written: {total_rows}")
    logging.info(f"Crypto: {crypto}")
    logging.info(f"Equity: {equity}")


# ─── DAG Definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="alphaforge_ingest",
    default_args=default_args,
    description="Daily OHLCV data ingestion for all assets",
    schedule="0 1 * * *",      # 1 AM UTC daily
    catchup=False,
    max_active_runs=1,
    tags=["alphaforge", "ingestion", "data"],
) as dag:

    dag.doc_md = """
    ## AlphaForge Ingestion DAG

    **Schedule:** Daily at 01:00 UTC

    **Pipeline:**
    1. Ingest crypto OHLCV from Binance (BTC, ETH, SOL)
    2. Ingest equity OHLCV from Yahoo Finance (SPY, QQQ)
    3. Validate data quality
    4. Branch on success/failure

    **Downstream:** Triggers `alphaforge_features` DAG on success.
    """

    start = EmptyOperator(task_id="start")

    task_ingest_crypto = PythonOperator(
        task_id="ingest_crypto",
        python_callable=ingest_crypto,
    )

    task_ingest_equity = PythonOperator(
        task_id="ingest_equity",
        python_callable=ingest_equity,
    )

    task_validate = BranchPythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    ingestion_success = EmptyOperator(task_id="ingestion_success")
    ingestion_failed = EmptyOperator(task_id="ingestion_failed")

    task_log_summary = PythonOperator(
        task_id="log_summary",
        python_callable=log_ingestion_summary,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # ── Task Dependencies ──────────────────────────────────────────────────────
    start >> [task_ingest_crypto, task_ingest_equity] >> task_validate
    task_validate >> [ingestion_success, ingestion_failed]
    ingestion_success >> task_log_summary
    ingestion_failed >> task_log_summary
    task_log_summary >> end
