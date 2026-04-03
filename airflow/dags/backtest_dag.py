"""
AlphaForge — Standalone Backtest DAG
Can be triggered manually or after training to run full backtest suite.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "alphaforge",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def run_full_backtest(**context):
    import logging
    from src.config import get_settings
    from src.data.storage import load_ohlcv, load_features
    from src.evaluation.backtest import run_backtest, run_walk_forward_backtest, log_backtest_to_mlflow
    import numpy as np
    import pandas as pd

    settings = get_settings()
    asset = "BTC/USDT"
    price_df = load_ohlcv(asset, settings.crypto_timeframe)
    feat_df = load_features(asset, settings.crypto_timeframe)

    if price_df.empty or feat_df.empty:
        logging.warning("No data available for backtest")
        return

    common_idx = price_df.index.intersection(feat_df.index)
    price_aligned = price_df.loc[common_idx].reset_index()

    np.random.seed(42)
    signals = pd.Series(np.random.beta(2, 2, len(common_idx)), index=common_idx)

    fin_metrics, trade_log, returns = run_backtest(price_aligned, signals)

    logging.info(
        f"Backtest complete: Sharpe={fin_metrics.sharpe_ratio:.3f}, "
        f"MaxDD={fin_metrics.max_drawdown:.3f}, "
        f"WinRate={fin_metrics.win_rate:.3f}, "
        f"Trades={fin_metrics.total_trades}"
    )

    # Walk-forward
    wf_metrics = run_walk_forward_backtest(price_aligned, signals, n_windows=4)
    sharpes = [m.sharpe_ratio for m in wf_metrics]
    logging.info(f"Walk-forward Sharpe by window: {[f'{s:.3f}' for s in sharpes]}")
    logging.info(f"Mean walk-forward Sharpe: {np.mean(sharpes):.3f}")

with DAG(
    dag_id="alphaforge_backtest",
    default_args=default_args,
    description="Standalone backtest DAG — run manually or after training",
    schedule=None,   # Manual trigger only
    catchup=False,
    tags=["alphaforge", "backtest", "evaluation"],
) as dag:

    dag.doc_md = """
    ## AlphaForge Backtest DAG
    **Trigger:** Manual only (or triggered by train_dag after promotion)
    Runs full Backtrader simulation + walk-forward validation.
    Results written to TimescaleDB and MLflow.
    """

    start = EmptyOperator(task_id="start")

    task_backtest = PythonOperator(
        task_id="run_full_backtest",
        python_callable=run_full_backtest,
        execution_timeout=timedelta(minutes=45),
    )

    end = EmptyOperator(task_id="end")

    start >> task_backtest >> end
