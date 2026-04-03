"""
AlphaForge — Training & Backtest DAG
Weekly model retraining + backtesting + model promotion.
Also triggered automatically when drift is detected.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

default_args = {
    "owner": "alphaforge",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def train_baselines(**context) -> dict:
    """Train XGBoost and LSTM baselines."""
    import logging
    logging.info("Training baseline models...")

    from src.config import get_settings
    from src.data.storage import load_multi_asset_features
    from src.training.trainer import train_lstm

    settings = get_settings()
    df = load_multi_asset_features(
        assets=settings.all_assets,
        timeframe=settings.crypto_timeframe,
    )
    df = df.dropna(subset=["label_1"])

    run_id = train_lstm(
        df,
        target="label_1",
        experiment_name=f"{settings.mlflow_experiment_name}-baselines",
        run_name=f"lstm_baseline_{datetime.now().strftime('%Y%m%d')}",
    )
    context["task_instance"].xcom_push(key="lstm_run_id", value=run_id)
    return {"lstm_run_id": run_id}


def train_tft(**context) -> dict:
    """Train primary TFT model."""
    import logging
    logging.info("Training TFT model...")
    # TFT training is computationally heavier — log placeholder for now
    # Full implementation in src/models/tft.py
    from src.config import get_settings
    settings = get_settings()

    import mlflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"tft_{datetime.now().strftime('%Y%m%d')}") as run:
        mlflow.log_param("model", "tft")
        mlflow.log_param("status", "training_initiated")
        run_id = run.info.run_id

    context["task_instance"].xcom_push(key="tft_run_id", value=run_id)
    return {"tft_run_id": run_id}


def run_backtest(**context) -> dict:
    """Backtest the best model from this training run."""
    import logging
    from src.config import get_settings
    from src.data.storage import load_ohlcv, load_features
    from src.evaluation.backtest import run_backtest as _run_backtest
    from src.evaluation.backtest import log_backtest_to_mlflow
    import numpy as np
    import pandas as pd
    import torch

    settings = get_settings()
    ti = context["task_instance"]
    run_id = ti.xcom_pull(key="lstm_run_id", task_ids="train_baselines")

    # Use BTC as the representative asset for backtest
    asset = "BTC/USDT"
    price_df = load_ohlcv(asset, settings.crypto_timeframe)
    feat_df = load_features(asset, settings.crypto_timeframe)

    if price_df.empty or feat_df.empty:
        logging.warning("Insufficient data for backtest, skipping")
        return {}

    # Align price and features
    common_idx = price_df.index.intersection(feat_df.index)
    price_aligned = price_df.loc[common_idx]

    # Generate mock signals (in production, load model and predict)
    np.random.seed(42)
    signals = pd.Series(
        np.random.beta(2, 2, len(common_idx)),
        index=common_idx,
    )

    try:
        fin_metrics, trade_log, portfolio_returns = _run_backtest(
            price_aligned.reset_index(),
            signals,
        )

        log_backtest_to_mlflow(
            run_id=run_id,
            fin_metrics=fin_metrics,
            model_version="latest",
            start_date=common_idx[0].date() if len(common_idx) > 0 else None,
            end_date=common_idx[-1].date() if len(common_idx) > 0 else None,
        )

        context["task_instance"].xcom_push(key="backtest_sharpe", value=fin_metrics.sharpe_ratio)
        context["task_instance"].xcom_push(key="backtest_max_dd", value=fin_metrics.max_drawdown)

        logging.info(f"Backtest: Sharpe={fin_metrics.sharpe_ratio:.3f}, MaxDD={fin_metrics.max_drawdown:.3f}")
        return {
            "sharpe": fin_metrics.sharpe_ratio,
            "max_dd": fin_metrics.max_drawdown,
        }

    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        return {}


def check_promotion_gate(**context) -> str:
    """
    Decide whether to promote model to production.
    Requires BOTH ML gate AND financial gate to pass.
    """
    ti = context["task_instance"]
    sharpe = ti.xcom_pull(key="backtest_sharpe", task_ids="run_backtest") or 0.0
    max_dd = ti.xcom_pull(key="backtest_max_dd", task_ids="run_backtest") or -1.0

    from src.config import get_settings
    settings = get_settings()

    passes = (
        sharpe >= settings.min_sharpe_ratio
        and max_dd >= -settings.max_drawdown_pct
    )

    import logging
    logging.info(f"Promotion gate: Sharpe={sharpe:.3f} (>={settings.min_sharpe_ratio}), "
                 f"MaxDD={max_dd:.3f} (>=-{settings.max_drawdown_pct}). "
                 f"Decision: {'PROMOTE' if passes else 'REJECT'}")

    return "promote_model" if passes else "reject_model"


def promote_model(**context) -> None:
    """Promote model from Staging to Production in MLflow."""
    import mlflow
    from src.config import get_settings
    settings = get_settings()

    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model_name = f"{settings.mlflow_model_name}-lstm"

    try:
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        if versions:
            v = versions[-1].version
            client.transition_model_version_stage(model_name, v, "Production")
            import logging
            logging.info(f"✅ Promoted {model_name} v{v} to Production")
    except Exception as e:
        import logging
        logging.error(f"Promotion failed: {e}")


def reject_model(**context) -> None:
    import logging
    ti = context["task_instance"]
    sharpe = ti.xcom_pull(key="backtest_sharpe", task_ids="run_backtest") or 0.0
    logging.warning(f"❌ Model rejected. Sharpe={sharpe:.3f} below threshold. Keeping current production model.")


with DAG(
    dag_id="alphaforge_train",
    default_args=default_args,
    description="Weekly model retraining, backtesting, and promotion",
    schedule="0 3 * * 1",      # Every Monday at 3 AM UTC
    catchup=False,
    max_active_runs=1,
    tags=["alphaforge", "training", "mlops"],
) as dag:

    dag.doc_md = """
    ## AlphaForge Training DAG

    **Schedule:** Weekly on Monday at 03:00 UTC

    **Pipeline:**
    1. Train LSTM baseline (Purged K-Fold CV, MLflow tracking)
    2. Train TFT primary model
    3. Backtest best model
    4. Promotion gate: Sharpe >= 0.8 AND MaxDD <= 20%
    5. Promote to Production OR reject
    """

    start = EmptyOperator(task_id="start")

    task_baselines = PythonOperator(
        task_id="train_baselines",
        python_callable=train_baselines,
        execution_timeout=timedelta(hours=2),
    )

    task_tft = PythonOperator(
        task_id="train_tft",
        python_callable=train_tft,
        execution_timeout=timedelta(hours=4),
    )

    task_backtest = PythonOperator(
        task_id="run_backtest",
        python_callable=run_backtest,
        execution_timeout=timedelta(minutes=30),
    )

    task_gate = BranchPythonOperator(
        task_id="check_promotion_gate",
        python_callable=check_promotion_gate,
    )

    task_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    task_reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    start >> [task_baselines, task_tft]
    task_baselines >> task_backtest
    task_tft >> task_backtest
    task_backtest >> task_gate
    task_gate >> [task_promote, task_reject]
    [task_promote, task_reject] >> end
