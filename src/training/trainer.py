"""
AlphaForge — Model Trainer
Full training pipeline with MLflow tracking, Purged K-Fold CV,
early stopping, and model registry promotion.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config import get_settings
from src.data.storage import load_multi_asset_features
from src.features.technical import FEATURE_COLUMNS
from src.logger import get_logger
from src.models.lstm import LSTMSignalModel, SequenceDataset
from src.training.cv import PurgedKFold, WalkForwardValidator

logger = get_logger(__name__)
settings = get_settings()


# ─── Training Entry Points ────────────────────────────────────────────────────

def train_lstm(
    df: pd.DataFrame,
    target: str = "label_1",
    experiment_name: str | None = None,
    run_name: str | None = None,
    hparams: dict[str, Any] | None = None,
) -> str:
    """
    Train LSTM baseline model.
    Returns MLflow run_id.
    """
    experiment_name = experiment_name or settings.mlflow_experiment_name
    hparams = hparams or {}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # ── Prepare data ──────────────────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    # Fill any remaining NaN
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    # Clip to prevent gradient explosions
    X = np.clip(X, -10, 10)

    seq_len = settings.feature_window
    n_features = X.shape[1]

    # ── Purged K-Fold CV ──────────────────────────────────────────────────────
    cv = PurgedKFold(n_splits=5, embargo_pct=0.01, horizon=20)
    fold_metrics: list[dict] = []

    with mlflow.start_run(run_name=run_name or f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id

        # Log hyperparameters
        default_hparams = {
            "model": "lstm",
            "hidden_size": hparams.get("hidden_size", 128),
            "num_layers": hparams.get("num_layers", 2),
            "dropout": hparams.get("dropout", 0.2),
            "learning_rate": hparams.get("learning_rate", settings.learning_rate),
            "seq_len": seq_len,
            "n_features": n_features,
            "target": target,
            "cv_strategy": "purged_kfold",
            "n_folds": 5,
        }
        mlflow.log_params(default_hparams)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
            logger.info("training_fold", fold=fold_idx, model="lstm",
                        train_size=len(train_idx), val_size=len(val_idx))

            # Build datasets
            train_ds = SequenceDataset(X[train_idx], y[train_idx], seq_len=seq_len)
            val_ds = SequenceDataset(X[val_idx], y[val_idx], seq_len=seq_len)

            train_dl = DataLoader(train_ds, batch_size=settings.batch_size,
                                  shuffle=False, num_workers=0, drop_last=True)
            val_dl = DataLoader(val_ds, batch_size=settings.batch_size,
                                shuffle=False, num_workers=0)

            # Build model
            model = LSTMSignalModel(
                n_features=n_features,
                hidden_size=default_hparams["hidden_size"],
                num_layers=default_hparams["num_layers"],
                dropout=default_hparams["dropout"],
                learning_rate=default_hparams["learning_rate"],
            )

            # Callbacks
            callbacks = [
                EarlyStopping(monitor="val_loss", patience=settings.early_stopping_patience,
                              mode="min"),
                LearningRateMonitor(logging_interval="epoch"),
            ]

            # Trainer
            trainer = L.Trainer(
                max_epochs=settings.max_epochs,
                callbacks=callbacks,
                enable_progress_bar=True,
                enable_model_summary=False,
                log_every_n_steps=10,
                gradient_clip_val=1.0,
                accelerator="auto",
            )

            trainer.fit(model, train_dl, val_dl)

            # ── Evaluate fold ─────────────────────────────────────────────────
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for x_batch, y_batch in val_dl:
                    logits = model(x_batch).squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(y_batch.cpu().numpy().tolist())

            probs_arr = np.array(all_probs)
            labels_arr = np.array(all_labels)
            preds_arr = (probs_arr > 0.5).astype(int)

            fold_m = {
                "fold": fold_idx,
                "auc_roc": roc_auc_score(labels_arr, probs_arr),
                "f1": f1_score(labels_arr, preds_arr, zero_division=0),
                "accuracy": accuracy_score(labels_arr, preds_arr),
            }
            fold_metrics.append(fold_m)

            logger.info("fold_complete", **fold_m)
            mlflow.log_metrics({
                f"fold_{fold_idx}_auc": fold_m["auc_roc"],
                f"fold_{fold_idx}_f1": fold_m["f1"],
            })

        # ── Aggregate CV metrics ──────────────────────────────────────────────
        auc_scores = [m["auc_roc"] for m in fold_metrics]
        f1_scores = [m["f1"] for m in fold_metrics]
        acc_scores = [m["accuracy"] for m in fold_metrics]

        mlflow.log_metrics({
            "cv_auc_mean": float(np.mean(auc_scores)),
            "cv_auc_std": float(np.std(auc_scores)),
            "cv_f1_mean": float(np.mean(f1_scores)),
            "cv_f1_std": float(np.std(f1_scores)),
            "cv_acc_mean": float(np.mean(acc_scores)),
        })

        logger.info(
            "training_complete",
            run_id=run_id,
            cv_auc_mean=float(np.mean(auc_scores)),
            cv_f1_mean=float(np.mean(f1_scores)),
        )

        # ── Train final model on all data & log to MLflow ─────────────────────
        final_ds = SequenceDataset(X, y, seq_len=seq_len)
        final_dl = DataLoader(final_ds, batch_size=settings.batch_size, shuffle=False)

        final_model = LSTMSignalModel(
            n_features=n_features,
            hidden_size=default_hparams["hidden_size"],
            num_layers=default_hparams["num_layers"],
            dropout=default_hparams["dropout"],
            learning_rate=default_hparams["learning_rate"],
        )

        final_trainer = L.Trainer(
            max_epochs=min(30, settings.max_epochs),
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=1.0,
        )
        final_trainer.fit(final_model, final_dl)

        mlflow.pytorch.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=f"{settings.mlflow_model_name}-lstm",
        )

        # ── Promotion gate ────────────────────────────────────────────────────
        mean_auc = float(np.mean(auc_scores))
        _maybe_promote_model(
            run_id=run_id,
            model_name=f"{settings.mlflow_model_name}-lstm",
            metrics={"auc_roc": mean_auc},
        )

    return run_id


def _maybe_promote_model(
    run_id: str,
    model_name: str,
    metrics: dict[str, float],
) -> None:
    """
    Promote model to 'Staging' if it meets minimum thresholds.
    Full promotion to 'Production' requires backtest metrics too (see evaluate.py).
    """
    client = mlflow.tracking.MlflowClient()

    passes_gate = metrics.get("auc_roc", 0) >= settings.min_auc_roc

    if passes_gate:
        try:
            # Get latest version
            versions = client.get_latest_versions(model_name, stages=["None"])
            if versions:
                version = versions[-1].version
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Staging",
                )
                logger.info("model_promoted_to_staging", model=model_name,
                            version=version, metrics=metrics)
        except Exception as e:
            logger.warning("promotion_failed", error=str(e))
    else:
        logger.info("model_failed_promotion_gate",
                    metrics=metrics, thresholds={"auc_roc": settings.min_auc_roc})


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--model", type=click.Choice(["lstm", "tft", "xgboost"]), default="lstm")
    @click.option("--experiment", default=None)
    @click.option("--target", default="label_1")
    def main(model: str, experiment: str | None, target: str) -> None:
        from src.logger import setup_logging
        setup_logging()

        logger.info("loading_features_for_training")
        df = load_multi_asset_features(
            assets=settings.all_assets,
            timeframe=settings.crypto_timeframe,
        )
        df = df.dropna(subset=[target])

        if model == "lstm":
            run_id = train_lstm(df, target=target, experiment_name=experiment)
            print(f"✅ Training complete. MLflow run_id: {run_id}")
        else:
            print(f"❌ Model '{model}' trainer not yet wired to CLI. Use airflow DAG.")

    main()
