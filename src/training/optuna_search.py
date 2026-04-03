"""
AlphaForge — Hyperparameter Optimization with Optuna
Uses TPE sampler + Hyperband pruner for efficient search.
Logs all trials to MLflow.
"""
from __future__ import annotations

import mlflow
import numpy as np
import optuna
from optuna.integration import MLflowCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.config import get_settings
from src.data.storage import load_multi_asset_features
from src.features.technical import FEATURE_COLUMNS
from src.logger import get_logger
from src.training.cv import PurgedKFold

logger = get_logger(__name__)
settings = get_settings()


def objective_lstm(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective function for LSTM hyperparameter search."""
    import lightning as L
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from src.models.lstm import LSTMSignalModel, SequenceDataset

    # ── Search Space ──────────────────────────────────────────────────────────
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    seq_len = trial.suggest_categorical("seq_len", [30, 60, 120])

    cv = PurgedKFold(n_splits=3, embargo_pct=0.01, horizon=20)
    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        train_ds = SequenceDataset(X[train_idx], y[train_idx], seq_len=seq_len)
        val_ds = SequenceDataset(X[val_idx], y[val_idx], seq_len=seq_len)

        train_dl = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        model = LSTMSignalModel(
            n_features=X.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
        )

        trainer = L.Trainer(
            max_epochs=20,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=1.0,
            callbacks=[
                L.pytorch.callbacks.EarlyStopping(
                    monitor="val_loss", patience=3, mode="min"
                )
            ],
        )

        trainer.fit(model, train_dl, val_dl)

        # Evaluate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                probs = torch.sigmoid(model(x_batch).squeeze(-1)).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(y_batch.cpu().numpy().tolist())

        if len(set(all_labels)) < 2:
            return 0.5  # degenerate fold

        auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
        auc_scores.append(auc)

        # Pruning: report intermediate value to Hyperband
        trial.report(float(np.mean(auc_scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(auc_scores))


def run_optuna_search(
    model_type: str = "lstm",
    n_trials: int = 50,
    experiment_name: str | None = None,
) -> optuna.Study:
    """
    Run hyperparameter search and log all trials to MLflow.
    Returns the completed Optuna study.
    """
    experiment_name = experiment_name or f"{settings.mlflow_experiment_name}-hparam"
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("loading_data_for_hparam_search")
    df = load_multi_asset_features(
        assets=settings.all_assets,
        timeframe=settings.crypto_timeframe,
    )
    df = df.dropna(subset=["label_1"])

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = np.nan_to_num(
        df[available_features].values.astype(np.float32),
        nan=0.0, posinf=1.0, neginf=-1.0,
    )
    X = np.clip(X, -10, 10)
    y = df["label_1"].values.astype(np.float32)

    # MLflow callback logs each trial as a run
    mlflow_cb = MLflowCallback(
        tracking_uri=settings.mlflow_tracking_uri,
        metric_name="cv_auc",
        mlflow_kwargs={"experiment_id": mlflow.get_experiment_by_name(experiment_name).experiment_id},
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3),
        study_name=f"alphaforge-{model_type}-search",
    )

    if model_type == "lstm":
        objective = lambda trial: objective_lstm(trial, X, y)
    else:
        raise NotImplementedError(f"Optuna search not implemented for {model_type}")

    logger.info("starting_optuna_search", n_trials=n_trials, model=model_type)

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[mlflow_cb],
        show_progress_bar=True,
    )

    logger.info(
        "optuna_search_complete",
        best_value=study.best_value,
        best_params=study.best_params,
    )

    print(f"\n{'='*60}")
    print(f"Best CV AUC-ROC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"{'='*60}")

    return study


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--model", default="lstm", type=click.Choice(["lstm", "tft"]))
    @click.option("--n-trials", default=50, type=int)
    def main(model: str, n_trials: int) -> None:
        from src.logger import setup_logging
        setup_logging()
        run_optuna_search(model_type=model, n_trials=n_trials)

    main()
