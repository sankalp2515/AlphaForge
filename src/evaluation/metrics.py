"""
AlphaForge — Evaluation Suite
Computes ML metrics AND financial metrics.
Tower Research cares about Sharpe > Accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from scipy.stats import spearmanr

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─── Metric Containers ────────────────────────────────────────────────────────

@dataclass
class MLMetrics:
    auc_roc: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    ece: float = 0.0               # Expected Calibration Error
    information_coefficient: float = 0.0  # Spearman corr: pred rank vs actual return

    def passes_gate(self) -> bool:
        return (
            self.auc_roc >= settings.min_auc_roc
            and self.information_coefficient >= settings.min_information_coefficient
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "ml/auc_roc": self.auc_roc,
            "ml/f1": self.f1_score,
            "ml/accuracy": self.accuracy,
            "ml/log_loss": self.log_loss,
            "ml/brier_score": self.brier_score,
            "ml/ece": self.ece,
            "ml/information_coefficient": self.information_coefficient,
        }


@dataclass
class FinancialMetrics:
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    turnover: float = 0.0
    avg_return_per_trade: float = 0.0

    def passes_gate(self) -> bool:
        return (
            self.sharpe_ratio >= settings.min_sharpe_ratio
            and self.max_drawdown <= settings.max_drawdown_pct
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "backtest/total_return": self.total_return,
            "backtest/sharpe_ratio": self.sharpe_ratio,
            "backtest/sortino_ratio": self.sortino_ratio,
            "backtest/max_drawdown": self.max_drawdown,
            "backtest/calmar_ratio": self.calmar_ratio,
            "backtest/win_rate": self.win_rate,
            "backtest/total_trades": float(self.total_trades),
            "backtest/turnover": self.turnover,
        }


# ─── ML Metrics Computation ───────────────────────────────────────────────────

def compute_ml_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
    fwd_returns: np.ndarray | None = None,
    n_bins: int = 10,
) -> MLMetrics:
    """
    Compute comprehensive ML evaluation metrics.

    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities (0-1)
        y_pred: Binary predictions (optional, derived from y_prob if None)
        fwd_returns: Actual forward returns (for Information Coefficient)
        n_bins: Number of bins for ECE computation
    """
    if y_pred is None:
        y_pred = (y_prob > 0.5).astype(int)

    metrics = MLMetrics()
    metrics.auc_roc = roc_auc_score(y_true, y_prob)
    metrics.f1_score = f1_score(y_true, y_pred, zero_division=0)
    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.log_loss = log_loss(y_true, y_prob)
    metrics.brier_score = brier_score_loss(y_true, y_prob)

    # Expected Calibration Error (how well-calibrated the probabilities are)
    metrics.ece = _compute_ece(y_true, y_prob, n_bins=n_bins)

    # Information Coefficient: Spearman rank correlation of predictions vs actual returns
    # IC > 0.03 is considered a good signal in quantitative finance
    if fwd_returns is not None and len(fwd_returns) > 10:
        ic, _ = spearmanr(y_prob, fwd_returns)
        metrics.information_coefficient = float(ic) if not np.isnan(ic) else 0.0

    return metrics


def _compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.
    Measures how well the predicted probabilities match observed frequencies.
    ECE = 0 means perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        bin_size = mask.sum() / len(y_true)

        ece += bin_size * abs(bin_conf - bin_acc)

    return float(ece)


# ─── Financial Metrics Computation ────────────────────────────────────────────

def compute_financial_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.05,  # 5% annual risk-free rate
    periods_per_year: int = 252,
) -> FinancialMetrics:
    """
    Compute financial performance metrics from a return series.

    Args:
        returns: Daily/bar-level strategy returns
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        periods_per_year: 252 for daily equity, 8760 for hourly crypto
    """
    metrics = FinancialMetrics()

    if len(returns) == 0:
        return metrics

    returns = returns.dropna()
    if len(returns) == 0:
        return metrics

    # Total return
    metrics.total_return = float((1 + returns).prod() - 1)

    # Sharpe Ratio
    excess = returns - risk_free_rate / periods_per_year
    metrics.sharpe_ratio = float(
        excess.mean() / excess.std() * np.sqrt(periods_per_year)
        if excess.std() > 0 else 0.0
    )

    # Sortino Ratio (only downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-8
    metrics.sortino_ratio = float(
        (returns.mean() - risk_free_rate / periods_per_year)
        / downside_std * np.sqrt(periods_per_year)
    )

    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    metrics.max_drawdown = float(drawdown.min())

    # Calmar Ratio
    if abs(metrics.max_drawdown) > 0:
        annual_return = (1 + metrics.total_return) ** (periods_per_year / len(returns)) - 1
        metrics.calmar_ratio = float(annual_return / abs(metrics.max_drawdown))

    # Win rate
    trades = returns[returns != 0]
    if len(trades) > 0:
        metrics.win_rate = float((trades > 0).mean())
        metrics.total_trades = int(len(trades))
        metrics.avg_return_per_trade = float(trades.mean())

    return metrics


# ─── SHAP Explainability ──────────────────────────────────────────────────────

def compute_shap_values(
    model: torch.nn.Module,
    X: np.ndarray,
    feature_names: list[str],
    n_samples: int = 200,
    seq_len: int = 60,
) -> dict[str, float]:
    """
    Compute SHAP feature importance for the model.
    Uses DeepExplainer for PyTorch models.
    Returns mean absolute SHAP value per feature.
    """
    model.eval()
    X_sample = X[:n_samples]

    # Create background (reference) dataset
    background_size = min(50, len(X_sample))
    background = torch.FloatTensor(X_sample[:background_size, :seq_len, :]) \
        if X_sample.ndim == 3 else torch.FloatTensor(X_sample[:background_size])

    try:
        explainer = shap.DeepExplainer(model, background)
        test_data = torch.FloatTensor(X_sample[:100]) if len(X_sample) > 100 else torch.FloatTensor(X_sample)
        shap_values = explainer.shap_values(test_data)

        # Mean absolute SHAP value per feature (across time and samples)
        if isinstance(shap_values, list):
            shap_arr = np.abs(shap_values[0])
        else:
            shap_arr = np.abs(shap_values)

        # Average over samples and time steps
        if shap_arr.ndim == 3:
            mean_shap = shap_arr.mean(axis=(0, 1))
        else:
            mean_shap = shap_arr.mean(axis=0)

        importance = {
            feat: float(val)
            for feat, val in zip(feature_names, mean_shap)
        }

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        logger.info("shap_complete", top_features=list(importance.keys())[:5])
        return importance

    except Exception as e:
        logger.warning("shap_failed", error=str(e))
        return {}


# ─── Evaluation Report ────────────────────────────────────────────────────────

def log_evaluation_to_mlflow(
    run_id: str,
    ml_metrics: MLMetrics,
    financial_metrics: FinancialMetrics,
    shap_importance: dict[str, float] | None = None,
) -> None:
    """Log full evaluation results to an existing MLflow run."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(ml_metrics.to_dict())
        mlflow.log_metrics(financial_metrics.to_dict())

        if shap_importance:
            # Log top 10 feature importances
            for feat, val in list(shap_importance.items())[:10]:
                mlflow.log_metric(f"shap/{feat}", val)

        # Log whether model passes promotion gate
        ml_gate = ml_metrics.passes_gate()
        fin_gate = financial_metrics.passes_gate()
        mlflow.log_params({
            "passes_ml_gate": str(ml_gate),
            "passes_financial_gate": str(fin_gate),
            "ready_for_production": str(ml_gate and fin_gate),
        })

    logger.info(
        "evaluation_logged",
        run_id=run_id,
        auc_roc=ml_metrics.auc_roc,
        sharpe=financial_metrics.sharpe_ratio,
        passes_ml_gate=ml_gate,
        passes_fin_gate=fin_gate,
    )
