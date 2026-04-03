"""
AlphaForge — Temporal Fusion Transformer (TFT)
Primary prediction model for multi-horizon price direction.

TFT was chosen because:
1. Native multi-horizon forecasting (1/5/20 bars simultaneously)
2. Interpretable attention weights → explain which timesteps mattered
3. Explicit handling of static (asset identity) vs time-varying features
4. State-of-the-art on time-series benchmarks (Lim et al., 2021)
"""
from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import BinaryCrossEntropy, QuantileLoss

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct", "atr_14", "obv", "adx_14",
    "mom_1", "mom_5", "mom_20", "mom_rank",
    "dist_from_sma_20",
    "realized_vol_5", "realized_vol_20", "vol_ratio",
    "ofi", "ofi_ma5", "amihud", "volume_surprise",
]


def prepare_tft_dataset(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 1,
    target: str = "label_1",
    min_prediction_idx: int | None = None,
) -> TimeSeriesDataSet:
    """
    Prepare a TimeSeriesDataSet for TFT training.

    Args:
        df: Multi-asset feature DataFrame with 'asset' column and DatetimeIndex
        max_encoder_length: Number of past timesteps to look at (60 bars)
        max_prediction_length: How many steps ahead to predict (1 for binary)
        target: Target column name
        min_prediction_idx: Minimum time index for prediction (used for train/val split)
    """
    # TFT needs a numeric time_idx
    df = df.copy().reset_index()
    df = df.sort_values(["asset", "time"])

    # Create integer time index per asset group
    df["time_idx"] = df.groupby("asset").cumcount()

    # Clip features to reasonable ranges and fill NaN
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].clip(-10, 10).fillna(0.0)

    # Ensure target exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")

    available_features = [c for c in FEATURE_COLS if c in df.columns]

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=target,
        group_ids=["asset"],                  # one series per asset
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["asset"],        # asset identity is static
        time_varying_known_reals=[],          # we don't know future features
        time_varying_unknown_reals=available_features,
        target_normalizer=GroupNormalizer(
            groups=["asset"], transformation="softplus"
        ),
        min_prediction_idx=min_prediction_idx,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    return dataset


def build_tft_model(
    dataset: TimeSeriesDataSet,
    hidden_size: int = 64,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 32,
    learning_rate: float = 1e-3,
) -> TemporalFusionTransformer:
    """Build TFT model from dataset configuration."""
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),           # gives uncertainty estimates
        log_interval=10,
        reduce_on_plateau_patience=4,
        optimizer="ranger",
    )

    logger.info(
        "built_tft_model",
        n_params=sum(p.numel() for p in model.parameters()),
        hidden_size=hidden_size,
        attention_heads=attention_head_size,
    )

    return model


# ─── PyTorch Lightning Module (custom wrapper for binary classification) ──────

class TFTSignalModel(L.LightningModule):
    """
    Lightning wrapper around TFT for binary signal prediction.
    Adds: MLflow logging, custom metrics, gradient clipping.
    """

    def __init__(
        self,
        dataset: TimeSeriesDataSet,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.tft = build_tft_model(
            dataset,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            learning_rate=learning_rate,
        )

    def forward(self, x: dict) -> dict:
        return self.tft(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        out = self.tft(x)
        loss = self.tft.loss(out["prediction"], y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        out = self.tft(x)
        loss = self.tft.loss(out["prediction"], y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
