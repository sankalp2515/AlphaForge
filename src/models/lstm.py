"""
AlphaForge — LSTM Baseline Model
Serves as the baseline to quantify TFT's marginal value.
Simple but properly implemented: LayerNorm, dropout, residual connections.
"""
from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.logger import get_logger

logger = get_logger(__name__)


class LSTMSignalModel(L.LightningModule):
    """
    LSTM model for binary price direction prediction.
    Input: (batch, seq_len, n_features)
    Output: (batch, 1) probability of upward move
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Input normalization
        self.input_norm = nn.LayerNorm(n_features)

        # LSTM stack
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Attention over time steps
        self.attention = nn.Linear(hidden_size, 1)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("built_lstm_model", n_params=n_params, hidden_size=hidden_size,
                    num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, 1)
        """
        # Normalize input features
        x = self.input_norm(x)

        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention pooling over time steps
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)              # (batch, hidden_size)

        # Classification
        logits = self.head(context)  # (batch, 1)
        return logits

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, y = batch
        logits = self(x).squeeze(-1)
        loss = self.loss_fn(logits, y.float())

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y.float()).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def predict_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, _ = batch
        logits = self(x).squeeze(-1)
        return torch.sigmoid(logits)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ─── Dataset helper ───────────────────────────────────────────────────────────

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Sliding window sequence dataset for LSTM.
    Each sample: (seq_len, n_features) → label
    """

    def __init__(
        self,
        features: Any,           # numpy array (n_samples, n_features)
        labels: Any,             # numpy array (n_samples,)
        seq_len: int = 60,
    ) -> None:
        import numpy as np
        self.X = np.array(features, dtype=np.float32)
        self.y = np.array(labels, dtype=np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx: idx + self.seq_len])
        y = torch.tensor(self.y[idx + self.seq_len - 1])
        return x, y
