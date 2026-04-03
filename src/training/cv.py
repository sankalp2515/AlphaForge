"""
AlphaForge — Purged K-Fold Cross-Validation
Prevents lookahead bias in time-series ML.

Standard K-Fold is WRONG for financial data because:
  - Labels overlap in time (fwd_ret_5 at t and t+1 both use price at t+5)
  - Train/val split can leak future information

Purged K-Fold (López de Prado, "Advances in Financial Machine Learning"):
  - Removes training samples whose labels overlap with the validation window
  - Adds an embargo gap after validation to prevent leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.logger import get_logger

logger = get_logger(__name__)


class PurgedKFold:
    """
    Purged K-Fold cross-validator for financial time series.

    Implements the method from López de Prado (2018):
    1. Split into K folds based on time index
    2. For each fold, 'purge' training samples whose event window
       overlaps with the test period
    3. Apply an embargo on samples immediately after the test period

    Args:
        n_splits: Number of folds
        embargo_pct: Fraction of total samples to embargo after test fold
        horizon: Max prediction horizon (bars) — used for purging
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        horizon: int = 20,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.horizon = horizon

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups: pd.Series | None = None,
    ):
        """
        Generate train/test index splits with purging and embargo.

        Yields:
            (train_idx, test_idx) — numpy arrays of integer indices
        """
        n_samples = len(X)
        embargo_size = max(1, int(n_samples * self.embargo_pct))

        # Create K equally-sized contiguous folds
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n_samples

            # Test indices (contiguous block)
            test_idx = indices[test_start:test_end]

            # Initial train: everything outside the test block
            train_idx = np.concatenate([
                indices[:test_start],
                indices[test_end:],
            ])

            # ── Purge ────────────────────────────────────────────────────────
            # Remove training samples whose label window extends into test period.
            # A sample at time t with horizon h has label window [t+1, t+h].
            # Purge if t + horizon >= test_start (label overlaps with test)
            purge_cutoff = test_start - self.horizon
            train_idx = train_idx[
                (train_idx < purge_cutoff) | (train_idx >= test_end)
            ]

            # ── Embargo ──────────────────────────────────────────────────────
            # Remove training samples immediately AFTER the test fold
            # (they may contain leaked info about the test period's outcomes)
            embargo_end = min(test_end + embargo_size, n_samples)
            train_idx = train_idx[
                ~((train_idx >= test_end) & (train_idx < embargo_end))
            ]

            if len(train_idx) == 0:
                logger.warning(
                    "empty_train_fold",
                    fold=k,
                    test_start=test_start,
                    test_end=test_end,
                )
                continue

            logger.debug(
                "fold_split",
                fold=k,
                train_size=len(train_idx),
                test_size=len(test_idx),
                purge_cutoff=purge_cutoff,
            )

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class WalkForwardValidator:
    """
    Walk-forward validation — simulates real deployment.
    Trains on all data up to time T, tests on [T, T+window].
    Slides forward in time.

    This is the most realistic evaluation for trading systems.
    """

    def __init__(
        self,
        train_window: int = 500,  # bars of training data
        test_window: int = 100,   # bars of test data
        step_size: int = 50,      # bars to step forward each fold
        min_train_size: int = 200,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        groups=None,
    ):
        """Generate expanding or rolling window train/test splits."""
        n_samples = len(X)
        start = self.min_train_size

        while start + self.test_window <= n_samples:
            # Expanding window: use all data up to start
            train_start = max(0, start - self.train_window)
            train_idx = np.arange(train_start, start)
            test_idx = np.arange(start, min(start + self.test_window, n_samples))

            logger.debug(
                "walk_forward_split",
                train_start=train_start,
                train_end=start,
                test_start=start,
                test_end=start + len(test_idx),
            )

            yield train_idx, test_idx
            start += self.step_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return -1
        n_samples = len(X)
        count = 0
        start = self.min_train_size
        while start + self.test_window <= n_samples:
            count += 1
            start += self.step_size
        return count
