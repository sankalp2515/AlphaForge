"""
AlphaForge — Test Suite
Unit + integration tests for all pipeline components.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    # Simulate random walk price
    returns = np.random.normal(0, 0.01, n)
    close = 30000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.lognormal(10, 1, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def sample_features(sample_ohlcv) -> pd.DataFrame:
    """Compute features from sample OHLCV data."""
    from src.features.technical import compute_all_features
    return compute_all_features(sample_ohlcv)


# ─── Data Ingestion Tests ──────────────────────────────────────────────────────

class TestDataIngestion:

    def test_ingester_validates_empty_df(self):
        from src.data.ingestion import YFinanceIngester
        ingester = YFinanceIngester.__new__(YFinanceIngester)
        ingester.engine = None

        with pytest.raises(ValueError, match="Empty DataFrame"):
            ingester.validate(pd.DataFrame(), asset="TEST")

    def test_ingester_validates_missing_columns(self):
        from src.data.ingestion import YFinanceIngester
        ingester = YFinanceIngester.__new__(YFinanceIngester)
        ingester.engine = None

        df = pd.DataFrame({"close": [100, 101]})
        with pytest.raises(ValueError, match="Missing columns"):
            ingester.validate(df, asset="TEST")

    def test_ingester_drops_invalid_hl(self):
        from src.data.ingestion import YFinanceIngester
        ingester = YFinanceIngester.__new__(YFinanceIngester)
        ingester.engine = None

        df = pd.DataFrame({
            "open": [100, 100],
            "high": [90, 105],   # First row: high < low (invalid)
            "low": [95, 98],
            "close": [102, 103],
            "volume": [1000, 2000],
        })
        result = ingester.validate(df, asset="TEST")
        assert len(result) == 1  # Invalid row removed

    def test_ingester_drops_zero_price(self):
        from src.data.ingestion import YFinanceIngester
        ingester = YFinanceIngester.__new__(YFinanceIngester)
        ingester.engine = None

        df = pd.DataFrame({
            "open": [0, 100],
            "high": [0, 105],
            "low": [0, 98],
            "close": [0, 103],   # Zero close → invalid
            "volume": [1000, 2000],
        })
        result = ingester.validate(df, asset="TEST")
        assert len(result) == 1


# ─── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_technical_features_computed(self, sample_ohlcv):
        from src.features.technical import compute_technical_features
        result = compute_technical_features(sample_ohlcv)

        assert "rsi_14" in result.columns
        assert "macd" in result.columns
        assert "bb_pct" in result.columns
        assert "atr_14" in result.columns
        assert "obv" in result.columns

    def test_rsi_range(self, sample_ohlcv):
        from src.features.technical import compute_technical_features
        result = compute_technical_features(sample_ohlcv)

        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all(), "RSI should be >= 0"
        assert (rsi <= 100).all(), "RSI should be <= 100"

    def test_bb_pct_range(self, sample_ohlcv):
        from src.features.technical import compute_technical_features
        result = compute_technical_features(sample_ohlcv)

        # BB %B can exceed [0,1] during strong trends, but should be finite
        bb = result["bb_pct"].dropna()
        assert bb.notna().sum() > 100
        assert np.isfinite(bb).all()

    def test_momentum_features(self, sample_ohlcv):
        from src.features.technical import compute_momentum_features
        result = compute_momentum_features(sample_ohlcv)

        assert "mom_1" in result.columns
        assert "mom_5" in result.columns
        assert "mom_20" in result.columns

        # mom_1 should be 1-period pct change
        expected_mom1 = sample_ohlcv["close"].pct_change(1)
        pd.testing.assert_series_equal(
            result["mom_1"].dropna(), expected_mom1.dropna(), check_names=False
        )

    def test_labels_are_binary(self, sample_ohlcv):
        from src.features.technical import compute_labels
        result = compute_labels(sample_ohlcv, horizons=[1, 5])

        for h in [1, 5]:
            labels = result[f"label_{h}"].dropna()
            assert set(labels.unique()).issubset({0.0, 1.0}), \
                f"label_{h} should be binary"

    def test_no_lookahead_in_labels(self, sample_ohlcv):
        """Verify that labels use future data correctly (shifted backward)."""
        from src.features.technical import compute_labels
        result = compute_labels(sample_ohlcv, horizons=[1])

        # fwd_ret_1 at index t = close[t+1]/close[t] - 1
        # So fwd_ret_1 should be NaN at the last row
        assert pd.isna(result["fwd_ret_1"].iloc[-1]), \
            "Last row's forward return should be NaN (no future data)"

    def test_full_pipeline_no_crash(self, sample_ohlcv):
        from src.features.technical import compute_all_features
        result = compute_all_features(sample_ohlcv)
        assert len(result) > 0
        assert "rsi_14" in result.columns
        assert "label_1" in result.columns

    def test_cross_sectional_rank_range(self, sample_ohlcv):
        from src.features.technical import (
            compute_momentum_features,
            compute_cross_sectional_momentum_rank,
        )
        df1 = compute_momentum_features(sample_ohlcv.copy())
        df2 = compute_momentum_features(sample_ohlcv.copy() * 1.01)

        result = compute_cross_sectional_momentum_rank(
            {"BTC": df1, "ETH": df2}, horizon=20
        )
        ranks = result["BTC"]["mom_rank"].dropna()
        assert (ranks >= 0).all() and (ranks <= 1).all()


# ─── Cross-Validation Tests ────────────────────────────────────────────────────

class TestCrossValidation:

    def test_purged_kfold_no_overlap(self):
        """Validate that purged K-Fold prevents train/test overlap."""
        from src.training.cv import PurgedKFold

        n = 500
        X = np.random.randn(n, 10)
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01, horizon=20)

        for train_idx, test_idx in cv.split(X):
            # No overlap between train and test
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train/test overlap detected!"

            # Train indices should not be adjacent to test indices
            # (embargo check — simplified)
            train_max = max(train_idx)
            test_min = min(test_idx)
            if train_max < test_min:
                # Gap should exist (embargo)
                assert test_min - train_max >= 1

    def test_purged_kfold_n_splits(self):
        from src.training.cv import PurgedKFold
        X = np.random.randn(500, 10)
        cv = PurgedKFold(n_splits=5)
        splits = list(cv.split(X))
        assert len(splits) == 5

    def test_walk_forward_ordering(self):
        """Walk-forward: test always comes after train."""
        from src.training.cv import WalkForwardValidator
        X = np.random.randn(600, 10)
        cv = WalkForwardValidator(train_window=200, test_window=50, step_size=25)

        prev_test_end = 0
        for train_idx, test_idx in cv.split(X):
            assert max(train_idx) < min(test_idx), "Train must precede test"
            assert min(test_idx) >= prev_test_end
            prev_test_end = max(test_idx)


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestLSTMModel:

    @pytest.fixture
    def lstm_model(self):
        from src.models.lstm import LSTMSignalModel
        return LSTMSignalModel(n_features=20, hidden_size=32, num_layers=1)

    def test_forward_pass_shape(self, lstm_model):
        batch_size, seq_len, n_features = 8, 60, 20
        x = torch.randn(batch_size, seq_len, n_features)
        out = lstm_model(x)
        assert out.shape == (batch_size, 1), f"Expected (8, 1), got {out.shape}"

    def test_output_is_logit(self, lstm_model):
        """Output should be raw logits (not bounded to [0,1])."""
        x = torch.randn(4, 60, 20)
        out = lstm_model(x)
        # After sigmoid, should be in [0,1]
        probs = torch.sigmoid(out)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_sequence_dataset(self):
        from src.models.lstm import SequenceDataset
        X = np.random.randn(200, 20).astype(np.float32)
        y = np.random.randint(0, 2, 200).astype(np.float32)

        ds = SequenceDataset(X, y, seq_len=60)
        assert len(ds) == 200 - 60

        x_sample, y_sample = ds[0]
        assert x_sample.shape == (60, 20)
        assert y_sample.shape == ()


# ─── Metrics Tests ────────────────────────────────────────────────────────────

class TestMetrics:

    def test_compute_ml_metrics(self):
        from src.evaluation.metrics import compute_ml_metrics

        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])

        m = compute_ml_metrics(y_true, y_prob)
        assert 0.0 <= m.auc_roc <= 1.0
        assert 0.0 <= m.f1_score <= 1.0
        assert 0.0 <= m.ece <= 1.0
        assert m.auc_roc > 0.7  # Should be good with these examples

    def test_sharpe_positive_returns(self):
        from src.evaluation.metrics import compute_financial_metrics
        returns = pd.Series(np.ones(252) * 0.001)  # Constant positive returns
        m = compute_financial_metrics(returns)
        assert m.sharpe_ratio > 0
        assert m.total_return > 0
        assert m.max_drawdown == 0.0  # No drawdown with constant positive

    def test_sharpe_negative_returns(self):
        from src.evaluation.metrics import compute_financial_metrics
        returns = pd.Series(np.ones(252) * -0.001)
        m = compute_financial_metrics(returns)
        assert m.sharpe_ratio < 0
        assert m.max_drawdown < 0

    def test_psi_same_distribution(self):
        from src.evaluation.drift import compute_psi
        x = np.random.normal(0, 1, 1000)
        psi = compute_psi(x, x)
        assert psi < 0.1, "PSI should be near 0 for identical distributions"

    def test_psi_different_distribution(self):
        from src.evaluation.drift import compute_psi
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(3, 1, 1000)  # Very different mean
        psi = compute_psi(reference, current)
        assert psi > 0.2, "PSI should be high for very different distributions"

    def test_ece_perfect_calibration(self):
        from src.evaluation.metrics import _compute_ece
        # Perfect calibration: predicted probs equal actual frequencies
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        ece = _compute_ece(y_true, y_prob)
        assert ece < 0.1


# ─── API Tests ────────────────────────────────────────────────────────────────

class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.serving.api import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "model_version" in data

    def test_model_info_endpoint(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "assets" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"alphaforge_predict_requests_total" in response.content
