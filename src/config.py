"""
AlphaForge Configuration
Centralized settings using pydantic-settings.
All values can be overridden via environment variables or .env file.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ─── Environment ───
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ─── Database ───
    db_url: str = Field(
        default="postgresql://alphaforge:alphaforge@localhost:5433/alphaforge",
        alias="ALPHAFORGE_DB_URL",
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # ─── Redis ───
    redis_url: str = "redis://localhost:6379/0"
    signal_cache_ttl_seconds: int = 60  # 1-minute TTL on signals

    # ─── MLflow ───
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "alphaforge-v1"
    mlflow_model_name: str = "alphaforge-signal-engine"

    # ─── Assets ───
    crypto_assets: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    equity_assets: list[str] = ["SPY", "QQQ"]

    # ─── Data ───
    crypto_timeframe: str = "1h"          # Binance OHLCV timeframe
    equity_timeframe: str = "1d"          # Yahoo Finance daily
    lookback_days: int = 730              # 2 years of history
    feature_window: int = 60             # 60 bars for sequence models

    # ─── Model ───
    prediction_horizons: list[int] = [1, 5, 20]  # bars ahead
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10

    # ─── Evaluation Gates (model promotion thresholds) ───
    min_auc_roc: float = 0.55
    min_sharpe_ratio: float = 0.8
    max_drawdown_pct: float = 0.20
    min_information_coefficient: float = 0.03

    # ─── Drift Detection ───
    psi_threshold: float = 0.2   # Population Stability Index
    drift_window_days: int = 30

    # ─── API ───
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2

    # ─── Binance ───
    binance_api_key: str = ""       # optional for public data
    binance_secret: str = ""        # optional for public data

    @property
    def all_assets(self) -> list[str]:
        return self.crypto_assets + self.equity_assets

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()
