"""
AlphaForge Configuration
Local-first defaults: SQLite DB, local MLflow, no Redis required.
Override any value via .env file or environment variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (where .env lives)
ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ────────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── Database ───────────────────────────────────────────────────────────────
    # Default: SQLite in data/ folder — zero install, works everywhere
    db_url: str = Field(
        default="sqlite:///data/alphaforge.db",
        alias="ALPHAFORGE_DB_URL",
    )

    # ── Feature Store ──────────────────────────────────────────────────────────
    # Parquet files stored locally — no Feast, no server
    feature_store_path: str = Field(
        default="data/features",
        alias="FEATURE_STORE_PATH",
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    # "mlruns" = local folder. No server needed.
    # View UI with: python -m mlflow ui  (then open http://localhost:5000)
    mlflow_tracking_uri: str = Field(
        default="mlruns",
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = "alphaforge-v1"
    mlflow_model_name: str = "alphaforge-signal-engine"

    # ── Redis ─────────────────────────────────────────────────────────────────
    # Optional — if empty, API uses in-memory dict cache instead
    redis_url: str = Field(default="", alias="REDIS_URL")
    signal_cache_ttl_seconds: int = 60

    # ── Assets ────────────────────────────────────────────────────────────────
    crypto_assets: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    equity_assets: list[str] = ["SPY", "QQQ"]

    # ── Data ──────────────────────────────────────────────────────────────────
    crypto_timeframe: str = "1h"
    equity_timeframe: str = "1d"
    lookback_days: int = 365       # 1 year default for local dev (faster)
    feature_window: int = 60

    # ── Model ─────────────────────────────────────────────────────────────────
    prediction_horizons: list[int] = [1, 5, 20]
    batch_size: int = 32
    max_epochs: int = 50
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10

    # ── Evaluation Gates ──────────────────────────────────────────────────────
    min_auc_roc: float = 0.55
    min_sharpe_ratio: float = 0.8
    max_drawdown_pct: float = 0.20
    min_information_coefficient: float = 0.03

    # ── Drift ─────────────────────────────────────────────────────────────────
    psi_threshold: float = 0.2
    drift_window_days: int = 30

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Binance ───────────────────────────────────────────────────────────────
    binance_api_key: str = ""
    binance_secret: str = ""

    @property
    def all_assets(self) -> list[str]:
        return self.crypto_assets + self.equity_assets

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url)

    @property
    def feature_store_dir(self) -> Path:
        p = ROOT / self.feature_store_path
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        p = ROOT / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()