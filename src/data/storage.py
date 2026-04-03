"""
AlphaForge Storage Layer
SQLAlchemy ORM models and query helpers for TimescaleDB.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Optional

import pandas as pd
from sqlalchemy import (
    BigInteger, Column, Date, Double, Integer,
    SmallInteger, String, Text, create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─── ORM Models ──────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class OHLCV(Base):
    __tablename__ = "ohlcv"

    time = Column(String, primary_key=True)       # TIMESTAMPTZ as string
    asset = Column(String, primary_key=True)
    timeframe = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    open = Column(Double, nullable=False)
    high = Column(Double, nullable=False)
    low = Column(Double, nullable=False)
    close = Column(Double, nullable=False)
    volume = Column(Double, nullable=False)


class Features(Base):
    __tablename__ = "features"

    time = Column(String, primary_key=True)
    asset = Column(String, primary_key=True)
    timeframe = Column(String, primary_key=True)
    rsi_14 = Column(Double)
    macd = Column(Double)
    macd_signal = Column(Double)
    macd_hist = Column(Double)
    bb_upper = Column(Double)
    bb_lower = Column(Double)
    bb_pct = Column(Double)
    atr_14 = Column(Double)
    obv = Column(Double)
    adx_14 = Column(Double)
    mom_1 = Column(Double)
    mom_5 = Column(Double)
    mom_20 = Column(Double)
    mom_rank = Column(Double)
    realized_vol_5 = Column(Double)
    realized_vol_20 = Column(Double)
    vol_ratio = Column(Double)
    ofi = Column(Double)
    amihud = Column(Double)
    fwd_ret_1 = Column(Double)
    fwd_ret_5 = Column(Double)
    fwd_ret_20 = Column(Double)
    label_1 = Column(SmallInteger)
    label_5 = Column(SmallInteger)
    label_20 = Column(SmallInteger)


class Predictions(Base):
    __tablename__ = "predictions"

    time = Column(String, primary_key=True)
    asset = Column(String, primary_key=True)
    horizon = Column(Integer, primary_key=True)
    model_version = Column(String, nullable=False)
    prob_up = Column(Double, nullable=False)
    signal = Column(SmallInteger, nullable=False)
    confidence = Column(Double, nullable=False)
    latency_ms = Column(Double)


class BacktestResults(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    total_return = Column(Double)
    sharpe_ratio = Column(Double)
    sortino_ratio = Column(Double)
    max_drawdown = Column(Double)
    calmar_ratio = Column(Double)
    win_rate = Column(Double)
    total_trades = Column(Integer)
    turnover = Column(Double)
    auc_roc = Column(Double)
    f1_score = Column(Double)
    accuracy = Column(Double)
    information_coeff = Column(Double)


# ─── Engine & Session ─────────────────────────────────────────────────────────

@lru_cache
def get_engine():
    return create_engine(
        settings.db_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )


@lru_cache
def get_session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine())


def get_session() -> Session:
    return get_session_factory()()


# ─── Query Helpers ────────────────────────────────────────────────────────────

def load_ohlcv(
    asset: str,
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load OHLCV data for a single asset into a DataFrame."""
    query = """
        SELECT time, open, high, low, close, volume
        FROM ohlcv
        WHERE asset = :asset AND timeframe = :timeframe
        {start_filter}
        {end_filter}
        ORDER BY time ASC
    """.format(
        start_filter="AND time >= :start" if start else "",
        end_filter="AND time <= :end" if end else "",
    )

    params: dict = {"asset": asset, "timeframe": timeframe}
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["time"])

    df = df.set_index("time").sort_index()
    logger.debug("loaded_ohlcv", asset=asset, timeframe=timeframe, rows=len(df))
    return df


def load_features(
    asset: str,
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load engineered features for a single asset."""
    query = """
        SELECT *
        FROM features
        WHERE asset = :asset AND timeframe = :timeframe
        {start_filter}
        {end_filter}
        ORDER BY time ASC
    """.format(
        start_filter="AND time >= :start" if start else "",
        end_filter="AND time <= :end" if end else "",
    )

    params: dict = {"asset": asset, "timeframe": timeframe}
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["time"])

    df = df.set_index("time").sort_index()
    return df


def load_multi_asset_features(
    assets: list[str],
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load features for multiple assets (for cross-sectional modeling)."""
    query = """
        SELECT *
        FROM features
        WHERE asset = ANY(:assets) AND timeframe = :timeframe
        {start_filter}
        ORDER BY time ASC, asset ASC
    """.format(
        start_filter="AND time >= :start" if start else "",
    )

    params: dict = {"assets": assets, "timeframe": timeframe}
    if start:
        params["start"] = start

    with get_engine().connect() as conn:
        df = pd.read_sql(text(query), conn, params=params, parse_dates=["time"])

    return df
