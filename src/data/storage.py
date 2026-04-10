"""
AlphaForge Storage Layer
SQLite for local dev, PostgreSQL for production.
Tables auto-created on first run — no manual DB setup needed.
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Boolean, Column, Double, Integer, SmallInteger,
    String, Text, UniqueConstraint,
    create_engine, event, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── SQLite performance pragmas ────────────────────────────────────────────────

def _set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


# ── ORM Models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class OHLCV(Base):
    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint("time", "asset", "timeframe", name="uq_ohlcv"),
    )

    id        = Column(Integer, primary_key=True, autoincrement=True)
    time      = Column(String, nullable=False, index=True)
    asset     = Column(String, nullable=False, index=True)
    source    = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    open      = Column(Double, nullable=False)
    high      = Column(Double, nullable=False)
    low       = Column(Double, nullable=False)
    close     = Column(Double, nullable=False)
    volume    = Column(Double, nullable=False)


class Features(Base):
    __tablename__ = "features"
    __table_args__ = (
        UniqueConstraint("time", "asset", "timeframe", name="uq_features"),
    )

    id        = Column(Integer, primary_key=True, autoincrement=True)
    time      = Column(String, nullable=False, index=True)
    asset     = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=False)

    # Technical
    rsi_14      = Column(Double)
    macd        = Column(Double)
    macd_signal = Column(Double)
    macd_hist   = Column(Double)
    bb_upper    = Column(Double)
    bb_lower    = Column(Double)
    bb_pct      = Column(Double)
    atr_14      = Column(Double)
    obv         = Column(Double)
    adx_14      = Column(Double)

    # Momentum
    mom_1              = Column(Double)
    mom_5              = Column(Double)
    mom_20             = Column(Double)
    mom_rank           = Column(Double)
    price_range_pos_20 = Column(Double)
    dist_from_sma_20   = Column(Double)

    # Volatility
    realized_vol_5  = Column(Double)
    realized_vol_20 = Column(Double)
    vol_ratio       = Column(Double)
    ret_autocorr_5  = Column(Double)

    # Microstructure
    ofi             = Column(Double)
    ofi_ma5         = Column(Double)
    amihud          = Column(Double)
    volume_surprise = Column(Double)

    # Labels
    fwd_ret_1  = Column(Double)
    fwd_ret_5  = Column(Double)
    fwd_ret_20 = Column(Double)
    label_1    = Column(SmallInteger)
    label_5    = Column(SmallInteger)
    label_20   = Column(SmallInteger)


class Predictions(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("time", "asset", "horizon", name="uq_predictions"),
    )

    id            = Column(Integer, primary_key=True, autoincrement=True)
    time          = Column(String, nullable=False, index=True)
    asset         = Column(String, nullable=False)
    horizon       = Column(Integer, nullable=False)
    model_version = Column(String, nullable=False)
    prob_up       = Column(Double, nullable=False)
    signal        = Column(SmallInteger, nullable=False)
    confidence    = Column(Double, nullable=False)
    latency_ms    = Column(Double)


class BacktestResults(Base):
    __tablename__ = "backtest_results"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    run_id        = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    strategy      = Column(String, nullable=False)
    start_date    = Column(String)
    end_date      = Column(String)
    total_return  = Column(Double)
    sharpe_ratio  = Column(Double)
    sortino_ratio = Column(Double)
    max_drawdown  = Column(Double)
    calmar_ratio  = Column(Double)
    win_rate      = Column(Double)
    total_trades  = Column(Integer)
    turnover      = Column(Double)
    auc_roc       = Column(Double)
    f1_score      = Column(Double)
    accuracy      = Column(Double)
    information_coeff = Column(Double)
    created_at    = Column(String, default=lambda: datetime.utcnow().isoformat())


class DriftReports(Base):
    __tablename__ = "drift_reports"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    report_date       = Column(String, nullable=False)
    asset             = Column(String)
    psi_score         = Column(Double)
    drift_detected    = Column(Boolean, nullable=False)
    affected_features = Column(Text)
    report_path       = Column(String)
    created_at        = Column(String, default=lambda: datetime.utcnow().isoformat())


# ── Engine & Session ──────────────────────────────────────────────────────────

@lru_cache
def get_engine():
    db_url = settings.db_url

    if db_url.startswith("sqlite"):
        db_path = db_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        event.listen(engine, "connect", _set_sqlite_pragma)
    else:
        engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )

    Base.metadata.create_all(engine)
    logger.info("database_ready", url=db_url)
    return engine


def get_session() -> Session:
    return sessionmaker(bind=get_engine())()


# ── Upsert Helpers ────────────────────────────────────────────────────────────

def _sqlite_upsert(conn, table: str, unique_cols: list[str],
                   data_cols: list[str], rows: list[dict]) -> int:
    """
    SQLite-compatible upsert using INSERT OR REPLACE.
    Works because we defined UniqueConstraint on the table.
    """
    all_cols = unique_cols + data_cols
    col_list = ", ".join(all_cols)
    val_list = ", ".join(f":{c}" for c in all_cols)

    sql = text(f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({val_list})")
    conn.execute(sql, rows)
    return len(rows)


def upsert_ohlcv(df: pd.DataFrame, asset: str, source: str, timeframe: str) -> int:
    """Insert/update OHLCV rows. Works with SQLite and PostgreSQL."""
    engine = get_engine()
    df = df.copy().reset_index()
    df["time"]      = df["time"].astype(str)
    df["asset"]     = asset
    df["source"]    = source
    df["timeframe"] = timeframe

    cols = ["time", "asset", "source", "timeframe",
            "open", "high", "low", "close", "volume"]
    rows = df[cols].to_dict("records")

    with engine.begin() as conn:
        n = _sqlite_upsert(
            conn, "ohlcv",
            unique_cols=["time", "asset", "source", "timeframe"],
            data_cols=["open", "high", "low", "close", "volume"],
            rows=rows,
        )

    logger.info("wrote_ohlcv", asset=asset, rows=n)
    return n


def upsert_features(df: pd.DataFrame, asset: str, timeframe: str) -> int:
    """Insert/update feature rows. Works with SQLite and PostgreSQL."""
    from src.features.technical import FEATURE_COLUMNS, LABEL_COLUMNS

    engine = get_engine()
    df = df.copy().reset_index()
    df["time"]      = df["time"].astype(str)
    df["asset"]     = asset
    df["timeframe"] = timeframe

    # Only keep columns that exist in both df and our schema
    all_schema_cols = (["time", "asset", "timeframe"]
                       + FEATURE_COLUMNS + LABEL_COLUMNS)
    cols = [c for c in all_schema_cols if c in df.columns]

    # Replace NaN with None so SQLite stores NULL cleanly
    df_out = df[cols].where(pd.notnull(df[cols]), other=None)
    rows = df_out.to_dict("records")

    unique_cols  = ["time", "asset", "timeframe"]
    data_cols    = [c for c in cols if c not in unique_cols]

    with engine.begin() as conn:
        n = _sqlite_upsert(conn, "features",
                           unique_cols=unique_cols,
                           data_cols=data_cols,
                           rows=rows)

    logger.info("wrote_features", asset=asset, rows=n)
    return n


# ── Query Helpers ─────────────────────────────────────────────────────────────

def load_ohlcv(
    asset: str,
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    conditions = "WHERE asset=:asset AND timeframe=:timeframe"
    params: dict = {"asset": asset, "timeframe": timeframe}
    if start:
        conditions += " AND time >= :start"
        params["start"] = str(start)
    if end:
        conditions += " AND time <= :end"
        params["end"] = str(end)

    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(f"SELECT time, open, high, low, close, volume "
                 f"FROM ohlcv {conditions} ORDER BY time ASC"),
            conn, params=params,
        )

    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def load_features(
    asset: str,
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    conditions = "WHERE asset=:asset AND timeframe=:timeframe"
    params: dict = {"asset": asset, "timeframe": timeframe}
    if start:
        conditions += " AND time >= :start"
        params["start"] = str(start)
    if end:
        conditions += " AND time <= :end"
        params["end"] = str(end)

    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(f"SELECT * FROM features {conditions} ORDER BY time ASC"),
            conn, params=params,
        )

    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def load_multi_asset_features(
    assets: list[str],
    timeframe: str,
    start: Optional[datetime] = None,
) -> pd.DataFrame:
    placeholders = ", ".join(f"'{a}'" for a in assets)
    conditions   = f"WHERE asset IN ({placeholders}) AND timeframe=:timeframe"
    params: dict = {"timeframe": timeframe}
    if start:
        conditions += " AND time >= :start"
        params["start"] = str(start)

    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(f"SELECT * FROM features {conditions} ORDER BY time ASC"),
            conn, params=params,
        )

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df