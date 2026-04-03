"""
AlphaForge Data Ingestion
Fetches OHLCV data from Binance (crypto) and Yahoo Finance (equities).
Writes to TimescaleDB with validation and retry logic.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─── Schema ───────────────────────────────────────────────────────────────────

OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


# ─── Base Ingester ────────────────────────────────────────────────────────────

class BaseIngester:
    """Common logic for all data sources."""

    def __init__(self) -> None:
        self.engine = create_engine(
            settings.db_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )

    def validate(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Basic data quality checks before writing."""
        if df.empty:
            raise ValueError(f"Empty DataFrame for {asset}")

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} for {asset}")

        # Drop rows with any nulls in core columns
        before = len(df)
        df = df.dropna(subset=list(REQUIRED_COLS))
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("dropped_null_rows", asset=asset, count=dropped)

        # Sanity: high >= low
        invalid = (df["high"] < df["low"]).sum()
        if invalid > 0:
            logger.warning("invalid_hl_rows", asset=asset, count=int(invalid))
            df = df[df["high"] >= df["low"]]

        # Sanity: no negative prices or volumes
        df = df[(df["close"] > 0) & (df["volume"] >= 0)]

        return df

    def write(
        self,
        df: pd.DataFrame,
        asset: str,
        source: str,
        timeframe: str,
    ) -> int:
        """Upsert OHLCV rows into TimescaleDB."""
        df = df.copy()
        df["asset"] = asset
        df["source"] = source
        df["timeframe"] = timeframe

        rows = df[["time", "asset", "source", "timeframe",
                   "open", "high", "low", "close", "volume"]].to_dict("records")

        upsert_sql = text("""
            INSERT INTO ohlcv (time, asset, source, timeframe, open, high, low, close, volume)
            VALUES (:time, :asset, :source, :timeframe, :open, :high, :low, :close, :volume)
            ON CONFLICT (time, asset, timeframe) DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)

        with self.engine.begin() as conn:
            conn.execute(upsert_sql, rows)

        logger.info("wrote_ohlcv", asset=asset, source=source,
                    timeframe=timeframe, rows=len(rows))
        return len(rows)


# ─── Binance Ingester ─────────────────────────────────────────────────────────

class BinanceIngester(BaseIngester):
    """Fetches OHLCV data from Binance via ccxt (no API key needed for public data)."""

    def __init__(self) -> None:
        super().__init__()
        self.exchange = ccxt.binance({
            "apiKey": settings.binance_api_key or None,
            "secret": settings.binance_secret or None,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Binance."""
        since_ms = None
        if since:
            since_ms = int(since.timestamp() * 1000)

        raw = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )

        if not raw:
            logger.warning("no_data_returned", symbol=symbol, timeframe=timeframe)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp"]).sort_values("time")

        return df

    def ingest(
        self,
        symbol: str,
        timeframe: str = "1h",
        lookback_days: int = 730,
    ) -> int:
        """Full ingestion: fetch → validate → write."""
        logger.info("starting_crypto_ingestion", symbol=symbol, timeframe=timeframe)

        since = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
        all_rows: list[pd.DataFrame] = []
        current_since = since

        # Paginate to get full history
        while True:
            df = self.fetch(symbol, timeframe=timeframe, since=current_since)
            if df.empty:
                break
            all_rows.append(df)
            last_time = df["time"].max()
            if last_time >= datetime.now(tz=timezone.utc) - timedelta(hours=2):
                break
            current_since = last_time + timedelta(seconds=1)
            time.sleep(0.5)  # respect rate limits

        if not all_rows:
            logger.warning("no_data_fetched", symbol=symbol)
            return 0

        full_df = pd.concat(all_rows, ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["time"])
        full_df = self.validate(full_df, asset=symbol)
        return self.write(full_df, asset=symbol, source="binance", timeframe=timeframe)


# ─── Yahoo Finance Ingester ───────────────────────────────────────────────────

class YFinanceIngester(BaseIngester):
    """Fetches OHLCV data from Yahoo Finance for equities."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def fetch(
        self,
        ticker: str,
        lookback_days: int = 730,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        raw = yf.download(
            ticker,
            start=start,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            logger.warning("yfinance_empty", ticker=ticker)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = raw.reset_index()
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]

        # Normalize column name for datetime index
        date_col = "date" if "date" in df.columns else "datetime"
        df = df.rename(columns={date_col: "time"})

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time")

        return df

    def ingest(
        self,
        ticker: str,
        lookback_days: int = 730,
        interval: str = "1d",
    ) -> int:
        """Full ingestion: fetch → validate → write."""
        logger.info("starting_equity_ingestion", ticker=ticker, interval=interval)

        df = self.fetch(ticker, lookback_days=lookback_days, interval=interval)
        df = self.validate(df, asset=ticker)
        return self.write(df, asset=ticker, source="yfinance", timeframe=interval)


# ─── Orchestration ────────────────────────────────────────────────────────────

def run_ingestion(
    crypto_assets: Optional[list[str]] = None,
    equity_assets: Optional[list[str]] = None,
    lookback_days: int = 730,
) -> dict[str, int]:
    """
    Run full ingestion for all configured assets.
    Returns dict of {asset: rows_written}.
    """
    results: dict[str, int] = {}

    crypto_assets = crypto_assets or settings.crypto_assets
    equity_assets = equity_assets or settings.equity_assets

    # ─── Crypto ───
    binance = BinanceIngester()
    for symbol in crypto_assets:
        try:
            n = binance.ingest(symbol, timeframe=settings.crypto_timeframe,
                               lookback_days=lookback_days)
            results[symbol] = n
        except Exception as e:
            logger.error("ingestion_failed", asset=symbol, error=str(e))
            results[symbol] = -1

    # ─── Equities ───
    yf_ingester = YFinanceIngester()
    for ticker in equity_assets:
        try:
            n = yf_ingester.ingest(ticker, lookback_days=lookback_days,
                                   interval=settings.equity_timeframe)
            results[ticker] = n
        except Exception as e:
            logger.error("ingestion_failed", asset=ticker, error=str(e))
            results[ticker] = -1

    logger.info("ingestion_complete", results=results)
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--assets", default=None, help="Comma-separated list of assets")
    @click.option("--source", type=click.Choice(["crypto", "equity", "all"]), default="all")
    @click.option("--lookback-days", default=730, type=int)
    def main(assets: Optional[str], source: str, lookback_days: int) -> None:
        from src.logger import setup_logging
        setup_logging()

        asset_list = assets.split(",") if assets else None

        crypto = asset_list if source == "crypto" else (None if source == "equity" else None)
        equity = asset_list if source == "equity" else (None if source == "crypto" else None)

        results = run_ingestion(
            crypto_assets=crypto,
            equity_assets=equity,
            lookback_days=lookback_days,
        )
        for asset, n in results.items():
            status = "✅" if n > 0 else "❌"
            print(f"{status} {asset}: {n} rows")

    main()
