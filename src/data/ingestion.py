"""
AlphaForge Data Ingestion
Fetches OHLCV from Binance (crypto) and Yahoo Finance (equities).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import ccxt
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.data.storage import upsert_ohlcv
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


# ── Validation ────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"Empty DataFrame for {asset}")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} for {asset}")
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLS))
    df = df[df["high"] >= df["low"]]
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("dropped_invalid_rows", asset=asset, count=dropped)
    return df


# ── Binance ───────────────────────────────────────────────────────────────────

class BinanceIngester:

    def __init__(self) -> None:
        self.exchange = ccxt.binance({
            "apiKey":  settings.binance_api_key or None,
            "secret":  settings.binance_secret or None,
            "enableRateLimit": True,
        })

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def fetch(self, symbol: str, timeframe: str = "1h",
              since: Optional[datetime] = None) -> pd.DataFrame:
        since_ms = int(since.timestamp() * 1000) if since else None
        raw = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=1000
        )
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw,
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.drop(columns=["timestamp"]).sort_values("time")

    def ingest(self, symbol: str, timeframe: str = "1h",
               lookback_days: int = 365) -> int:
        logger.info("ingesting_crypto", symbol=symbol)
        since   = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
        frames: list[pd.DataFrame] = []
        current = since

        while True:
            df = self.fetch(symbol, timeframe=timeframe, since=current)
            if df.empty:
                break
            frames.append(df)
            last = df["time"].max()
            if last >= datetime.now(tz=timezone.utc) - timedelta(hours=2):
                break
            current = last + timedelta(seconds=1)
            time.sleep(0.3)

        if not frames:
            logger.warning("no_data_fetched", symbol=symbol)
            return 0

        full = pd.concat(frames).drop_duplicates(subset=["time"])
        full = validate(full, symbol)
        full = full.set_index("time")
        n = upsert_ohlcv(full, asset=symbol, source="binance", timeframe=timeframe)
        logger.info("crypto_ingest_done", symbol=symbol, rows=n)
        return n


# ── Yahoo Finance ─────────────────────────────────────────────────────────────

class YFinanceIngester:

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    def fetch(self, ticker: str, lookback_days: int = 365,
              interval: str = "1d") -> pd.DataFrame:

        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        raw = yf.download(
            ticker,
            start=start,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,   # yfinance >=0.2.38 flat columns flag
        )

        if raw.empty:
            logger.warning("yfinance_empty_response", ticker=ticker)
            return pd.DataFrame()

        df = raw.reset_index()

        # ── Flatten MultiIndex columns (yfinance sometimes returns these) ──────
        if isinstance(df.columns, pd.MultiIndex):
            # e.g. ("Close", "SPY") -> "close"
            df.columns = [
                str(col[0]).lower().strip() if isinstance(col, tuple) else str(col).lower().strip()
                for col in df.columns
            ]
        else:
            df.columns = [str(c).lower().strip() for c in df.columns]

        logger.debug("yfinance_columns", ticker=ticker, columns=list(df.columns))

        # ── Find the datetime column ────────────────────────────────────────────
        # yfinance uses "Date" for daily, "Datetime" for intraday
        time_col = None
        for candidate in ["date", "datetime", "index", "timestamp"]:
            if candidate in df.columns:
                time_col = candidate
                break

        if time_col is None:
            logger.error("yfinance_no_time_col",
                         ticker=ticker, columns=list(df.columns))
            return pd.DataFrame()

        df = df.rename(columns={time_col: "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True)

        # ── Ensure standard OHLCV column names ─────────────────────────────────
        # yfinance sometimes returns "adj close" — rename to "close"
        if "adj close" in df.columns and "close" not in df.columns:
            df = df.rename(columns={"adj close": "close"})

        required = ["time", "open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            logger.error("yfinance_missing_cols",
                         ticker=ticker, missing=missing, columns=list(df.columns))
            return pd.DataFrame()

        return df[required].sort_values("time")

    def ingest(self, ticker: str, lookback_days: int = 365,
               interval: str = "1d") -> int:
        logger.info("ingesting_equity", ticker=ticker)
        df = self.fetch(ticker, lookback_days=lookback_days, interval=interval)
        if df.empty:
            return 0
        df = validate(df, ticker)
        df = df.set_index("time")
        n = upsert_ohlcv(df, asset=ticker, source="yfinance", timeframe=interval)
        logger.info("equity_ingest_done", ticker=ticker, rows=n)
        return n


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_ingestion(
    crypto_assets: Optional[list[str]] = None,
    equity_assets: Optional[list[str]] = None,
    lookback_days: int = 365,
) -> dict[str, int]:
    results: dict[str, int] = {}

    crypto_assets = crypto_assets or settings.crypto_assets
    equity_assets = equity_assets or settings.equity_assets

    binance = BinanceIngester()
    for symbol in crypto_assets:
        try:
            results[symbol] = binance.ingest(
                symbol,
                timeframe=settings.crypto_timeframe,
                lookback_days=lookback_days,
            )
        except Exception as e:
            logger.error("ingest_failed", asset=symbol, error=str(e))
            results[symbol] = -1

    yf_ing = YFinanceIngester()
    for ticker in equity_assets:
        try:
            results[ticker] = yf_ing.ingest(
                ticker,
                lookback_days=lookback_days,
                interval=settings.equity_timeframe,
            )
        except Exception as e:
            logger.error("ingest_failed", asset=ticker, error=str(e))
            results[ticker] = -1

    logger.info("ingestion_complete", results=results)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()

    print("Starting data ingestion...")
    results = run_ingestion(lookback_days=settings.lookback_days)
    print()
    for asset, n in results.items():
        status = "✅" if n > 0 else "❌"
        print(f"  {status}  {asset}: {n} rows")