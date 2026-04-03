"""
AlphaForge Feature Engineering
Computes alpha signals from raw OHLCV data:
  - Technical indicators (RSI, MACD, Bollinger, ATR, OBV, ADX)
  - Momentum features (multi-horizon, cross-sectional rank)
  - Volatility features (realized vol, vol ratio)
  - Microstructure proxies (order flow imbalance, Amihud illiquidity)
  - Forward return labels for supervised learning
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.logger import get_logger

logger = get_logger(__name__)


# ─── Technical Indicators ─────────────────────────────────────────────────────

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard technical analysis indicators.
    Input: DataFrame with OHLCV columns and DatetimeIndex.
    Returns: DataFrame with new feature columns.
    """
    out = df.copy()

    # RSI (14-period)
    out["rsi_14"] = ta.rsi(out["close"], length=14)

    # MACD (12, 26, 9)
    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd"] = macd.iloc[:, 0]          # MACD line
        out["macd_signal"] = macd.iloc[:, 2]   # Signal line
        out["macd_hist"] = macd.iloc[:, 1]     # Histogram

    # Bollinger Bands (20, 2)
    bb = ta.bbands(out["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        out["bb_upper"] = bb.iloc[:, 2]  # Upper band
        out["bb_lower"] = bb.iloc[:, 0]  # Lower band
        # %B: position within bands (0=lower, 1=upper)
        band_width = out["bb_upper"] - out["bb_lower"]
        out["bb_pct"] = (out["close"] - out["bb_lower"]) / band_width.replace(0, np.nan)

    # ATR (14-period Average True Range)
    out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    # OBV (On-Balance Volume)
    out["obv"] = ta.obv(out["close"], out["volume"])

    # ADX (Average Directional Index, 14-period)
    adx = ta.adx(out["high"], out["low"], out["close"], length=14)
    if adx is not None and not adx.empty:
        out["adx_14"] = adx.iloc[:, 0]  # ADX line

    logger.debug("computed_technical_features", n_features=6)
    return out


# ─── Momentum Features ────────────────────────────────────────────────────────

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-horizon momentum (rate of change) features.
    These are among the most powerful alpha signals in academic research.
    """
    out = df.copy()

    # Raw return over N bars
    for n in [1, 5, 20]:
        out[f"mom_{n}"] = out["close"].pct_change(n)

    # Price relative to rolling max/min (range position)
    out["price_range_pos_20"] = (
        (out["close"] - out["low"].rolling(20).min())
        / (out["high"].rolling(20).max() - out["low"].rolling(20).min()).replace(0, np.nan)
    )

    # Distance from 20-bar SMA (mean reversion signal)
    sma_20 = out["close"].rolling(20).mean()
    out["dist_from_sma_20"] = (out["close"] - sma_20) / sma_20.replace(0, np.nan)

    logger.debug("computed_momentum_features")
    return out


def compute_cross_sectional_momentum_rank(
    dfs: dict[str, pd.DataFrame],
    horizon: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Compute cross-sectional momentum rank.
    Rank each asset's N-bar return vs. all other assets at each timestamp.
    Rank 1.0 = top performer, 0.0 = worst performer.
    """
    # Align all assets on time index
    returns = pd.DataFrame({
        asset: df[f"mom_{horizon}"]
        for asset, df in dfs.items()
        if f"mom_{horizon}" in df.columns
    })

    # Rank within each row (cross-sectional at each timestamp)
    ranked = returns.rank(axis=1, pct=True)

    # Write back to individual DataFrames
    result = {}
    for asset, df in dfs.items():
        out = df.copy()
        if asset in ranked.columns:
            out["mom_rank"] = ranked[asset]
        else:
            out["mom_rank"] = np.nan
        result[asset] = out

    logger.debug("computed_cross_sectional_rank", assets=list(dfs.keys()))
    return result


# ─── Volatility Features ──────────────────────────────────────────────────────

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realized volatility features.
    Short/long vol ratio captures vol regime changes.
    """
    out = df.copy()

    log_ret = np.log(out["close"] / out["close"].shift(1))

    out["realized_vol_5"] = log_ret.rolling(5).std() * np.sqrt(252)
    out["realized_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)

    # Vol ratio: short/long (> 1 = rising vol, < 1 = falling vol)
    out["vol_ratio"] = (
        out["realized_vol_5"] / out["realized_vol_20"].replace(0, np.nan)
    )

    # Autocorrelation of returns (mean-reversion vs momentum regime)
    out["ret_autocorr_5"] = log_ret.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 5 else np.nan,
        raw=False,
    )

    logger.debug("computed_volatility_features")
    return out


# ─── Microstructure Features ──────────────────────────────────────────────────

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market microstructure features — proxies for order flow and liquidity.
    These require only OHLCV data (no L2 book needed).
    """
    out = df.copy()

    # Order Flow Imbalance proxy (Corwin & Schultz high-low spread estimator)
    # Positive OFI → buying pressure, Negative OFI → selling pressure
    hi = out["high"]
    lo = out["low"]
    cl = out["close"]
    op = out["open"]

    # Simple OFI proxy: (close - open) / (high - low)
    hl_range = (hi - lo).replace(0, np.nan)
    out["ofi"] = (cl - op) / hl_range

    # Rolling OFI (smoothed)
    out["ofi_ma5"] = out["ofi"].rolling(5).mean()

    # Amihud Illiquidity Ratio: |return| / volume
    # Higher = more illiquid (large price impact per unit volume)
    abs_ret = out["close"].pct_change().abs()
    dollar_vol = out["close"] * out["volume"]
    out["amihud"] = (abs_ret / dollar_vol.replace(0, np.nan)).rolling(20).mean() * 1e6

    # Volume surprise: current volume vs. 20-bar average
    avg_vol = out["volume"].rolling(20).mean()
    out["volume_surprise"] = out["volume"] / avg_vol.replace(0, np.nan)

    logger.debug("computed_microstructure_features")
    return out


# ─── Labels ───────────────────────────────────────────────────────────────────

def compute_labels(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute forward return labels for supervised learning.
    Binary: 1 = price went up over horizon, 0 = price went down.
    Note: forward returns are shifted backward — no lookahead in training
    if we use purged K-Fold CV.
    """
    horizons = horizons or [1, 5, 20]
    out = df.copy()

    for h in horizons:
        fwd_ret = out["close"].pct_change(h).shift(-h)
        out[f"fwd_ret_{h}"] = fwd_ret
        out[f"label_{h}"] = (fwd_ret > 0).astype(float)

    logger.debug("computed_labels", horizons=horizons)
    return out


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def compute_all_features(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run the full single-asset feature engineering pipeline.
    Order matters: some features depend on others.
    """
    df = compute_technical_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_microstructure_features(df)
    df = compute_labels(df, horizons=horizons)

    # Drop rows with NaN features (initial warmup period)
    n_before = len(df)
    df = df.dropna(subset=["rsi_14", "macd", "mom_20", "realized_vol_20"])
    n_after = len(df)

    logger.info(
        "feature_pipeline_complete",
        rows_before=n_before,
        rows_after=n_after,
        warmup_rows_dropped=n_before - n_after,
    )

    return df


# ─── Feature Column Registry ─────────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Technical
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_pct",
    "atr_14", "obv", "adx_14",
    # Momentum
    "mom_1", "mom_5", "mom_20", "mom_rank",
    "price_range_pos_20", "dist_from_sma_20",
    # Volatility
    "realized_vol_5", "realized_vol_20", "vol_ratio", "ret_autocorr_5",
    # Microstructure
    "ofi", "ofi_ma5", "amihud", "volume_surprise",
]

LABEL_COLUMNS = ["fwd_ret_1", "fwd_ret_5", "fwd_ret_20",
                 "label_1", "label_5", "label_20"]
