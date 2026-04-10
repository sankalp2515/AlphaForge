"""
AlphaForge Feature Engineering
Computes alpha signals from raw OHLCV data using the `ta` library.

Changed from pandas-ta (pre-release, PyPI conflicts) to `ta`
(stable, pure-Python, zero dependency conflicts).

Signals computed:
  - Technical indicators (RSI, MACD, Bollinger, ATR, OBV, ADX)
  - Momentum features (multi-horizon, cross-sectional rank)
  - Volatility features (realized vol, vol ratio)
  - Microstructure proxies (order flow imbalance, Amihud illiquidity)
  - Forward return labels for supervised learning
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

from src.logger import get_logger

logger = get_logger(__name__)


# ─── Technical Indicators ─────────────────────────────────────────────────────

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard technical analysis indicators using the `ta` library.
    Input: DataFrame with open/high/low/close/volume columns and DatetimeIndex.
    """
    out = df.copy()
    close = out["close"]
    high  = out["high"]
    low   = out["low"]
    vol   = out["volume"]

    # RSI (14-period)
    out["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # MACD (12, 26, 9)
    macd_ind = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    out["macd"]        = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()
    out["macd_hist"]   = macd_ind.macd_diff()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_pct"]   = bb.bollinger_pband()   # %B: 0=lower, 1=upper

    # ATR (14-period Average True Range)
    out["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # OBV (On-Balance Volume)
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=vol
    ).on_balance_volume()

    # ADX (Average Directional Index, 14-period)
    out["adx_14"] = ta.trend.ADXIndicator(
        high=high, low=low, close=close, window=14
    ).adx()

    logger.debug("computed_technical_features", n_features=10)
    return out


# ─── Momentum Features ────────────────────────────────────────────────────────

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-horizon momentum and mean-reversion features."""
    out = df.copy()

    for n in [1, 5, 20]:
        out[f"mom_{n}"] = out["close"].pct_change(n)

    # Price position within 20-bar range (0=bottom, 1=top)
    low_20  = out["low"].rolling(20).min()
    high_20 = out["high"].rolling(20).max()
    out["price_range_pos_20"] = (
        (out["close"] - low_20) / (high_20 - low_20).replace(0, np.nan)
    )

    # Distance from 20-bar SMA (mean-reversion signal)
    sma_20 = out["close"].rolling(20).mean()
    out["dist_from_sma_20"] = (out["close"] - sma_20) / sma_20.replace(0, np.nan)

    logger.debug("computed_momentum_features")
    return out


def compute_cross_sectional_momentum_rank(
    dfs: dict[str, pd.DataFrame],
    horizon: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Rank each asset's N-bar return vs. all assets at each timestamp.
    Rank 1.0 = best performer, 0.0 = worst.
    """
    returns = pd.DataFrame({
        asset: df[f"mom_{horizon}"]
        for asset, df in dfs.items()
        if f"mom_{horizon}" in df.columns
    })
    ranked = returns.rank(axis=1, pct=True)

    result = {}
    for asset, df in dfs.items():
        out = df.copy()
        out["mom_rank"] = ranked[asset] if asset in ranked.columns else np.nan
        result[asset] = out

    logger.debug("computed_cross_sectional_rank", assets=list(dfs.keys()))
    return result


# ─── Volatility Features ──────────────────────────────────────────────────────

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Realized volatility and vol-regime features."""
    out = df.copy()
    log_ret = np.log(out["close"] / out["close"].shift(1))

    out["realized_vol_5"]  = log_ret.rolling(5).std()  * np.sqrt(252)
    out["realized_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    out["vol_ratio"] = (
        out["realized_vol_5"] / out["realized_vol_20"].replace(0, np.nan)
    )

    # Return autocorrelation: >0 = momentum regime, <0 = mean-reversion regime
    out["ret_autocorr_5"] = log_ret.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 5 else np.nan,
        raw=False,
    )

    logger.debug("computed_volatility_features")
    return out


# ─── Microstructure Features ──────────────────────────────────────────────────

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Microstructure proxies from OHLCV data.
    No Level-2 order book required.
    """
    out = df.copy()
    hi = out["high"]
    lo = out["low"]
    cl = out["close"]
    op = out["open"]

    # Order Flow Imbalance proxy: (close-open)/(high-low)
    hl_range = (hi - lo).replace(0, np.nan)
    out["ofi"]     = (cl - op) / hl_range
    out["ofi_ma5"] = out["ofi"].rolling(5).mean()

    # Amihud Illiquidity: |return| / dollar_volume (scaled)
    abs_ret    = cl.pct_change().abs()
    dollar_vol = cl * out["volume"]
    out["amihud"] = (
        (abs_ret / dollar_vol.replace(0, np.nan)).rolling(20).mean() * 1e6
    )

    # Volume surprise: current vs 20-bar average
    avg_vol = out["volume"].rolling(20).mean()
    out["volume_surprise"] = out["volume"] / avg_vol.replace(0, np.nan)

    logger.debug("computed_microstructure_features")
    return out


# ─── Labels ───────────────────────────────────────────────────────────────────

def compute_labels(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Binary forward-return labels for supervised learning."""
    horizons = horizons or [1, 5, 20]
    out = df.copy()
    for h in horizons:
        fwd_ret = out["close"].pct_change(h).shift(-h)
        out[f"fwd_ret_{h}"] = fwd_ret
        out[f"label_{h}"]   = (fwd_ret > 0).astype(float)
    logger.debug("computed_labels", horizons=horizons)
    return out


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def compute_all_features(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Run the full single-asset feature pipeline."""
    df = compute_technical_features(df)
    df = compute_momentum_features(df)
    df = compute_volatility_features(df)
    df = compute_microstructure_features(df)
    df = compute_labels(df, horizons=horizons)

    n_before = len(df)
    df = df.dropna(subset=["rsi_14", "macd", "mom_20", "realized_vol_20"])
    logger.info(
        "feature_pipeline_complete",
        rows_before=n_before,
        rows_after=len(df),
        warmup_dropped=n_before - len(df),
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

LABEL_COLUMNS = [
    "fwd_ret_1", "fwd_ret_5", "fwd_ret_20",
    "label_1", "label_5", "label_20",
]