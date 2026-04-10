"""
AlphaForge Feature Pipeline
Orchestrates feature computation for all assets and writes to SQLite.
"""
from __future__ import annotations

import pandas as pd

from src.config import get_settings
from src.data.storage import load_ohlcv, upsert_features
from src.features.technical import (
    compute_all_features,
    compute_cross_sectional_momentum_rank,
)
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def run_feature_pipeline(
    assets: list[str] | None = None,
    timeframe: str | None = None,
    horizons: list[int] | None = None,
) -> dict[str, int]:
    """
    Full feature engineering pipeline:
      1. Load OHLCV from DB
      2. Compute single-asset features (RSI, MACD, momentum, vol, microstructure)
      3. Compute cross-sectional momentum rank across all assets
      4. Write features back to DB via upsert_features()
    Returns: {asset: rows_written}
    """
    assets    = assets    or settings.all_assets
    timeframe = timeframe or settings.crypto_timeframe
    horizons  = horizons  or settings.prediction_horizons

    logger.info("starting_feature_pipeline", assets=assets, timeframe=timeframe)

    # ── Step 1: Load OHLCV ────────────────────────────────────────────────────
    raw_data: dict[str, pd.DataFrame] = {}
    for asset in assets:
        try:
            df = load_ohlcv(asset, timeframe)
            if df.empty:
                logger.warning("no_ohlcv_data", asset=asset)
                continue
            raw_data[asset] = df
            logger.info("loaded_ohlcv", asset=asset, rows=len(df))
        except Exception as e:
            logger.error("load_ohlcv_failed", asset=asset, error=str(e))

    if not raw_data:
        raise RuntimeError("No OHLCV data found. Run 'setup ingest' first.")

    # ── Step 2: Per-asset feature computation ─────────────────────────────────
    featured: dict[str, pd.DataFrame] = {}
    for asset, df in raw_data.items():
        try:
            featured[asset] = compute_all_features(df, horizons=horizons)
        except Exception as e:
            logger.error("feature_computation_failed", asset=asset, error=str(e))

    if not featured:
        raise RuntimeError("Feature computation failed for all assets.")

    # ── Step 3: Cross-sectional momentum rank ─────────────────────────────────
    featured = compute_cross_sectional_momentum_rank(featured, horizon=20)

    # ── Step 4: Write to DB ───────────────────────────────────────────────────
    results: dict[str, int] = {}
    for asset, df in featured.items():
        try:
            # upsert_features handles: reset_index, astype(str) on time,
            # NaN→None conversion, and INSERT OR REPLACE — all SQLite safe.
            n = upsert_features(df, asset=asset, timeframe=timeframe)
            results[asset] = n
        except Exception as e:
            logger.error("write_features_failed", asset=asset, error=str(e))
            results[asset] = -1

    logger.info("feature_pipeline_complete", results=results)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.logger import setup_logging
    setup_logging()

    results = run_feature_pipeline()
    print()
    for asset, n in results.items():
        status = "✅" if n > 0 else "❌"
        print(f"  {status}  {asset}: {n} feature rows")