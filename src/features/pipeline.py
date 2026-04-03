"""
AlphaForge Feature Pipeline
Orchestrates feature computation for all assets and writes to TimescaleDB.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.config import get_settings
from src.data.storage import get_engine, load_ohlcv
from src.features.technical import (
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
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
    Run feature engineering for all assets.
    1. Load OHLCV from TimescaleDB
    2. Compute single-asset features
    3. Compute cross-sectional momentum rank (requires all assets)
    4. Write features back to TimescaleDB
    Returns: {asset: rows_written}
    """
    assets = assets or settings.all_assets
    timeframe = timeframe or settings.crypto_timeframe
    horizons = horizons or settings.prediction_horizons

    logger.info("starting_feature_pipeline", assets=assets, timeframe=timeframe)

    # ── Step 1: Load all OHLCV ────────────────────────────────────────────────
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
        raise RuntimeError("No OHLCV data available. Run ingestion first.")

    # ── Step 2: Single-asset features ─────────────────────────────────────────
    featured_data: dict[str, pd.DataFrame] = {}
    for asset, df in raw_data.items():
        try:
            feat_df = compute_all_features(df, horizons=horizons)
            featured_data[asset] = feat_df
        except Exception as e:
            logger.error("feature_computation_failed", asset=asset, error=str(e))

    # ── Step 3: Cross-sectional momentum rank ─────────────────────────────────
    # This requires all assets to be computed simultaneously
    featured_data = compute_cross_sectional_momentum_rank(featured_data, horizon=20)

    # ── Step 4: Write to TimescaleDB ──────────────────────────────────────────
    results: dict[str, int] = {}
    for asset, df in featured_data.items():
        try:
            n = _write_features(df, asset=asset, timeframe=timeframe)
            results[asset] = n
        except Exception as e:
            logger.error("write_features_failed", asset=asset, error=str(e))
            results[asset] = -1

    logger.info("feature_pipeline_complete", results=results)
    return results


def _write_features(df: pd.DataFrame, asset: str, timeframe: str) -> int:
    """Upsert feature rows into TimescaleDB."""
    df = df.reset_index()
    df["asset"] = asset
    df["timeframe"] = timeframe

    # Select only columns that exist in our schema
    all_cols = ["time", "asset", "timeframe"] + FEATURE_COLUMNS + LABEL_COLUMNS
    existing_cols = [c for c in all_cols if c in df.columns]
    df = df[existing_cols]

    # Build upsert SQL dynamically based on available columns
    feature_cols = [c for c in existing_cols if c not in ("time", "asset", "timeframe")]
    col_list = ", ".join(existing_cols)
    placeholder_list = ", ".join(f":{c}" for c in existing_cols)
    update_list = ", ".join(f"{c} = EXCLUDED.{c}" for c in feature_cols)

    upsert_sql = text(f"""
        INSERT INTO features ({col_list})
        VALUES ({placeholder_list})
        ON CONFLICT (time, asset, timeframe) DO UPDATE SET
            {update_list}
    """)

    rows = df.to_dict("records")
    with get_engine().begin() as conn:
        conn.execute(upsert_sql, rows)

    logger.info("wrote_features", asset=asset, rows=len(rows))
    return len(rows)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--assets", default=None, help="Comma-separated assets")
    @click.option("--timeframe", default=None)
    @click.option("--run-all", is_flag=True, default=False)
    def main(assets: str | None, timeframe: str | None, run_all: bool) -> None:
        from src.logger import setup_logging
        setup_logging()

        asset_list = assets.split(",") if assets else None
        results = run_feature_pipeline(
            assets=asset_list,
            timeframe=timeframe,
        )
        for asset, n in results.items():
            status = "✅" if n > 0 else "❌"
            print(f"{status} {asset}: {n} feature rows")

    main()
