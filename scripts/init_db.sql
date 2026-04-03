-- AlphaForge TimescaleDB Schema
-- Initializes hypertables for time-series market data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ─── MLflow database ───
CREATE DATABASE mlflow;

-- ─── Raw OHLCV Data ───
CREATE TABLE IF NOT EXISTS ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    asset       TEXT            NOT NULL,
    source      TEXT            NOT NULL,   -- 'binance' | 'yfinance'
    timeframe   TEXT            NOT NULL,   -- '1h' | '1d'
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    created_at  TIMESTAMPTZ     DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable (partitioned by time)
SELECT create_hypertable(
    'ohlcv', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '7 days'
);

-- Indexes
CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique
    ON ohlcv (time, asset, timeframe);

CREATE INDEX IF NOT EXISTS idx_ohlcv_asset_time
    ON ohlcv (asset, time DESC);

-- ─── Engineered Features ───
CREATE TABLE IF NOT EXISTS features (
    time        TIMESTAMPTZ     NOT NULL,
    asset       TEXT            NOT NULL,
    timeframe   TEXT            NOT NULL,

    -- Technical features
    rsi_14      DOUBLE PRECISION,
    macd        DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist   DOUBLE PRECISION,
    bb_upper    DOUBLE PRECISION,
    bb_lower    DOUBLE PRECISION,
    bb_pct      DOUBLE PRECISION,
    atr_14      DOUBLE PRECISION,
    obv         DOUBLE PRECISION,
    adx_14      DOUBLE PRECISION,

    -- Momentum features
    mom_1       DOUBLE PRECISION,  -- 1-bar return
    mom_5       DOUBLE PRECISION,  -- 5-bar return
    mom_20      DOUBLE PRECISION,  -- 20-bar return
    mom_rank    DOUBLE PRECISION,  -- cross-sectional momentum rank

    -- Volatility features
    realized_vol_5  DOUBLE PRECISION,
    realized_vol_20 DOUBLE PRECISION,
    vol_ratio       DOUBLE PRECISION,  -- short/long vol ratio

    -- Microstructure features
    ofi             DOUBLE PRECISION,  -- order flow imbalance proxy
    amihud          DOUBLE PRECISION,  -- Amihud illiquidity

    -- Labels (forward returns)
    fwd_ret_1   DOUBLE PRECISION,  -- 1-bar forward return
    fwd_ret_5   DOUBLE PRECISION,  -- 5-bar forward return
    fwd_ret_20  DOUBLE PRECISION,  -- 20-bar forward return
    label_1     SMALLINT,          -- 1 if fwd_ret_1 > 0 else 0
    label_5     SMALLINT,
    label_20    SMALLINT,

    created_at  TIMESTAMPTZ     DEFAULT NOW()
);

SELECT create_hypertable(
    'features', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_features_unique
    ON features (time, asset, timeframe);

CREATE INDEX IF NOT EXISTS idx_features_asset_time
    ON features (asset, time DESC);

-- ─── Model Predictions ───
CREATE TABLE IF NOT EXISTS predictions (
    time            TIMESTAMPTZ     NOT NULL,
    asset           TEXT            NOT NULL,
    model_version   TEXT            NOT NULL,
    horizon         INTEGER         NOT NULL,  -- bars ahead
    prob_up         DOUBLE PRECISION NOT NULL, -- probability of upward move
    signal          SMALLINT        NOT NULL,  -- 1=long, -1=short, 0=flat
    confidence      DOUBLE PRECISION NOT NULL,
    latency_ms      DOUBLE PRECISION,
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

SELECT create_hypertable(
    'predictions', 'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '30 days'
);

-- ─── Backtest Results ───
CREATE TABLE IF NOT EXISTS backtest_results (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT            NOT NULL,  -- MLflow run ID
    model_version   TEXT            NOT NULL,
    strategy        TEXT            NOT NULL,
    start_date      DATE            NOT NULL,
    end_date        DATE            NOT NULL,

    -- Financial metrics
    total_return    DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    sortino_ratio   DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    calmar_ratio    DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    total_trades    INTEGER,
    turnover        DOUBLE PRECISION,

    -- ML metrics
    auc_roc         DOUBLE PRECISION,
    f1_score        DOUBLE PRECISION,
    accuracy        DOUBLE PRECISION,
    information_coeff DOUBLE PRECISION,

    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- ─── Drift Reports ───
CREATE TABLE IF NOT EXISTS drift_reports (
    id              SERIAL PRIMARY KEY,
    report_date     DATE            NOT NULL,
    asset           TEXT,                      -- NULL = portfolio-level
    psi_score       DOUBLE PRECISION,
    drift_detected  BOOLEAN         NOT NULL,
    affected_features TEXT[],
    report_path     TEXT,                      -- path to Evidently HTML report
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- ─── Useful Views ───
CREATE OR REPLACE VIEW latest_signals AS
    SELECT DISTINCT ON (asset, horizon)
        time, asset, horizon, prob_up, signal, confidence, model_version
    FROM predictions
    ORDER BY asset, horizon, time DESC;

CREATE OR REPLACE VIEW feature_stats AS
    SELECT
        asset,
        COUNT(*) as n_rows,
        MIN(time) as earliest,
        MAX(time) as latest,
        AVG(rsi_14) as avg_rsi,
        STDDEV(mom_1) as vol_1bar
    FROM features
    GROUP BY asset;

COMMENT ON TABLE ohlcv IS 'Raw OHLCV market data — TimescaleDB hypertable';
COMMENT ON TABLE features IS 'Engineered alpha features — TimescaleDB hypertable';
COMMENT ON TABLE predictions IS 'Model predictions and trading signals';
COMMENT ON TABLE backtest_results IS 'Backtest performance results per model version';
