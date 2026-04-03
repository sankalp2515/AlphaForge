"""
AlphaForge — Signal Serving API
FastAPI application that serves ML trading signals.

Endpoints:
  GET  /health          → liveness + readiness check
  GET  /metrics         → Prometheus metrics
  POST /predict         → get signals for asset(s)
  GET  /signals/latest  → latest cached signals
  GET  /model/info      → current model version info
  GET  /backtest/latest → latest backtest summary
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

from src.config import get_settings
from src.data.storage import load_features
from src.features.technical import FEATURE_COLUMNS
from src.logger import get_logger, setup_logging
from src.serving.cache import SignalCache
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    SignalEnum,
)

setup_logging()
logger = get_logger(__name__)
settings = get_settings()


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

PREDICT_REQUESTS = Counter(
    "alphaforge_predict_requests_total",
    "Total prediction requests",
    ["asset", "status"],
)

PREDICT_LATENCY = Histogram(
    "alphaforge_predict_latency_seconds",
    "Prediction request latency",
    ["asset"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

SIGNAL_GAUGE = Gauge(
    "alphaforge_latest_signal",
    "Latest trading signal (-1=short, 0=flat, 1=long)",
    ["asset", "horizon"],
)

PROB_UP_GAUGE = Gauge(
    "alphaforge_prob_up",
    "Latest probability of upward price move",
    ["asset", "horizon"],
)

MODEL_VERSION_GAUGE = Gauge(
    "alphaforge_model_version_info",
    "Current production model version",
    ["model_name", "version"],
)

CACHE_HIT_COUNTER = Counter(
    "alphaforge_cache_hits_total",
    "Redis cache hits",
    ["asset"],
)

CACHE_MISS_COUNTER = Counter(
    "alphaforge_cache_misses_total",
    "Redis cache misses",
    ["asset"],
)


# ─── App State ────────────────────────────────────────────────────────────────

class AppState:
    model: Any = None
    model_version: str = "unknown"
    cache: SignalCache | None = None
    feature_cols: list[str] = []


state = AppState()


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialize connections on startup."""
    logger.info("api_starting_up")

    # Initialize Redis cache
    try:
        state.cache = SignalCache(settings.redis_url)
        logger.info("redis_connected")
    except Exception as e:
        logger.warning("redis_unavailable", error=str(e))
        state.cache = None

    # Load production model from MLflow registry
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        model_uri = f"models:/{settings.mlflow_model_name}-lstm/Production"
        state.model = mlflow.pytorch.load_model(model_uri)
        state.model.eval()
        logger.info("model_loaded", uri=model_uri)

        # Get model version info
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            f"{settings.mlflow_model_name}-lstm", stages=["Production"]
        )
        if versions:
            state.model_version = versions[0].version
            MODEL_VERSION_GAUGE.labels(
                model_name=f"{settings.mlflow_model_name}-lstm",
                version=state.model_version,
            ).set(1)

    except Exception as e:
        logger.warning("model_load_failed", error=str(e))
        logger.info("running_without_model_mock_mode")

    # Determine available feature columns
    state.feature_cols = FEATURE_COLUMNS

    logger.info("api_ready", model_version=state.model_version)
    yield

    # Shutdown
    logger.info("api_shutting_down")
    if state.cache:
        state.cache.close()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="AlphaForge Signal API",
    description="Production ML trading signal engine. Serves directional predictions for crypto and equity assets.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware: Request Logging ──────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Liveness + readiness probe."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        cache_connected=state.cache is not None and state.cache.is_healthy(),
        environment=settings.environment,
    )


@app.get("/metrics", tags=["System"])
async def prometheus_metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Signals"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Generate trading signal for a single asset.

    Returns probability of upward price move + signal direction.
    Signals: 1=LONG, -1=SHORT, 0=FLAT
    """
    asset = request.asset
    horizon = request.horizon

    # Check cache first
    if state.cache:
        cached = state.cache.get(asset, horizon)
        if cached:
            CACHE_HIT_COUNTER.labels(asset=asset).inc()
            return PredictResponse(**cached)
        CACHE_MISS_COUNTER.labels(asset=asset).inc()

    start_time = time.time()

    try:
        # Load recent features for this asset
        features_df = load_features(
            asset=asset,
            timeframe=settings.crypto_timeframe if "/" in asset else settings.equity_timeframe,
        )

        if features_df.empty:
            raise HTTPException(status_code=404, detail=f"No feature data for {asset}")

        # Get last seq_len rows
        seq_len = settings.feature_window
        available_cols = [c for c in state.feature_cols if c in features_df.columns]
        X = features_df[available_cols].tail(seq_len).values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        X = np.clip(X, -10, 10)

        if len(X) < seq_len:
            # Pad with zeros if insufficient history
            pad = np.zeros((seq_len - len(X), X.shape[1]), dtype=np.float32)
            X = np.vstack([pad, X])

        # Run model inference
        if state.model is not None:
            with torch.no_grad():
                x_tensor = torch.FloatTensor(X).unsqueeze(0)  # (1, seq, features)
                logits = state.model(x_tensor)
                prob_up = float(torch.sigmoid(logits).squeeze().item())
        else:
            # Mock mode: return random signal for demo
            prob_up = float(np.random.beta(2, 2))

        # Determine signal direction
        if prob_up > 0.55:
            signal = SignalEnum.LONG
        elif prob_up < 0.45:
            signal = SignalEnum.SHORT
        else:
            signal = SignalEnum.FLAT

        confidence = abs(prob_up - 0.5) * 2  # 0-1 scale
        latency_ms = (time.time() - start_time) * 1000

        result = PredictResponse(
            asset=asset,
            horizon=horizon,
            prob_up=prob_up,
            signal=signal,
            confidence=round(confidence, 4),
            model_version=state.model_version,
            latency_ms=round(latency_ms, 2),
        )

        # Update Prometheus gauges
        PREDICT_REQUESTS.labels(asset=asset, status="success").inc()
        PREDICT_LATENCY.labels(asset=asset).observe(latency_ms / 1000)
        SIGNAL_GAUGE.labels(asset=asset, horizon=str(horizon)).set(signal.value)
        PROB_UP_GAUGE.labels(asset=asset, horizon=str(horizon)).set(prob_up)

        # Cache result
        if state.cache:
            state.cache.set(asset, horizon, result.model_dump())

        # Persist prediction to DB
        _persist_prediction(result, latency_ms)

        return result

    except HTTPException:
        PREDICT_REQUESTS.labels(asset=asset, status="not_found").inc()
        raise
    except Exception as e:
        PREDICT_REQUESTS.labels(asset=asset, status="error").inc()
        logger.error("predict_failed", asset=asset, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Signals"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Generate signals for multiple assets in one request."""
    results = []
    errors = []

    for asset in request.assets:
        try:
            single_req = PredictRequest(asset=asset, horizon=request.horizon)
            result = await predict(single_req)
            results.append(result)
        except Exception as e:
            errors.append({"asset": asset, "error": str(e)})

    return BatchPredictResponse(
        predictions=results,
        errors=errors,
        model_version=state.model_version,
    )


@app.get("/signals/latest", tags=["Signals"])
async def get_latest_signals() -> dict:
    """Return latest cached signals for all configured assets."""
    signals = {}
    for asset in settings.all_assets:
        for horizon in settings.prediction_horizons:
            if state.cache:
                cached = state.cache.get(asset, horizon)
                if cached:
                    signals[f"{asset}_{horizon}"] = cached
    return {"signals": signals, "model_version": state.model_version}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Current model version information."""
    return ModelInfoResponse(
        model_name=f"{settings.mlflow_model_name}-lstm",
        version=state.model_version,
        stage="Production",
        mlflow_uri=settings.mlflow_tracking_uri,
        assets=settings.all_assets,
        horizons=settings.prediction_horizons,
    )


@app.get("/backtest/latest", tags=["Evaluation"])
async def latest_backtest() -> dict:
    """Return latest backtest results from database."""
    from sqlalchemy import text
    from src.data.storage import get_engine

    sql = text("""
        SELECT * FROM backtest_results
        ORDER BY created_at DESC
        LIMIT 5
    """)

    try:
        with get_engine().connect() as conn:
            rows = conn.execute(sql).mappings().all()
        return {"results": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _persist_prediction(pred: "PredictResponse", latency_ms: float) -> None:
    """Write prediction to TimescaleDB (async-safe fire-and-forget)."""
    from datetime import datetime
    from sqlalchemy import text
    from src.data.storage import get_engine

    try:
        row = {
            "time": datetime.utcnow().isoformat(),
            "asset": pred.asset,
            "horizon": pred.horizon,
            "model_version": pred.model_version,
            "prob_up": pred.prob_up,
            "signal": pred.signal.value,
            "confidence": pred.confidence,
            "latency_ms": latency_ms,
        }
        sql = text("""
            INSERT INTO predictions
                (time, asset, horizon, model_version, prob_up, signal, confidence, latency_ms)
            VALUES
                (:time, :asset, :horizon, :model_version, :prob_up, :signal, :confidence, :latency_ms)
            ON CONFLICT (time, asset, horizon) DO NOTHING
        """)
        with get_engine().begin() as conn:
            conn.execute(sql, row)
    except Exception as e:
        logger.warning("persist_prediction_failed", error=str(e))
