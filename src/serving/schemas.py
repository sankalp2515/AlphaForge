"""
AlphaForge — API Schemas
Pydantic v2 models for request/response validation.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SignalEnum(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1


# ─── Request Models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    asset: str = Field(
        ...,
        description="Asset symbol (e.g. 'BTC/USDT', 'SPY')",
        examples=["BTC/USDT", "ETH/USDT", "SPY"],
    )
    horizon: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Prediction horizon in bars (1, 5, or 20)",
    )

    @field_validator("asset")
    @classmethod
    def normalize_asset(cls, v: str) -> str:
        return v.upper().strip()


class BatchPredictRequest(BaseModel):
    assets: list[str] = Field(
        ...,
        description="List of asset symbols",
        min_length=1,
        max_length=20,
    )
    horizon: int = Field(default=1, ge=1, le=20)


# ─── Response Models ──────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    asset: str
    horizon: int
    prob_up: float = Field(..., ge=0.0, le=1.0,
                           description="Probability of upward price move")
    signal: SignalEnum = Field(..., description="Trading signal: 1=LONG, -1=SHORT, 0=FLAT")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Signal confidence (0=low, 1=high)")
    model_version: str
    latency_ms: float = Field(..., description="Inference latency in milliseconds")

    model_config = {"use_enum_values": False}


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    errors: list[dict]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    cache_connected: bool
    environment: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    stage: str
    mlflow_uri: str
    assets: list[str]
    horizons: list[int]
