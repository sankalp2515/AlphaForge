"""
AlphaForge — Signal Cache
Redis-backed cache for trading signals.
TTL-based expiry ensures signals are refreshed on schedule.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import redis

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class SignalCache:
    """Redis-backed cache for trading signals with TTL."""

    def __init__(self, redis_url: str) -> None:
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = settings.signal_cache_ttl_seconds

    def _key(self, asset: str, horizon: int) -> str:
        clean = asset.replace("/", "_")
        return f"alphaforge:signal:{clean}:{horizon}"

    def get(self, asset: str, horizon: int) -> Optional[dict]:
        try:
            val = self.client.get(self._key(asset, horizon))
            return json.loads(val) if val else None
        except Exception as e:
            logger.warning("cache_get_failed", asset=asset, error=str(e))
            return None

    def set(self, asset: str, horizon: int, data: dict) -> bool:
        try:
            self.client.setex(
                self._key(asset, horizon),
                self.ttl,
                json.dumps(data, default=str),
            )
            return True
        except Exception as e:
            logger.warning("cache_set_failed", asset=asset, error=str(e))
            return False

    def delete(self, asset: str, horizon: int) -> None:
        try:
            self.client.delete(self._key(asset, horizon))
        except Exception:
            pass

    def flush_all_signals(self) -> int:
        """Clear all cached signals (call before model version change)."""
        try:
            keys = self.client.keys("alphaforge:signal:*")
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning("cache_flush_failed", error=str(e))
            return 0

    def is_healthy(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    def close(self) -> None:
        self.client.close()
