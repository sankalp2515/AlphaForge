"""
AlphaForge Signal Cache
Uses Redis when REDIS_URL is configured, otherwise falls back to
an in-memory dict — no Redis install needed for local dev.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── In-Memory Cache ───────────────────────────────────────────────────────────

class InMemoryCache:
    """TTL-based in-memory cache. Used when Redis is not configured."""

    def __init__(self, ttl: int = 60) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        self._store[key] = (value, time.time() + (ttl or self.ttl))
        return True

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def ping(self) -> bool:
        return True


# ── Redis Cache ───────────────────────────────────────────────────────────────

class RedisCache:
    """Redis-backed cache. Only instantiated when REDIS_URL is set."""

    def __init__(self, redis_url: str, ttl: int = 60) -> None:
        import redis as redis_lib
        self._client = redis_lib.from_url(redis_url, decode_responses=True)
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        try:
            self._client.setex(key, ttl or self.ttl, value)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


# ── Unified Signal Cache ──────────────────────────────────────────────────────

class SignalCache:
    """
    Public cache interface.
    Auto-selects Redis (if REDIS_URL set) or in-memory dict (local dev).
    Call with no arguments: cache = SignalCache()
    """

    def __init__(self) -> None:          # ← no arguments
        self.ttl = settings.signal_cache_ttl_seconds

        if settings.use_redis:
            try:
                backend = RedisCache(settings.redis_url, ttl=self.ttl)
                backend.ping()
                self._backend: InMemoryCache | RedisCache = backend
                logger.info("cache_backend", backend="redis")
            except Exception as e:
                logger.warning("redis_unavailable", error=str(e), fallback="in_memory")
                self._backend = InMemoryCache(ttl=self.ttl)
        else:
            self._backend = InMemoryCache(ttl=self.ttl)
            logger.info("cache_backend", backend="in_memory")

    def _key(self, asset: str, horizon: int) -> str:
        return f"alphaforge:signal:{asset.replace('/', '_')}:{horizon}"

    def get(self, asset: str, horizon: int) -> Optional[dict]:
        raw = self._backend.get(self._key(asset, horizon))
        return json.loads(raw) if raw else None

    def set(self, asset: str, horizon: int, data: dict) -> bool:
        return self._backend.set(
            self._key(asset, horizon),
            json.dumps(data, default=str),
        )

    def is_healthy(self) -> bool:
        return self._backend.ping()

    def close(self) -> None:
        pass