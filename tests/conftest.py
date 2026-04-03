"""
AlphaForge Test Configuration
Shared fixtures and test settings.
"""
import os
import pytest

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("ALPHAFORGE_DB_URL", "postgresql://alphaforge:alphaforge@localhost:5433/alphaforge")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///test_mlflow.db")


@pytest.fixture(scope="session")
def settings():
    from src.config import get_settings
    return get_settings()


@pytest.fixture(scope="session")
def db_engine(settings):
    try:
        from src.data.storage import get_engine
        engine = get_engine()
        with engine.connect():
            pass
        return engine
    except Exception:
        pytest.skip("Database not available")


@pytest.fixture
def redis_client(settings):
    try:
        import redis
        client = redis.from_url(settings.redis_url)
        client.ping()
        yield client
        client.flushdb()
    except Exception:
        pytest.skip("Redis not available")
