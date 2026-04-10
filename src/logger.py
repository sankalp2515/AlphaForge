"""
AlphaForge — Structured Logging
Uses structlog with JSON output for production, pretty console for dev.

Fix: removed stdlib.add_logger_name processor which caused
     "PrintLogger has no attribute 'name'" when structlog uses
     PrintLoggerFactory instead of stdlib logging.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from src.config import get_settings


def setup_logging() -> None:
    """Configure structlog. Call once at application startup."""
    settings = get_settings()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        # NOTE: add_logger_name is intentionally excluded — it requires
        # stdlib Logger objects and crashes with PrintLoggerFactory.
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Mirror to stdlib so third-party libs (uvicorn, airflow) stay in sync
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        level=settings.log_level.upper(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named bound logger. Use this everywhere in the codebase."""
    return structlog.get_logger(name)
