"""Structured logging configuration using structlog.

Provides consistent, JSON-formatted logging across all modules
with context binding and performance tracking.
"""

import logging
import sys
from functools import lru_cache
from typing import Any

import structlog
from structlog.types import Processor

from vnull.core.config import settings


def _add_log_level(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def _add_timestamp(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add ISO timestamp to event dict."""
    from datetime import datetime, timezone

    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def configure_logging() -> None:
    """Configure structlog with appropriate processors."""
    # Determine if we're in a TTY (interactive) or not (production/CI)
    is_tty = sys.stderr.isatty()

    # Shared processors for all outputs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if is_tty:
        # Pretty console output for development
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
    else:
        # JSON output for production
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging for third-party libs
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, settings.log_level),
        stream=sys.stderr,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


@lru_cache(maxsize=128)
def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured structlog BoundLogger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started", url="https://example.com")
    """
    # Ensure logging is configured on first call
    if not structlog.is_configured():
        configure_logging()

    logger = structlog.get_logger(name or "vnull")
    return logger


class LogContext:
    """Context manager for adding temporary context to logs.

    Example:
        >>> with LogContext(request_id="abc123", user="john"):
        ...     logger.info("Processing request")  # includes request_id and user
        >>> logger.info("Done")  # no longer includes context
    """

    def __init__(self, **kwargs: Any) -> None:
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_performance(operation: str):
    """Decorator to log function execution time.

    Args:
        operation: Name of the operation being timed.

    Example:
        >>> @log_performance("html_parsing")
        ... def parse_html(content: str) -> dict:
        ...     ...
    """
    import functools
    import time

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug(
                    "Operation completed",
                    operation=operation,
                    elapsed_ms=round(elapsed * 1000, 2),
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    "Operation failed",
                    operation=operation,
                    elapsed_ms=round(elapsed * 1000, 2),
                    error=str(e),
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug(
                    "Operation completed",
                    operation=operation,
                    elapsed_ms=round(elapsed * 1000, 2),
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error(
                    "Operation failed",
                    operation=operation,
                    elapsed_ms=round(elapsed * 1000, 2),
                    error=str(e),
                )
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
