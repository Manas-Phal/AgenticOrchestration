"""Resilience utilities: Exponential backoff retries, timeout management, and latency tracking."""

import functools
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app.config import get_settings
from app.core.exceptions import TransientError
from app.logging_config import get_logger, sanitize_value

logger = get_logger("app.core.resilience")

F = TypeVar("F", bound=Callable[..., Any])


def retry_with_exponential_backoff(
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    jitter: bool = True,
    retry_exceptions: tuple[type[Exception], ...] = (TransientError, TimeoutError),
) -> Callable[[F], F]:
    """Decorator performing exponential backoff retries on transient errors."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            retries_limit = max_retries if max_retries is not None else settings.max_retries
            delay_base = base_delay if base_delay is not None else settings.retry_base_delay
            delay_max = max_delay if max_delay is not None else settings.retry_max_delay

            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as err:
                    if attempt > retries_limit:
                        logger.error(
                            "Max retries exceeded during execution",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_retries": retries_limit,
                                "error": sanitize_value(str(err)),
                            },
                        )
                        raise

                    # Compute backoff with optional jitter
                    calculated_delay = min(delay_max, delay_base * (2 ** (attempt - 1)))
                    if jitter:
                        calculated_delay = calculated_delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        "Transient failure encountered, retrying with exponential backoff",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "delay_seconds": round(calculated_delay, 3),
                            "error": sanitize_value(str(err)),
                        },
                    )
                    time.sleep(calculated_delay)

        return wrapper  # type: ignore[return-value]

    return decorator


def measure_latency(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Execute callable and return result along with latency measured in milliseconds."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return result, elapsed_ms
