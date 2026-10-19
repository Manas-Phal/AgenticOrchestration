"""Structured Logging Module for Production Operations with Secret-Safe Sanitization."""

import json
import logging
import re
import sys
from typing import Any

SECRET_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9_-]{12,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)bearer\s+[a-zA-Z0-9._-]+"), "Bearer [REDACTED_TOKEN]"),
    (
        re.compile(r"(?i)(api[-_]?key|password|secret|auth|token)\s*=\s*['\"]?[^\s'\"]+['\"]?"),
        r"\1=[REDACTED]",
    ),
]

SENSITIVE_KEYS = {
    "api_key",
    "openai_api_key",
    "anthropic_api_key",
    "secret",
    "password",
    "token",
    "authorization",
}


def sanitize_value(val: Any) -> Any:
    """Recursively scrub sensitive keys and secret regex patterns from strings, dicts, and lists."""
    if isinstance(val, str):
        sanitized = val
        for pattern, replacement in SECRET_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized
    elif isinstance(val, dict):
        cleaned_dict: dict[str, Any] = {}
        for k, v in val.items():
            if str(k).lower() in SENSITIVE_KEYS:
                cleaned_dict[str(k)] = "[REDACTED]"
            else:
                cleaned_dict[str(k)] = sanitize_value(v)
        return cleaned_dict
    elif isinstance(val, list):
        return [sanitize_value(item) for item in val]
    return val


class JSONFormatter(logging.Formatter):
    """Custom JSON Formatter for structured logging output with secret scrubbing."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as a structured JSON object."""
        msg_str = sanitize_value(record.getMessage())
        exc_str = sanitize_value(self.formatException(record.exc_info)) if record.exc_info else None

        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": msg_str,
            "module": record.module,
            "line": record.lineno,
        }

        if exc_str:
            log_data["exception"] = exc_str

        # Include custom extra fields if passed via extra parameter
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
        }
        extra_fields = {
            key: sanitize_value(val)
            for key, val in record.__dict__.items()
            if key not in standard_attrs and not key.startswith("_")
        }
        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure system-wide root logger with structured formatting and secret scrubbing."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicated logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)

    if json_format:
        stream_handler.setFormatter(JSONFormatter())
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Get named logger instance."""
    return logging.getLogger(name)
