"""Test suite for system hardening, resilience, secret scrubbing, and health checks."""

import logging
import time

import pytest

from app.config import get_settings
from app.core.exceptions import (
    ConfigurationError,
    HumanRequiredError,
    OrchestrationBaseError,
    ToolError,
    TransientError,
    UnexpectedError,
    ValidationError,
)
from app.core.health import check_system_health
from app.core.resilience import measure_latency, retry_with_exponential_backoff
from app.logging_config import JSONFormatter, sanitize_value
from app.orchestration.runner import OrchestrationEngine


def test_exception_taxonomy_serialization():
    """Verify error taxonomy exception hierarchy and to_dict serialization."""
    err = TransientError("Network timeout connecting to LLM service", details={"retry_after": 5})
    assert isinstance(err, OrchestrationBaseError)
    payload = err.to_dict()
    assert payload["error_type"] == "TransientError"
    assert payload["message"] == "Network timeout connecting to LLM service"
    assert payload["details"]["retry_after"] == 5

    val_err = ValidationError("Field 'quality_score' out of bounds", details={"val": 1.5})
    assert val_err.to_dict()["error_type"] == "ValidationError"

    tool_err = ToolError("External search API unavailable")
    assert tool_err.to_dict()["error_type"] == "ToolError"

    human_err = HumanRequiredError("Iteration limit reached")
    assert human_err.to_dict()["error_type"] == "HumanRequiredError"

    cfg_err = ConfigurationError("Invalid environment variable")
    assert cfg_err.to_dict()["error_type"] == "ConfigurationError"

    unexp_err = UnexpectedError("Uncaught runtime failure")
    assert unexp_err.to_dict()["error_type"] == "UnexpectedError"


def test_secret_sanitization():
    """Verify that sensitive API keys and tokens are scrubbed from log output and dicts."""
    raw_text = "Connecting with API key sk-proj-1234567890abcdef and Bearer token_secret_xyz123"
    clean_text = sanitize_value(raw_text)
    assert "sk-proj-1234567890abcdef" not in clean_text
    assert "[REDACTED_API_KEY]" in clean_text
    assert "Bearer [REDACTED_TOKEN]" in clean_text

    raw_dict = {
        "user": "admin",
        "openai_api_key": "sk-secret-key-abcdef123456",
        "nested": {"token": "my-secret-auth-token"},
    }
    clean_dict = sanitize_value(raw_dict)
    assert clean_dict["openai_api_key"] == "[REDACTED]"
    assert clean_dict["nested"]["token"] == "[REDACTED]"


def test_json_formatter_logging(caplog):
    """Verify JSONFormatter formats logs cleanly without leaking secrets."""
    caplog.set_level(logging.INFO)
    formatter = JSONFormatter()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="API auth using sk-1234567890abcdefgh",
        args=(),
        exc_info=None,
    )
    record.thread_id = "thread-123"

    formatted_json = formatter.format(record)
    assert '"level": "INFO"' in formatted_json
    assert "sk-1234567890" not in formatted_json
    assert "[REDACTED_API_KEY]" in formatted_json


def test_exponential_backoff_retry_success():
    """Verify retry_with_exponential_backoff retries transient failures and eventually succeeds."""
    attempts = 0

    @retry_with_exponential_backoff(max_retries=3, base_delay=0.01, jitter=False)
    def flaky_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientError("Temporary connection reset")
        return "SUCCESS"

    result = flaky_operation()
    assert result == "SUCCESS"
    assert attempts == 3


def test_exponential_backoff_retry_failure():
    """Verify retry_with_exponential_backoff raises exception when max retries are exceeded."""
    attempts = 0

    @retry_with_exponential_backoff(max_retries=2, base_delay=0.01, jitter=False)
    def failing_operation():
        nonlocal attempts
        attempts += 1
        raise TransientError("Persistent service failure")

    with pytest.raises(TransientError):
        failing_operation()

    assert attempts == 3  # 1 initial + 2 retries


def test_measure_latency_utility():
    """Verify latency measurement utility tracks execution duration in milliseconds."""
    def sample_work():
        time.sleep(0.02)
        return "done"

    result, duration_ms = measure_latency(sample_work)
    assert result == "done"
    assert duration_ms >= 15.0  # At least ~15-20ms elapsed


def test_health_check_mechanism():
    """Verify system health check mechanism returns a structured HealthCheckReport."""
    report = check_system_health()
    assert report.status in ("healthy", "degraded", "unhealthy")
    assert report.app_name == "Agentic Orchestration Engine"
    assert "config" in report.components
    assert "execution_mode" in report.components
    assert "validation_layer" in report.components
    assert "langgraph_engine" in report.components


def test_offline_deterministic_mock_mode():
    """Verify system executes completely offline when MOCK_MODE=True without requiring API keys."""
    settings = get_settings()
    assert settings.mock_mode is True

    engine = OrchestrationEngine()
    result = engine.start_request(
        user_query="instant_pass: Offline Deterministic Test Request",
        thread_id="thread-offline-101",
    )

    assert result["status"] == "completed"
    assert result["final_response"].status == "completed"
    assert "node_durations_ms" in result
    assert result["total_duration_ms"] > 0.0
