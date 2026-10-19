"""Smoke tests for foundation verification."""

import pytest

from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from app.main import main


def test_settings_loading() -> None:
    """Test that application configuration loads defaults properly."""
    settings = get_settings()
    assert settings.app_name == "Agentic Orchestration Engine"
    assert settings.environment in ["development", "staging", "production", "test"]
    assert isinstance(settings.debug, bool)


def test_logging_setup() -> None:
    """Test structured logging setup without raising exceptions."""
    setup_logging(log_level="DEBUG", json_format=True)
    logger = get_logger("tests.smoke")
    logger.debug("Smoke test debug log message", extra={"test_key": "test_value"})


@pytest.mark.asyncio
async def test_main_execution() -> None:
    """Test that main async entry point executes cleanly."""
    await main()
