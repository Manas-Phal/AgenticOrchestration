"""Application Configuration Module using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading and validation."""

    app_name: str = "Agentic Orchestration Engine"
    environment: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = True

    # Workflow Orchestration defaults
    workflow_max_iterations: int = 3
    workflow_quality_threshold: float = 0.8

    # Resilience & Offline Execution Settings
    mock_mode: bool = True
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 5.0
    request_timeout_seconds: float = 30.0

    # API Key placeholders (optional when mock_mode=True)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton instance."""
    return Settings()
