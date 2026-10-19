"""System Health Check Mechanism for Production Operations."""

import time
from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("app.core.health")


class ComponentHealth(BaseModel):
    """Status details for an individual system component."""

    status: Literal["healthy", "degraded", "unhealthy"]
    details: str


class HealthCheckReport(BaseModel):
    """Top-level health check diagnostic report model."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000.0)
    app_name: str
    environment: str
    mock_mode: bool
    components: dict[str, ComponentHealth]


def check_system_health() -> HealthCheckReport:
    """Perform comprehensive health checks across configuration, validation, and graph engine."""
    settings = get_settings()
    components: dict[str, ComponentHealth] = {}
    overall_status: Literal["healthy", "degraded", "unhealthy"] = "healthy"

    # 1. Configuration check
    try:
        components["config"] = ComponentHealth(
            status="healthy",
            details=f"Config loaded for environment '{settings.environment}'",
        )
    except Exception as err:
        components["config"] = ComponentHealth(
            status="unhealthy", details=f"Config error: {err}"
        )
        overall_status = "unhealthy"

    # 2. Execution Mode & Secrets check
    if settings.mock_mode:
        components["execution_mode"] = ComponentHealth(
            status="healthy",
            details="Running in deterministic offline mock mode (no API key required)",
        )
    else:
        if not settings.openai_api_key and not settings.anthropic_api_key:
            components["execution_mode"] = ComponentHealth(
                status="degraded",
                details="MOCK_MODE is disabled but no LLM API keys are configured",
            )
            if overall_status == "healthy":
                overall_status = "degraded"
        else:
            components["execution_mode"] = ComponentHealth(
                status="healthy", details="Live LLM API keys configured"
            )

    # 3. Validation Layer check
    try:
        from app.core.validation import validate_research_output  # noqa: F401

        components["validation_layer"] = ComponentHealth(
            status="healthy", details="PydanticAI validation boundaries ready"
        )
    except Exception as err:
        components["validation_layer"] = ComponentHealth(
            status="unhealthy", details=f"Validation layer failure: {err}"
        )
        overall_status = "unhealthy"

    # 4. LangGraph Engine check
    try:
        from app.orchestration.graph import create_orchestrator_graph

        _ = create_orchestrator_graph()
        components["langgraph_engine"] = ComponentHealth(
            status="healthy", details="LangGraph StateGraph compiled cleanly"
        )
    except Exception as err:
        components["langgraph_engine"] = ComponentHealth(
            status="unhealthy", details=f"LangGraph compilation error: {err}"
        )
        overall_status = "unhealthy"

    report = HealthCheckReport(
        status=overall_status,
        app_name=settings.app_name,
        environment=settings.environment,
        mock_mode=settings.mock_mode,
        components=components,
    )

    logger.info(
        "System health check executed",
        extra={"status": report.status, "mock_mode": report.mock_mode},
    )
    return report
