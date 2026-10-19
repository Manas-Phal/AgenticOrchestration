"""Application entry point for Production-Grade Agentic Orchestration."""

import asyncio

from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from scripts.demo_phase1 import demonstrate_pydantic_ai_agent, demonstrate_pydantic_validation

logger = get_logger("app.main")


async def main() -> None:
    """Async main application entry point."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level, json_format=settings.log_json)

    logger.info(
        "Phase 1 Pydantic + PydanticAI Agent Prototype initialized",
        extra={
            "app_name": settings.app_name,
            "environment": settings.environment,
            "debug": settings.debug,
            "log_level": settings.log_level,
        },
    )

    logger.info(
        "Architectural Boundaries Enforced: LangGraph (Workflow Control) | "
        "PydanticAI (Type Safety & Output Validation) | DSPy (Model Optimization)"
    )

    # Execute Phase 1 Schema & Agent Demonstrations
    demonstrate_pydantic_validation()
    await demonstrate_pydantic_ai_agent()

    # Execute Phase 2 LangGraph Workflow Demonstration
    from scripts.demo_phase2 import run_phase2_demonstration
    run_phase2_demonstration()

    # Execute Phase 3 DSPy Optimization Demonstration
    from scripts.demo_phase3 import run_phase3_demonstration
    run_phase3_demonstration()


if __name__ == "__main__":
    asyncio.run(main())
