"""Interactive demonstration script for Phase 1: Pydantic + PydanticAI Prototype.

Demonstrates all 12 key capabilities:
1. Pydantic BaseModel
2. Field constraints
3. Nested models
4. Enums / Literals
5. Serialization (model_dump / model_dump_json)
6. Validation errors handling
7. PydanticAI Agent instantiation
8. Structured output generation
9. Dependency injection
10. Tool usage
11. Output validation
12. Retry & recovery behavior (ModelRetry)
"""

import asyncio

from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from app.agents.research_agent import (
    MockSearchClient,
    ResearchDependencies,
    create_research_agent,
)
from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from app.schemas.research import (
    ResearchResult,
    ResearchSource,
    ResearchSummary,
    SourceType,
)

logger = get_logger("demo_phase1")


def demonstrate_pydantic_validation() -> None:
    """Demonstrate items 1-6: Models, Field Constraints, Serialization, Errors."""
    logger.info("=== DEMO 1: Pydantic Schema Validation & Serialization ===")

    # 1-4. Create valid nested model instance with enums and field constraints
    source1 = ResearchSource(
        url="https://ai.pydantic.dev/",
        title="PydanticAI Technical Specification",
        source_type=SourceType.DOCUMENTATION,
        reliability_score=0.95,
    )
    summary = ResearchSummary(
        topic="Agentic Orchestration",
        title="Production Agent Architecture",
        summary="Combining LangGraph, PydanticAI, and DSPy into decoupled layers.",
        category="ai_architecture",
        key_findings=[
            "LangGraph handles graph control flow and persistence.",
            "PydanticAI enforces strict output type validation.",
            "DSPy compiles and optimizes LM prompt signatures.",
        ],
        confidence=0.90,
        sources=[source1],
    )
    result = ResearchResult(
        query="Explain agentic orchestration architecture",
        summary=summary,
        execution_time_ms=145.2,
        status="success",
    )

    logger.info("Successfully created typed ResearchResult object.")

    # 5. Serialization demonstration
    json_output = result.model_dump_json(indent=2)
    logger.info("Serialized JSON Output:\n%s", json_output)

    parsed_back = ResearchResult.model_validate_json(json_output)
    assert parsed_back.summary.confidence == 0.90
    logger.info("Successfully deserialized back into validated Python object.")

    # 6. Validation error demonstration (invalid confidence > 1.0)
    logger.info("Testing invalid confidence score (1.5 > 1.0)...")
    try:
        ResearchSummary(
            topic="Test",
            title="Invalid Summary",
            summary="This summary should fail validation.",
            category="ai_architecture",
            key_findings=["Finding 1"],
            confidence=1.5,  # VIOLATION
            sources=[source1],
        )
    except ValidationError as e:
        logger.info("Caught expected ValidationError:\n%s", str(e))


async def demonstrate_pydantic_ai_agent() -> None:
    """Demonstrate items 7-12: PydanticAI Agent, Output Validation, Retry."""
    logger.info("=== DEMO 2: PydanticAI Agent with Dependency Injection & Tools ===")

    settings = get_settings()

    # Determine whether to use real API key or TestModel
    if settings.openai_api_key and settings.openai_api_key != "your-openai-api-key-here":
        logger.info("Using real OpenAI API model...")
        agent = create_research_agent(model="openai:gpt-4o-mini")
    else:
        logger.info("No API key set. Using deterministic TestModel for execution...")
        mock_args = {
            "query": "What is PydanticAI?",
            "summary": {
                "topic": "PydanticAI Overview",
                "title": "Agentic Type Safety Framework",
                "summary": (
                    "PydanticAI brings structured output validation and tool injection to LLMs."
                ),
                "category": "ai_architecture",
                "key_findings": [
                    "Guarantees type-safe LLM outputs using Pydantic models.",
                    "Provides seamless dependency injection for tools.",
                ],
                "confidence": 0.92,
                "sources": [
                    {
                        "url": "https://ai.pydantic.dev/",
                        "title": "PydanticAI Documentation",
                        "source_type": "documentation",
                        "reliability_score": 0.98,
                    }
                ],
            },
            "execution_time_ms": 120.0,
            "status": "success",
        }
        test_model = TestModel(custom_output_args=mock_args)
        agent = create_research_agent(model=test_model)

    # 9. Dependency Injection container
    deps = ResearchDependencies(
        search_client=MockSearchClient(),
        min_confidence=0.7,
        max_sources=2,
    )

    # 7-11. Execute Agent Run
    user_query = "What is PydanticAI and why use it for structured outputs?"
    logger.info("Running agent with user query: '%s'", user_query)

    run_result = await agent.run(user_query, deps=deps)

    # 8. Structured output typed validation
    typed_result: ResearchResult = run_result.output
    logger.info("Agent execution completed successfully!")
    logger.info("Result Query: %s", typed_result.query)
    logger.info("Result Topic: %s", typed_result.summary.topic)
    logger.info("Result Confidence: %.2f", typed_result.summary.confidence)
    logger.info("Sources Count: %d", len(typed_result.summary.sources))


async def main() -> None:
    """Run all Phase 1 demonstrations."""
    setup_logging(log_level="INFO", json_format=False)
    demonstrate_pydantic_validation()
    await demonstrate_pydantic_ai_agent()


if __name__ == "__main__":
    asyncio.run(main())
