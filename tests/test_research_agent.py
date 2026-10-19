"""Integration tests for PydanticAI Research Agent, Deps, Tools, and Retries."""

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.agents.research_agent import (
    MockSearchClient,
    ResearchDependencies,
    create_research_agent,
)
from app.schemas.research import ResearchResult, SourceType


@pytest.mark.asyncio
async def test_successful_agent_execution() -> None:
    """Test successful agent run using deterministic TestModel with custom output args."""
    valid_args = {
        "query": "Explain Python async execution",
        "summary": {
            "topic": "Python Asyncio",
            "title": "Asynchronous Programming in Python",
            "summary": "Asyncio provides event loop abstractions for non-blocking I/O operations.",
            "category": "systems_design",
            "key_findings": ["Event loop manages tasks", "async/await syntax"],
            "confidence": 0.88,
            "sources": [
                {
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "title": "Python Asyncio Docs",
                    "source_type": SourceType.DOCUMENTATION.value,
                    "reliability_score": 0.95,
                }
            ],
        },
        "execution_time_ms": 95.0,
        "status": "success",
    }

    agent = create_research_agent(model=TestModel(custom_output_args=valid_args))
    deps = ResearchDependencies(search_client=MockSearchClient(), min_confidence=0.7)

    result = await agent.run("Explain Python async execution", deps=deps)

    assert isinstance(result.output, ResearchResult)
    assert result.output.summary.topic == "Python Asyncio"
    assert result.output.summary.confidence == 0.88
    assert len(result.output.summary.sources) == 1


@pytest.mark.asyncio
async def test_agent_validation_retry_and_recovery() -> None:
    """Test agent retry behavior when initial output confidence is below required threshold."""
    attempt = 0

    def retry_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempt
        attempt += 1
        output_tool_name = info.output_tools[0].name if info.output_tools else "final_result"

        if attempt == 1:
            # First attempt: output confidence 0.4 (triggers ModelRetry because min_confidence=0.7)
            low_confidence_args = {
                "query": "Test query",
                "summary": {
                    "topic": "Low Confidence Topic",
                    "title": "Uncertain Findings",
                    "summary": "This summary has insufficient confidence level.",
                    "category": "general_technology",
                    "key_findings": ["Unverified finding"],
                    "confidence": 0.40,  # Below min threshold 0.70
                    "sources": [
                        {
                            "url": "https://example.com",
                            "title": "Example",
                            "source_type": "web_article",
                            "reliability_score": 0.5,
                        }
                    ],
                },
                "execution_time_ms": 50.0,
                "status": "partial",
            }
            part = ToolCallPart(tool_name=output_tool_name, args=low_confidence_args)
            return ModelResponse(parts=[part])
        else:
            # Second attempt (after retry prompt): output valid confidence 0.85
            recovered_args = {
                "query": "Test query",
                "summary": {
                    "topic": "Recovered Topic",
                    "title": "Verified Findings",
                    "summary": "Detailed summary after retry with refined analysis.",
                    "category": "ai_architecture",
                    "key_findings": ["Verified finding after retry"],
                    "confidence": 0.85,  # Valid confidence >= 0.70
                    "sources": [
                        {
                            "url": "https://ai.pydantic.dev/",
                            "title": "PydanticAI Specs",
                            "source_type": "documentation",
                            "reliability_score": 0.95,
                        }
                    ],
                },
                "execution_time_ms": 110.0,
                "status": "success",
            }
            part = ToolCallPart(tool_name=output_tool_name, args=recovered_args)
            return ModelResponse(parts=[part])

    agent = create_research_agent(model=FunctionModel(retry_model_fn))
    deps = ResearchDependencies(min_confidence=0.7)

    result = await agent.run("Test query", deps=deps)

    assert attempt == 2  # Proves retry cycle occurred
    assert isinstance(result.output, ResearchResult)
    assert result.output.summary.confidence == 0.85
    assert result.output.summary.topic == "Recovered Topic"


@pytest.mark.asyncio
async def test_agent_max_retries_exceeded() -> None:
    """Test UnexpectedModelBehavior error when model repeatedly fails validation."""

    def invalid_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        output_tool_name = info.output_tools[0].name if info.output_tools else "final_result"
        invalid_args = {
            "query": "Failing query",
            "summary": {
                "topic": "Always Failing",
                "title": "Bad Output",
                "summary": "Consistently fails confidence validation.",
                "category": "general_technology",
                "key_findings": ["Finding"],
                "confidence": 0.30,  # Below threshold 0.70
                "sources": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "source_type": "other",
                        "reliability_score": 0.4,
                    }
                ],
            },
            "execution_time_ms": 30.0,
            "status": "error",
        }
        return ModelResponse(parts=[ToolCallPart(tool_name=output_tool_name, args=invalid_args)])

    agent = create_research_agent(model=FunctionModel(invalid_model_fn))
    deps = ResearchDependencies(min_confidence=0.7)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await agent.run("Failing query", deps=deps)

    err_msg = str(exc_info.value).lower()
    assert "exceeded maximum" in err_msg or "retries" in err_msg
