"""Unit tests for Pydantic schema validation rules."""

import pytest
from pydantic import ValidationError

from app.schemas.research import (
    ResearchResult,
    ResearchSource,
    ResearchSummary,
    SourceType,
)


def test_valid_research_summary() -> None:
    """Test creating a valid ResearchSummary instance."""
    source = ResearchSource(
        url="https://ai.pydantic.dev/",
        title="PydanticAI Docs",
        source_type=SourceType.DOCUMENTATION,
        reliability_score=0.9,
    )
    summary = ResearchSummary(
        topic="Agentic AI",
        title="PydanticAI Validation Layer",
        summary="Detailed summary of structured output enforcement for LLMs.",
        category="ai_architecture",
        key_findings=["Finding 1", "Finding 2"],
        confidence=0.85,
        sources=[source],
    )
    assert summary.topic == "Agentic AI"
    assert summary.confidence == 0.85
    assert len(summary.sources) == 1


def test_invalid_confidence_too_high() -> None:
    """Test validation error when confidence > 1.0."""
    source = ResearchSource(
        url="https://ai.pydantic.dev/",
        title="PydanticAI Docs",
        source_type=SourceType.DOCUMENTATION,
        reliability_score=0.9,
    )
    with pytest.raises(ValidationError) as exc_info:
        ResearchSummary(
            topic="Agentic AI",
            title="PydanticAI Validation Layer",
            summary="Detailed summary of structured output enforcement.",
            category="ai_architecture",
            key_findings=["Finding 1"],
            confidence=1.5,  # Invalid: > 1.0
            sources=[source],
        )
    assert "confidence" in str(exc_info.value)


def test_invalid_confidence_too_low() -> None:
    """Test validation error when confidence < 0.0."""
    source = ResearchSource(
        url="https://ai.pydantic.dev/",
        title="PydanticAI Docs",
        source_type=SourceType.DOCUMENTATION,
        reliability_score=0.9,
    )
    with pytest.raises(ValidationError) as exc_info:
        ResearchSummary(
            topic="Agentic AI",
            title="PydanticAI Validation Layer",
            summary="Detailed summary of structured output enforcement.",
            category="ai_architecture",
            key_findings=["Finding 1"],
            confidence=-0.1,  # Invalid: < 0.0
            sources=[source],
        )
    assert "confidence" in str(exc_info.value)


def test_missing_required_fields() -> None:
    """Test validation error when required fields are missing."""
    with pytest.raises(ValidationError) as exc_info:
        # Missing title, summary, category, key_findings, confidence, sources
        ResearchSummary(topic="Only Topic")  # type: ignore[call-arg]
    errors = str(exc_info.value)
    assert "title" in errors
    assert "summary" in errors
    assert "sources" in errors


def test_empty_sources_list() -> None:
    """Test validation error when sources list is empty."""
    with pytest.raises(ValidationError) as exc_info:
        ResearchSummary(
            topic="Agentic AI",
            title="PydanticAI Validation Layer",
            summary="Detailed summary of structured output enforcement.",
            category="ai_architecture",
            key_findings=["Finding 1"],
            confidence=0.8,
            sources=[],  # Invalid: empty list
        )
    assert "sources" in str(exc_info.value)


def test_cross_field_validation_high_confidence_low_reliability() -> None:
    """Test model validator enforcing source reliability for high confidence."""
    unreliable_source = ResearchSource(
        url="https://example.com/blog",
        title="Blog Post",
        source_type=SourceType.WEB_ARTICLE,
        reliability_score=0.3,  # Low reliability
    )
    with pytest.raises(ValidationError) as exc_info:
        ResearchSummary(
            topic="Agentic AI",
            title="PydanticAI Validation Layer",
            summary="Detailed summary of structured output enforcement.",
            category="ai_architecture",
            key_findings=["Finding 1"],
            confidence=0.95,  # High confidence with unreliable source
            sources=[unreliable_source],
        )
    assert "Confidence score" in str(exc_info.value)


def test_serialization_and_deserialization() -> None:
    """Test round-trip JSON serialization and validation."""
    source = ResearchSource(
        url="https://ai.pydantic.dev/",
        title="PydanticAI Docs",
        source_type=SourceType.DOCUMENTATION,
        reliability_score=0.9,
    )
    summary = ResearchSummary(
        topic="Serialization Test",
        title="Valid Title",
        summary="Detailed summary for testing roundtrip serialization.",
        category="ai_architecture",
        key_findings=["Finding 1"],
        confidence=0.75,
        sources=[source],
    )
    result = ResearchResult(
        query="Test query",
        summary=summary,
        execution_time_ms=50.0,
        status="success",
    )

    json_data = result.model_dump_json()
    reconstructed = ResearchResult.model_validate_json(json_data)
    assert reconstructed.query == "Test query"
    assert reconstructed.summary.confidence == 0.75
    assert reconstructed.summary.sources[0].url == "https://ai.pydantic.dev/"
