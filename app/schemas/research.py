"""Strictly validated Pydantic models for Research Agent outputs."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SourceType(StrEnum):
    """Types of research sources."""

    DOCUMENTATION = "documentation"
    ACADEMIC_PAPER = "academic_paper"
    WEB_ARTICLE = "web_article"
    REPOSITORY = "repository"
    OTHER = "other"


ResearchCategory = Literal[
    "ai_architecture",
    "data_engineering",
    "systems_design",
    "general_technology",
]


class ResearchSource(BaseModel):
    """Represents a validated individual reference source."""

    url: str = Field(..., min_length=5, description="Valid source URL or URI identifier")
    title: str = Field(
        ..., min_length=1, max_length=200, description="Non-empty title of the source"
    )
    source_type: SourceType = Field(..., description="Categorized source origin")
    reliability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Source reliability rating between 0.0 and 1.0",
    )

    @field_validator("title", "url", mode="before")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Ensure string fields are trimmed and non-empty after stripping."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("Field cannot be empty or blank whitespace")
            return stripped
        return value


class ResearchSummary(BaseModel):
    """Structured research output with strict schema rules and confidence bounds."""

    topic: str = Field(..., min_length=1, max_length=150, description="Core topic of research")
    title: str = Field(
        ..., min_length=1, max_length=200, description="Title of the research report"
    )
    summary: str = Field(
        ...,
        min_length=10,
        description="Comprehensive summary of findings (min 10 chars)",
    )
    category: ResearchCategory = Field(..., description="High-level category classification")
    key_findings: list[str] = Field(
        ...,
        min_length=1,
        description="List of key takeaway findings (at least 1 item)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall research confidence score between 0.0 and 1.0",
    )
    sources: list[ResearchSource] = Field(
        ...,
        min_length=1,
        description="Non-empty list of research sources used",
    )

    @field_validator("title", "topic", "summary", mode="before")
    @classmethod
    def strip_text_fields(cls, value: str) -> str:
        """Strip text fields and prevent whitespace-only strings."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("String field cannot be empty or whitespace only")
            return stripped
        return value

    @field_validator("key_findings")
    @classmethod
    def validate_key_findings(cls, findings: list[str]) -> list[str]:
        """Ensure all key findings are non-empty strings."""
        cleaned = [f.strip() for f in findings if isinstance(f, str) and f.strip()]
        if not cleaned:
            raise ValueError("Key findings must contain at least one non-empty string item")
        return cleaned

    @model_validator(mode="after")
    def validate_confidence_against_sources(self) -> "ResearchSummary":
        """Cross-field validation: High confidence (>0.8) requires reliable sources (>0.6)."""
        if self.confidence > 0.8:
            avg_reliability = sum(s.reliability_score for s in self.sources) / len(self.sources)
            if avg_reliability < 0.6:
                raise ValueError(
                    f"Confidence score ({self.confidence}) cannot exceed 0.8 "
                    f"when average source reliability is low ({avg_reliability:.2f})"
                )
        return self


class ResearchResult(BaseModel):
    """Top-level container for complete agent execution result."""

    query: str = Field(..., min_length=1, description="Original user prompt query")
    summary: ResearchSummary = Field(..., description="Nested structured research summary")
    execution_time_ms: float = Field(
        ..., ge=0.0, description="Agent execution time in milliseconds"
    )
    status: Literal["success", "partial", "error"] = Field(
        default="success",
        description="Status flag of research result",
    )
