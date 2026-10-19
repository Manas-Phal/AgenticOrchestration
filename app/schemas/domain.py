"""Strongly typed Pydantic domain models for production-grade agentic orchestration."""

import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.research import SourceType


class Plan(BaseModel):
    """Structured execution plan model."""

    request_id: str = Field(..., min_length=1, description="Unique request identifier")
    user_query: str = Field(..., min_length=1, description="Original user prompt query")
    steps: list[str] = Field(..., min_length=1, description="List of plan execution steps")
    created_at_ms: float = Field(
        default_factory=lambda: time.time() * 1000.0,
        ge=0.0,
        description="Plan creation timestamp in milliseconds",
    )

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, steps: list[str]) -> list[str]:
        """Ensure steps list contains non-empty strings."""
        cleaned = [s.strip() for s in steps if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError("Plan must contain at least one non-empty step string.")
        return cleaned


class ResearchSource(BaseModel):
    """Validated research source citation model."""

    url: str = Field(..., min_length=5, description="Valid source URL or URI identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Source title")
    source_type: SourceType = Field(
        default=SourceType.DOCUMENTATION, description="Source origin category"
    )
    reliability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Reliability score between 0.0 and 1.0"
    )

    @field_validator("title", "url", mode="before")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        """Strip whitespace and enforce non-empty string."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("String field cannot be empty whitespace.")
            return stripped
        return value


class ResearchResult(BaseModel):
    """Domain object representing validated research findings."""

    topic: str = Field(..., min_length=1, max_length=150, description="Core topic classification")
    summary: str = Field(
        ..., min_length=5, description="Comprehensive technical summary of findings"
    )
    key_findings: list[str] = Field(
        ..., min_length=1, description="List of key takeaway findings"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    sources: list[ResearchSource] = Field(
        default_factory=list, description="Validated research sources"
    )
    iteration: int = Field(default=1, ge=1, description="Workflow loop iteration counter")

    @field_validator("key_findings")
    @classmethod
    def validate_key_findings(cls, findings: list[str]) -> list[str]:
        """Ensure key findings contain valid non-empty strings."""
        cleaned = [f.strip() for f in findings if isinstance(f, str) and f.strip()]
        if not cleaned:
            raise ValueError("Key findings must contain at least one non-empty string.")
        return cleaned


class Critique(BaseModel):
    """Detailed critique assessment object."""

    strengths: list[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Identified weaknesses or gaps")
    factual_concerns: list[str] = Field(
        default_factory=list, description="Identified factual errors"
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list, description="Actionable improvement recommendations"
    )


class QualityAssessment(BaseModel):
    """Quality Gate evaluation assessment object."""

    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Numerical score between 0.0 and 1.0"
    )
    passed: bool = Field(
        ..., description="Boolean flag indicating whether quality threshold was met"
    )
    threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Required quality gate threshold"
    )
    critique: Critique = Field(..., description="Nested critique evaluation object")


class HumanReviewDecision(BaseModel):
    """Structured decision payload from human-in-the-loop review."""

    action: Literal["approve", "reject", "edit"] = Field(
        ..., description="Human review decision action"
    )
    reviewer_id: str = Field(default="human_reviewer", description="Identifier of human reviewer")
    feedback: str | None = Field(default=None, description="Optional text feedback from reviewer")
    edited_plan: list[str] | None = Field(default=None, description="Optional edited plan steps")
    edited_quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional override quality score"
    )


class FinalResponse(BaseModel):
    """Top-level validated final response schema."""

    request_id: str = Field(..., min_length=1, description="Unique request identifier")
    user_query: str = Field(..., min_length=1, description="Original user prompt query")
    final_answer: str = Field(..., min_length=1, description="Synthesized final answer string")
    plan: Plan = Field(..., description="Validated execution plan model")
    research_result: ResearchResult = Field(..., description="Validated research result model")
    quality_assessment: QualityAssessment = Field(
        ..., description="Validated quality assessment model"
    )
    total_iterations: int = Field(..., ge=1, description="Total graph loop iterations executed")
    execution_time_ms: float = Field(
        ..., ge=0.0, description="Total execution time in milliseconds"
    )
    status: Literal["completed", "human_approved", "human_edited", "rejected", "failed"] = Field(
        ..., description="Final execution status flag"
    )
