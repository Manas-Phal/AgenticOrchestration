"""Explicit state schema for the Integrated Agentic Orchestration System."""

import time
import uuid
from typing import Any, TypedDict

from app.schemas.domain import (
    FinalResponse,
    HumanReviewDecision,
    Plan,
    QualityAssessment,
    ResearchResult,
)


class OrchestratorState(TypedDict, total=False):
    """Explicit application state dictionary passed through LangGraph nodes.

    Contains validated domain Pydantic objects rather than raw prompt strings,
    along with telemetry counters, latency measurements, and error taxonomy logs.
    """

    request_id: str
    user_query: str
    plan: Plan | None
    research_result: ResearchResult | None
    quality_assessment: QualityAssessment | None
    human_decision: HumanReviewDecision | None
    final_response: FinalResponse | None
    iteration: int
    max_iterations: int
    quality_threshold: float
    status: str
    errors: list[str]
    error_details: list[dict[str, Any]]
    execution_start_ms: float
    total_duration_ms: float
    retry_count: int
    validation_failures: int
    node_durations_ms: dict[str, float]
    token_usage: dict[str, int]
    estimated_cost: float


def create_initial_orchestrator_state(
    user_query: str,
    request_id: str | None = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.8,
) -> OrchestratorState:
    """Factory helper constructing initial OrchestratorState dict."""
    return OrchestratorState(
        request_id=request_id or f"req-{uuid.uuid4().hex[:8]}",
        user_query=user_query,
        plan=None,
        research_result=None,
        quality_assessment=None,
        human_decision=None,
        final_response=None,
        iteration=0,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        status="pending",
        errors=[],
        error_details=[],
        execution_start_ms=time.time() * 1000.0,
        total_duration_ms=0.0,
        retry_count=0,
        validation_failures=0,
        node_durations_ms={},
        token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        estimated_cost=0.0,
    )
