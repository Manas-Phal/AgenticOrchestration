"""Explicit state definition for the LangGraph research workflow."""

import uuid
from typing import Any, TypedDict


class ResearchWorkflowState(TypedDict, total=False):
    """Explicit application state dictionary passed through LangGraph nodes.

    Contains raw application data rather than prompt-formatted strings.
    """

    request_id: str
    user_query: str
    plan: list[str] | str | None
    research_results: list[dict[str, Any]]
    critique: str | None
    quality_score: float
    iteration: int
    status: str
    final_answer: str | None
    errors: list[str]
    max_iterations: int
    quality_threshold: float


def create_initial_state(
    user_query: str,
    request_id: str | None = None,
    max_iterations: int = 3,
    quality_threshold: float = 0.8,
) -> ResearchWorkflowState:
    """Factory helper to construct a clean initial ResearchWorkflowState dictionary."""
    return ResearchWorkflowState(
        request_id=request_id or f"req-{uuid.uuid4().hex[:8]}",
        user_query=user_query,
        plan=None,
        research_results=[],
        critique=None,
        quality_score=0.0,
        iteration=0,
        status="pending",
        final_answer=None,
        errors=[],
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
    )
