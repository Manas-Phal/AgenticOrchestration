"""Deterministic and mock node implementations for the LangGraph research workflow."""

from typing import Any

from langgraph.types import interrupt

from app.logging_config import get_logger
from app.workflow.state import ResearchWorkflowState

logger = get_logger("app.workflow.nodes")


def planner_node(state: ResearchWorkflowState) -> dict[str, Any]:
    """Planner node: Deconstructs user_query into a structured research plan."""
    query = state.get("user_query", "")
    logger.info(
        "Executing Planner Node",
        extra={"request_id": state.get("request_id"), "query": query},
    )

    plan = [
        f"Deconstruct core objective for '{query}'",
        "Gather empirical research sources and raw data points",
        "Evaluate gathered data against quality standards",
        "Synthesize structured final answer",
    ]

    return {
        "plan": plan,
        "status": "planning",
    }


def research_node(state: ResearchWorkflowState) -> dict[str, Any]:
    """Research node: Executes mock data retrieval, incrementing iteration counter."""
    current_iteration = state.get("iteration", 0) + 1
    query = state.get("user_query", "")
    existing_results = list(state.get("research_results", []))

    logger.info(
        "Executing Research Node",
        extra={
            "request_id": state.get("request_id"),
            "iteration": current_iteration,
            "max_iterations": state.get("max_iterations", 3),
        },
    )

    new_result: dict[str, Any] = {
        "iteration": current_iteration,
        "query": query,
        "source_id": f"src-iter-{current_iteration}",
        "source_title": f"Technical Data Source #{current_iteration} for '{query}'",
        "findings": [
            f"Iteration {current_iteration} empirical observation A",
            f"Iteration {current_iteration} empirical observation B",
        ],
        "relevance_score": min(0.6 + 0.15 * current_iteration, 0.95),
    }

    existing_results.append(new_result)

    return {
        "iteration": current_iteration,
        "research_results": existing_results,
        "status": "researching",
    }


def critic_node(state: ResearchWorkflowState) -> dict[str, Any]:
    """Critic node: Evaluates research quality score and issues critique text."""
    query = state.get("user_query", "")
    current_iteration = state.get("iteration", 1)
    threshold = state.get("quality_threshold", 0.8)

    # Deterministic test hooks for controlling quality gate paths
    if "instant_pass" in query.lower():
        score = 0.95
    elif "always_fail" in query.lower():
        score = 0.50
    else:
        # Default behavior: iteration 1 = 0.60 (retry needed), iteration 2+ = 0.85 (pass)
        score = 0.60 if current_iteration == 1 else 0.85

    critique = (
        f"Iteration {current_iteration} evaluation for '{query}': "
        f"quality_score={score:.2f} (Required Threshold: {threshold:.2f})."
    )

    logger.info(
        "Executing Critic Node",
        extra={
            "request_id": state.get("request_id"),
            "iteration": current_iteration,
            "score": score,
            "threshold": threshold,
        },
    )

    return {
        "quality_score": score,
        "critique": critique,
        "status": "critiquing",
    }


def quality_gate_router(state: ResearchWorkflowState) -> str:
    """Conditional Edge Router (Quality Gate).

    Routes:
    - score >= threshold -> 'finalizer'
    - iteration >= max_iterations -> 'human_review'
    - score < threshold -> 'research' (retry loop)
    """
    quality_score = state.get("quality_score", 0.0)
    threshold = state.get("quality_threshold", 0.8)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    logger.info(
        "Evaluating Quality Gate Router",
        extra={
            "request_id": state.get("request_id"),
            "quality_score": quality_score,
            "threshold": threshold,
            "iteration": iteration,
            "max_iterations": max_iterations,
        },
    )

    if quality_score >= threshold:
        return "finalizer"
    if iteration >= max_iterations:
        return "human_review"
    return "research"


def human_review_node(state: ResearchWorkflowState) -> dict[str, Any]:
    """Human Review node: Triggers an interrupt for human intervention.

    Pauses execution until resumed with a human approval or rejection command.
    """
    logger.info(
        "Entering Human Review Node - Triggering Interrupt",
        extra={
            "request_id": state.get("request_id"),
            "iteration": state.get("iteration"),
            "score": state.get("quality_score"),
        },
    )

    human_response = interrupt(
        {
            "request_id": state.get("request_id"),
            "user_query": state.get("user_query"),
            "iteration": state.get("iteration"),
            "quality_score": state.get("quality_score"),
            "reason": (
                f"Iteration limit ({state.get('max_iterations')}) reached with quality score "
                f"{state.get('quality_score', 0.0):.2f} below threshold "
                f"{state.get('quality_threshold', 0.8):.2f}. Human approval required."
            ),
        }
    )

    logger.info(
        "Human Review Node Resumed",
        extra={"request_id": state.get("request_id"), "response": human_response},
    )

    if isinstance(human_response, dict) and (
        human_response.get("action") == "approve" or human_response.get("approved") is True
    ):
        approved_score = max(
            state.get("quality_score", 0.0), state.get("quality_threshold", 0.8)
        )
        return {
            "status": "human_approved",
            "quality_score": approved_score,
            "critique": (state.get("critique") or "") + " [Approved via Human Review]",
        }

    errors = list(state.get("errors", []))
    errors.append("Execution rejected during human review.")
    return {
        "status": "human_rejected",
        "errors": errors,
    }


def finalizer_node(state: ResearchWorkflowState) -> dict[str, Any]:
    """Finalizer node: Synthesizes final answer from state data."""
    status = state.get("status", "")
    query = state.get("user_query", "")
    iteration = state.get("iteration", 0)
    score = state.get("quality_score", 0.0)
    results = state.get("research_results", [])

    logger.info(
        "Executing Finalizer Node",
        extra={"request_id": state.get("request_id"), "status": status},
    )

    if status == "human_rejected":
        final_answer = (
            f"Request '{query}' was rejected during human review after {iteration} iterations."
        )
    else:
        final_answer = (
            f"Final Answer for '{query}': Synthesized from {len(results)} research iterations. "
            f"Final Quality Score: {score:.2f}."
        )

    return {
        "final_answer": final_answer,
        "status": "completed",
    }
