"""LangGraph StateGraph builder for the Integrated Agentic Orchestration System."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.orchestration.nodes import (
    critic_node,
    finalizer_node,
    human_review_node,
    planner_node,
    quality_gate_router,
    research_node,
)
from app.orchestration.state import OrchestratorState


def create_orchestrator_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any]:
    """Construct and compile the full Production-Grade Agentic Orchestrator graph.

    Workflow topology:
    START -> Planner -> Research (DSPy + Pydantic Validation) -> Critic (DSPy + Pydantic Validation)
    ├── passed == True -> Finalizer -> END
    ├── passed == False & iteration < max -> Research (Retry Loop)
    └── iteration >= max -> Human Review (Interrupt) -> Finalizer -> END
    """
    builder = StateGraph(OrchestratorState)

    # Add Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("research", research_node)
    builder.add_node("critic", critic_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalizer", finalizer_node)

    # Add Edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "research")
    builder.add_edge("research", "critic")

    # Quality Gate Conditional Routing
    builder.add_conditional_edges(
        "critic",
        quality_gate_router,
        {
            "finalizer": "finalizer",
            "human_review": "human_review",
            "research": "research",
        },
    )

    # Post Human-Review Edge
    builder.add_edge("human_review", "finalizer")

    # Terminal Edge
    builder.add_edge("finalizer", END)

    # Compile with checkpointer for thread state management
    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)
