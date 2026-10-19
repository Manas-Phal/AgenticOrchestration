"""LangGraph StateGraph builder, compilation, checkpointer integration, and visualization export."""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.workflow.nodes import (
    critic_node,
    finalizer_node,
    human_review_node,
    planner_node,
    quality_gate_router,
    research_node,
)
from app.workflow.state import ResearchWorkflowState


def create_workflow_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any]:
    """Construct and compile the Phase 2 LangGraph research workflow.

    Workflow topology:
    START -> Planner -> Research -> Critic -> Quality Gate (Conditional Router)
    ├── score >= threshold -> Finalizer -> END
    ├── score < threshold & iteration < limit -> Research (Cycle / Retry)
    └── iteration >= limit -> Human Review (Interrupt) -> Finalizer -> END
    """
    builder = StateGraph(ResearchWorkflowState)

    # Add Workflow Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("research", research_node)
    builder.add_node("critic", critic_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalizer", finalizer_node)

    # Add Edges
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "research")
    builder.add_edge("research", "critic")

    # Quality Gate Conditional Edge Routing
    builder.add_conditional_edges(
        "critic",
        quality_gate_router,
        {
            "finalizer": "finalizer",
            "human_review": "human_review",
            "research": "research",
        },
    )

    # Post Human-Review edge
    builder.add_edge("human_review", "finalizer")

    # Terminal edge
    builder.add_edge("finalizer", END)

    # Compile graph with MemorySaver checkpointer for thread ID state persistence
    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def export_graph_mermaid(compiled_graph: CompiledStateGraph[Any, Any, Any]) -> str:
    """Generate Mermaid syntax markdown string for workflow visualization."""
    return compiled_graph.get_graph().draw_mermaid()
