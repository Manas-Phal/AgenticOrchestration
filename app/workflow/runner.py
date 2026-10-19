"""High-level runner helper for executing and resuming LangGraph research workflows."""

import uuid
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import get_settings
from app.logging_config import get_logger
from app.workflow.graph import create_workflow_graph
from app.workflow.state import ResearchWorkflowState, create_initial_state

logger = get_logger("app.workflow.runner")


class WorkflowRunner:
    """Manages compiled LangGraph workflow execution, thread checkpointers, and human interrupts."""

    def __init__(self, checkpointer: MemorySaver | None = None) -> None:
        """Initialize runner with checkpointer and compiled workflow graph."""
        self.checkpointer = checkpointer or MemorySaver()
        self.graph: CompiledStateGraph[Any, Any, Any] = create_workflow_graph(
            checkpointer=self.checkpointer
        )

    def start_workflow(
        self,
        user_query: str,
        thread_id: str | None = None,
        max_iterations: int | None = None,
        quality_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Start a new workflow thread and run until completion or interrupt.

        Args:
            user_query: Prompt question or topic to research.
            thread_id: Unique thread identifier for state checkpointing.
            max_iterations: Maximum loop iteration limit (prevents infinite cycles).
            quality_threshold: Minimum required score to pass Quality Gate.

        Returns:
            The current state payload (or interrupted state payload).
        """
        settings = get_settings()
        tid = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        req_id = f"req-{uuid.uuid4().hex[:8]}"

        max_iter = (
            max_iterations
            if max_iterations is not None
            else settings.workflow_max_iterations
        )
        threshold = (
            quality_threshold
            if quality_threshold is not None
            else settings.workflow_quality_threshold
        )

        initial_state = create_initial_state(
            user_query=user_query,
            request_id=req_id,
            max_iterations=max_iter,
            quality_threshold=threshold,
        )

        config = cast(RunnableConfig, {"configurable": {"thread_id": tid}})

        logger.info(
            "Starting workflow thread",
            extra={"thread_id": tid, "request_id": req_id, "query": user_query},
        )

        result = self.graph.invoke(initial_state, config=config)
        return cast(dict[str, Any], result)

    def resume_workflow(
        self,
        thread_id: str,
        action: str = "approve",
        approved: bool = True,
    ) -> dict[str, Any]:
        """Resume an interrupted workflow thread following human review decision.

        Args:
            thread_id: The thread ID of the paused workflow.
            action: Human action command ('approve' or 'reject').
            approved: Boolean flag representing approval decision.

        Returns:
            The state payload after resumed workflow finishes.
        """
        config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
        resume_payload = {"action": action, "approved": approved}

        logger.info(
            "Resuming workflow thread",
            extra={"thread_id": thread_id, "resume_payload": resume_payload},
        )

        result = self.graph.invoke(Command(resume=resume_payload), config=config)
        return cast(dict[str, Any], result)

    def get_state(self, thread_id: str) -> ResearchWorkflowState | None:
        """Fetch the current state snapshot for a given thread ID."""
        config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
        snapshot = self.graph.get_state(config)
        if snapshot and snapshot.values:
            return cast(ResearchWorkflowState, snapshot.values)
        return None
