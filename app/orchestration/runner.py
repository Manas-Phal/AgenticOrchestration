"""Orchestration Engine wrapper managing execution, thread checkpointing, and human interrupts."""

import uuid
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import get_settings
from app.core.exceptions import (
    OrchestrationBaseError,
    UnexpectedError,
    ValidationError,
)
from app.core.resilience import retry_with_exponential_backoff
from app.logging_config import get_logger, sanitize_value
from app.orchestration.graph import create_orchestrator_graph
from app.orchestration.state import OrchestratorState, create_initial_orchestrator_state
from app.schemas.domain import HumanReviewDecision

logger = get_logger("app.orchestration.runner")


class OrchestrationEngine:
    """Production-grade Agentic Orchestration Engine."""

    def __init__(self, checkpointer: MemorySaver | None = None) -> None:
        """Initialize engine with MemorySaver checkpointer and compiled graph."""
        self.checkpointer = checkpointer or MemorySaver()
        self.graph: CompiledStateGraph[Any, Any, Any] = create_orchestrator_graph(
            checkpointer=self.checkpointer
        )

    def start_request(
        self,
        user_query: str,
        thread_id: str | None = None,
        max_iterations: int | None = None,
        quality_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Start a new workflow thread and execute until completion or human interrupt."""
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

        initial_state = create_initial_orchestrator_state(
            user_query=user_query,
            request_id=req_id,
            max_iterations=max_iter,
            quality_threshold=threshold,
        )

        config = cast(RunnableConfig, {"configurable": {"thread_id": tid}})

        logger.info(
            "Starting Orchestration request",
            extra={"thread_id": tid, "request_id": req_id, "query": sanitize_value(user_query)},
        )

        @retry_with_exponential_backoff(
            max_retries=settings.max_retries,
            base_delay=settings.retry_base_delay,
            max_delay=settings.retry_max_delay,
        )
        def _execute_graph() -> dict[str, Any]:
            return cast(dict[str, Any], self.graph.invoke(initial_state, config=config))

        try:
            return _execute_graph()
        except (ValidationError, OrchestrationBaseError) as err:
            logger.error(
                "Domain or Validation Error during graph execution",
                extra={"thread_id": tid, "request_id": req_id, "error": sanitize_value(str(err))},
            )
            raise
        except Exception as err:
            logger.error(
                "Unhandled exception during graph invocation",
                extra={"thread_id": tid, "request_id": req_id, "error": sanitize_value(str(err))},
            )
            raise UnexpectedError(
                f"Unhandled exception during graph invocation: {err}",
                details={"thread_id": tid, "request_id": req_id},
            ) from err

    def resume_request(
        self,
        thread_id: str,
        decision: HumanReviewDecision | dict[str, Any] | str = "approve",
    ) -> dict[str, Any]:
        """Resume an interrupted workflow thread following a human-in-the-loop decision."""
        config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})

        if isinstance(decision, str):
            payload: dict[str, Any] = {"action": decision}
        elif isinstance(decision, HumanReviewDecision):
            payload = decision.model_dump()
        else:
            payload = decision

        logger.info(
            "Resuming Orchestration request",
            extra={"thread_id": thread_id, "decision": sanitize_value(payload)},
        )

        try:
            result = self.graph.invoke(Command(resume=payload), config=config)
            return cast(dict[str, Any], result)
        except Exception as err:
            logger.error(
                "Unhandled exception during graph resume",
                extra={"error": sanitize_value(str(err)), "thread_id": thread_id},
            )
            raise UnexpectedError(
                f"Unhandled exception during graph resume: {err}",
                details={"thread_id": thread_id},
            ) from err

    def get_state(self, thread_id: str) -> OrchestratorState | None:
        """Fetch current checkpointed state snapshot for a given thread ID."""
        config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
        snapshot = self.graph.get_state(config)
        if snapshot and snapshot.values:
            return cast(OrchestratorState, snapshot.values)
        return None
