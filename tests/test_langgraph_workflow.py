"""Comprehensive Unit Test Suite for Phase 2 LangGraph Research Workflow."""

from langgraph.checkpoint.memory import MemorySaver

from app.workflow import (
    WorkflowRunner,
    create_initial_state,
    create_workflow_graph,
    export_graph_mermaid,
)


def test_initial_state_construction() -> None:
    """Test state initialization helper constructs required raw fields."""
    state = create_initial_state(user_query="Test Architecture Query", request_id="req-123")
    assert state["request_id"] == "req-123"
    assert state["user_query"] == "Test Architecture Query"
    assert state["plan"] is None
    assert state["research_results"] == []
    assert state["critique"] is None
    assert state["quality_score"] == 0.0
    assert state["iteration"] == 0
    assert state["status"] == "pending"
    assert state["final_answer"] is None
    assert state["errors"] == []
    assert state["max_iterations"] == 3
    assert state["quality_threshold"] == 0.8


def test_successful_instant_pass_path() -> None:
    """Test successful path when quality gate passes immediately on iteration 1."""
    runner = WorkflowRunner()
    result = runner.start_workflow(user_query="instant_pass: Graph DB Architecture")

    assert result["status"] == "completed"
    assert result["iteration"] == 1
    assert result["quality_score"] >= 0.8
    assert result["plan"] is not None
    assert len(result["research_results"]) == 1
    assert "Final Answer for" in (result["final_answer"] or "")
    assert result["errors"] == []


def test_failed_quality_gate_and_retry_path() -> None:
    """Test default retry path: fails quality gate on iter 1, retries and succeeds on iter 2."""
    runner = WorkflowRunner()
    thread_id = "retry_thread_001"

    result = runner.start_workflow(
        user_query="Distributed Consensus Protocols",
        thread_id=thread_id,
        max_iterations=3,
        quality_threshold=0.8,
    )

    # First iteration produced 0.60, retried on iteration 2 producing 0.85 -> passed
    assert result["status"] == "completed"
    assert result["iteration"] == 2
    assert result["quality_score"] == 0.85
    assert len(result["research_results"]) == 2
    assert "Final Answer for" in (result["final_answer"] or "")


def test_iteration_limit_and_human_interruption() -> None:
    """Test hitting iteration limit routes to human review interrupt."""
    runner = WorkflowRunner()
    thread_id = "limit_thread_001"

    # Set max_iterations=2 with query that always fails quality threshold
    result = runner.start_workflow(
        user_query="always_fail: Quantum Encryption Standard",
        thread_id=thread_id,
        max_iterations=2,
        quality_threshold=0.8,
    )

    # Verify graph interrupted at human_review node
    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert len(interrupts) > 0

    interrupt_val = interrupts[0].value
    assert interrupt_val["iteration"] == 2
    assert interrupt_val["quality_score"] < 0.8
    assert "Human approval required" in interrupt_val["reason"]

    # Verify snapshot state in checkpointer
    snapshot = runner.graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snapshot.next == ("human_review",)
    state_snapshot = runner.get_state(thread_id)
    assert state_snapshot is not None
    assert state_snapshot["iteration"] == 2


def test_human_resume_with_approval() -> None:
    """Test resuming an interrupted workflow thread with human approval."""
    runner = WorkflowRunner()
    thread_id = "resume_approve_001"

    # Start workflow and trigger interrupt
    runner.start_workflow(
        user_query="always_fail: High-Frequency Trading Core",
        thread_id=thread_id,
        max_iterations=2,
    )

    # Resume workflow with approval decision
    resumed_result = runner.resume_workflow(thread_id=thread_id, action="approve", approved=True)

    assert resumed_result["status"] == "completed"
    assert resumed_result["quality_score"] >= 0.8
    assert "Approved via Human Review" in (resumed_result["critique"] or "")
    assert "Final Answer for" in (resumed_result["final_answer"] or "")


def test_human_resume_with_rejection() -> None:
    """Test resuming an interrupted workflow thread with human rejection."""
    runner = WorkflowRunner()
    thread_id = "resume_reject_001"

    # Start workflow and trigger interrupt
    runner.start_workflow(
        user_query="always_fail: Unsafe System Modification",
        thread_id=thread_id,
        max_iterations=2,
    )

    # Resume workflow with rejection decision
    resumed_result = runner.resume_workflow(thread_id=thread_id, action="reject", approved=False)

    assert resumed_result["status"] == "completed"
    assert "was rejected during human review" in (resumed_result["final_answer"] or "")
    assert any("rejected" in err.lower() for err in resumed_result.get("errors", []))


def test_explicit_state_fields_completeness() -> None:
    """Verify all 10 required explicit state keys exist and maintain correct types."""
    memory = MemorySaver()
    graph = create_workflow_graph(checkpointer=memory)

    config = {"configurable": {"thread_id": "completeness_thread"}}
    initial_state = create_initial_state(
        user_query="instant_pass: State Completeness Verification",
        request_id="req-state-test",
    )

    final_state = graph.invoke(initial_state, config=config)

    # Explicit state keys check
    expected_keys = {
        "request_id",
        "user_query",
        "plan",
        "research_results",
        "critique",
        "quality_score",
        "iteration",
        "status",
        "final_answer",
        "errors",
    }
    for key in expected_keys:
        assert key in final_state, f"State missing required key: {key}"

    assert isinstance(final_state["request_id"], str)
    assert isinstance(final_state["user_query"], str)
    assert isinstance(final_state["plan"], list)
    assert isinstance(final_state["research_results"], list)
    assert isinstance(final_state["critique"], str)
    assert isinstance(final_state["quality_score"], float)
    assert isinstance(final_state["iteration"], int)
    assert isinstance(final_state["status"], str)
    assert isinstance(final_state["final_answer"], str)
    assert isinstance(final_state["errors"], list)


def test_graph_mermaid_export() -> None:
    """Test graph Mermaid export generates non-empty graph visualization string."""
    graph = create_workflow_graph()
    mermaid_str = export_graph_mermaid(graph)

    assert isinstance(mermaid_str, str)
    assert len(mermaid_str) > 0
    assert "planner" in mermaid_str
    assert "research" in mermaid_str
    assert "critic" in mermaid_str
    assert "human_review" in mermaid_str
    assert "finalizer" in mermaid_str
