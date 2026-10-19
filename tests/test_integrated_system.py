"""Comprehensive Integration Test Suite for Full Integrated Agentic Orchestration System."""

import pytest
from pydantic import ValidationError

from app.core.validation import OrchestrationValidationError, validate_research_output
from app.orchestration import OrchestrationEngine, create_initial_orchestrator_state
from app.schemas.domain import (
    FinalResponse,
    HumanReviewDecision,
    Plan,
    QualityAssessment,
    ResearchResult,
)


def test_1_successful_end_to_end_execution() -> None:
    """Test 1: Successful end-to-end workflow execution."""
    engine = OrchestrationEngine()
    result = engine.start_request(user_query="instant_pass: High-Throughput Event Streaming")

    assert result["status"] == "completed"
    assert "final_response" in result
    final_resp: FinalResponse = result["final_response"]

    assert final_resp.status == "completed"
    assert final_resp.total_iterations == 1
    assert final_resp.quality_assessment.passed is True
    assert final_resp.quality_assessment.quality_score >= 0.8
    assert final_resp.execution_time_ms >= 0.0


def test_2_invalid_researcher_output_handling() -> None:
    """Test 2: Invalid researcher output schema validation error handling and recovery."""
    # Test valid fallback normalization on partial or missing output
    res = validate_research_output(dspy_output=None, iteration=1, query="Test Query")
    assert isinstance(res, ResearchResult)
    assert res.topic is not None
    assert len(res.sources) > 0

    # Test explicit validation error raising on directive
    engine = OrchestrationEngine()
    with pytest.raises(OrchestrationValidationError) as exc_info:
        engine.start_request(user_query="raise_researcher_validation: Fault Injection")

    assert "Simulated invalid Researcher output" in str(exc_info.value)


def test_3_invalid_critic_output_handling() -> None:
    """Test 3: Invalid critic output schema validation error handling and recovery."""
    engine = OrchestrationEngine()
    with pytest.raises(OrchestrationValidationError) as exc_info:
        engine.start_request(user_query="raise_critic_validation: Fault Injection")

    assert "Simulated invalid Critic output" in str(exc_info.value)


def test_4_retry_behavior() -> None:
    """Test 4: Retry behavior (fails iter 1 quality gate, retries and passes on iter 2)."""
    engine = OrchestrationEngine()
    result = engine.start_request(user_query="Distributed Consensus Protocols", max_iterations=3)

    assert result["status"] == "completed"
    assert result["iteration"] == 2
    final_resp: FinalResponse = result["final_response"]
    assert final_resp.quality_assessment.quality_score == 0.85
    assert final_resp.total_iterations == 2


def test_5_quality_failure() -> None:
    """Test 5: Quality failure triggers Quality Gate routing."""
    state = create_initial_orchestrator_state(user_query="always_fail: Test Quality Failure")
    state["quality_assessment"] = QualityAssessment(
        quality_score=0.5,
        passed=False,
        threshold=0.8,
        critique=state.get("quality_assessment", None) or {"strengths": []},  # type: ignore[arg-type]
    )

    from app.orchestration.nodes import quality_gate_router

    state["iteration"] = 1
    state["max_iterations"] = 3
    route = quality_gate_router(state)
    assert route == "research"  # Routes to retry loop


def test_6_graph_loop() -> None:
    """Test 6: Verifies retry graph loop execution from Quality Gate back to Research."""
    engine = OrchestrationEngine()
    thread_id = "loop_test_001"
    result = engine.start_request(
        user_query="Default Query For Retry Loop",
        thread_id=thread_id,
        max_iterations=3,
    )

    assert result["iteration"] == 2
    assert result["status"] == "completed"


def test_7_iteration_limit() -> None:
    """Test 7: Iteration limit reached routes to Human Review interrupt."""
    engine = OrchestrationEngine()
    thread_id = "limit_test_001"

    result = engine.start_request(
        user_query="always_fail: Quantum Security Protocols",
        thread_id=thread_id,
        max_iterations=2,
    )

    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert len(interrupts) > 0
    assert interrupts[0].value["iteration"] == 2

    # Check snapshot state
    snapshot = engine.graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snapshot.next == ("human_review",)


def test_8_human_approval() -> None:
    """Test 8: Resuming interrupted workflow thread with human approval."""
    engine = OrchestrationEngine()
    thread_id = "human_approve_001"

    engine.start_request(
        user_query="always_fail: System Overhaul",
        thread_id=thread_id,
        max_iterations=2,
    )

    decision = HumanReviewDecision(action="approve", reviewer_id="admin_user_1")
    result = engine.resume_request(thread_id=thread_id, decision=decision)

    assert result["status"] == "human_approved"
    final_resp: FinalResponse = result["final_response"]
    assert final_resp.status == "human_approved"
    assert final_resp.quality_assessment.passed is True


def test_9_human_rejection() -> None:
    """Test 9: Resuming interrupted workflow thread with human rejection."""
    engine = OrchestrationEngine()
    thread_id = "human_reject_001"

    engine.start_request(
        user_query="always_fail: Unsafe Operation",
        thread_id=thread_id,
        max_iterations=2,
    )

    decision = HumanReviewDecision(action="reject", reviewer_id="security_officer")
    result = engine.resume_request(thread_id=thread_id, decision=decision)

    assert result["status"] == "rejected"
    final_resp: FinalResponse = result["final_response"]
    assert final_resp.status == "rejected"
    assert "rejected during human review" in final_resp.final_answer.lower()


def test_10_human_editing() -> None:
    """Test 10: Resuming interrupted workflow thread with human plan and score edits."""
    engine = OrchestrationEngine()
    thread_id = "human_edit_001"

    engine.start_request(
        user_query="always_fail: Ambiguous Infrastructure Request",
        thread_id=thread_id,
        max_iterations=2,
    )

    decision = HumanReviewDecision(
        action="edit",
        reviewer_id="lead_architect",
        edited_plan=["Human edited step 1", "Human edited step 2"],
        edited_quality_score=0.90,
    )
    result = engine.resume_request(thread_id=thread_id, decision=decision)

    assert result["status"] == "human_edited"
    final_resp: FinalResponse = result["final_response"]
    assert final_resp.status == "human_edited"
    assert final_resp.plan.steps == ["Human edited step 1", "Human edited step 2"]
    assert final_resp.quality_assessment.quality_score == 0.90


def test_11_final_schema_validation() -> None:
    """Test 11: Validating FinalResponse Pydantic schema integrity."""
    engine = OrchestrationEngine()
    result = engine.start_request(user_query="instant_pass: Final Schema Validation")

    final_resp = result["final_response"]
    assert isinstance(final_resp, FinalResponse)
    assert isinstance(final_resp.plan, Plan)
    assert isinstance(final_resp.research_result, ResearchResult)
    assert isinstance(final_resp.quality_assessment, QualityAssessment)

    # Test raising ValidationError on invalid data types
    with pytest.raises(ValidationError):
        FinalResponse(
            request_id="test",
            user_query="test",
            final_answer="test",
            plan="invalid_plan_type",  # type: ignore[arg-type]
            research_result="invalid_res_type",  # type: ignore[arg-type]
            quality_assessment="invalid_qa_type",  # type: ignore[arg-type]
            total_iterations=-1,  # Invalid ge=1
            execution_time_ms=10.0,
            status="completed",
        )


def test_12_unexpected_exception_handling() -> None:
    """Test 12: Unexpected exception propagation without silent swallowing."""
    engine = OrchestrationEngine()

    with pytest.raises(OrchestrationValidationError):
        engine.start_request(user_query="raise_researcher_validation: Exception Test")
