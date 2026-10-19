"""End-to-end Demonstration of the Production-Grade Agentic Orchestration System.

Combines:
- LangGraph (Workflow Control, Routing, Loops, Checkpointing, Human Interrupts)
- Pydantic / PydanticAI (Schema Validation, Type Safety, Domain Conversion Boundaries)
- DSPy (Declarative Research Synthesis & Critic Auditing)
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import setup_logging
from app.orchestration import OrchestrationEngine
from app.schemas.domain import HumanReviewDecision


def run_full_system_demonstration() -> None:
    """Execute complete end-to-end Agentic Orchestration system demonstration."""
    setup_logging(log_level="INFO", json_format=False)
    print("=" * 80)
    print("PRODUCTION-GRADE AGENTIC ORCHESTRATION SYSTEM: END-TO-END DEMO")
    print("Architectural Boundaries Enforced:")
    print("  • LangGraph   : Workflow Control | Loops | State | Interrupts")
    print("  • PydanticAI  : Type Safety | Schema Validation | Conversion Boundaries")
    print("  • DSPy        : Declarative Research & Critic Auditing")
    print("=" * 80)

    engine = OrchestrationEngine()

    # --- 1. Successful End-to-End Execution (Instant Pass Path) ---
    print("\n[1] Executing Request (Instant Pass Path)...")
    thread_1 = "full-demo-thread-pass"
    res_1 = engine.start_request(
        user_query="instant_pass: High-Availability Microservices Pattern",
        thread_id=thread_1,
    )

    final_resp_1 = res_1["final_response"]
    print("\n--- Request 1 Result ---")
    print(f"Request ID : {final_resp_1.request_id}")
    print(f"Status     : {final_resp_1.status}")
    print(f"Score      : {final_resp_1.quality_assessment.quality_score:.2f}")
    print(f"Iterations : {final_resp_1.total_iterations}")
    print(f"Answer     : {final_resp_1.final_answer}")

    # --- 2. Retry Loop & Quality Gate Enforcement ---
    print("\n[2] Executing Request with Quality Gate Retry Loop...")
    thread_2 = "full-demo-thread-retry"
    res_2 = engine.start_request(
        user_query="Distributed Consensus Protocols & Replication",
        thread_id=thread_2,
        max_iterations=3,
        quality_threshold=0.8,
    )

    final_resp_2 = res_2["final_response"]
    print("\n--- Request 2 Result (After Retry Loop) ---")
    print(f"Request ID : {final_resp_2.request_id}")
    print(f"Status     : {final_resp_2.status}")
    print(f"Iterations : {final_resp_2.total_iterations}")
    print(f"Score      : {final_resp_2.quality_assessment.quality_score:.2f}")
    print(f"Answer     : {final_resp_2.final_answer}")

    # --- 3. Iteration Limit & Human Review Escalation ---
    print("\n[3] Executing Request hitting Iteration Limit (Human Escalation)...")
    thread_3 = "full-demo-thread-human"
    res_3 = engine.start_request(
        user_query="always_fail: High-Risk System Kernel Upgrade",
        thread_id=thread_3,
        max_iterations=2,
    )

    print("\n--- Request 3 Status (Interrupted at Human Review) ---")
    if "__interrupt__" in res_3:
        interrupt_payload = res_3["__interrupt__"][0].value
        print("Graph Interrupted : Yes")
        print(f"Reason            : {interrupt_payload.get('reason')}")
        print(f"Quality Score     : {interrupt_payload.get('quality_score')}")

    # --- 4. Resuming Interrupted Request with Human Plan & Score Edit ---
    print("\n[4] Resuming Interrupted Request with Human Plan Edit...")
    decision = HumanReviewDecision(
        action="edit",
        reviewer_id="lead_security_architect",
        feedback="Approved with mandatory staging environment verification",
        edited_plan=[
            "Staging environment canary deployment",
            "Kernel patch verification & audit log review",
            "Production rollout with rollback triggers",
        ],
        edited_quality_score=0.92,
    )

    resumed_res = engine.resume_request(thread_id=thread_3, decision=decision)
    final_resp_3 = resumed_res["final_response"]

    print("\n--- Request 3 Result (Resumed & Completed) ---")
    print(f"Status      : {final_resp_3.status}")
    print(f"Final Score : {final_resp_3.quality_assessment.quality_score:.2f}")
    print(f"Plan Steps  : {json.dumps(final_resp_3.plan.steps, indent=2)}")
    print(f"Answer      : {final_resp_3.final_answer}")

    print("\n" + "=" * 80)
    print("FULL AGENTIC ORCHESTRATION SYSTEM DEMONSTRATION SUCCESSFUL!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_system_demonstration()
