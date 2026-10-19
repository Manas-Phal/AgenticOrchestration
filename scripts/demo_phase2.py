"""Phase 2 LangGraph Workflow Demonstration Script.

Demonstrates all 12 core workflow features:
1. State
2. Nodes
3. START/END
4. Edges
5. Conditional routing
6. Cycles
7. Retry behavior
8. Iteration limits
9. Checkpointing
10. Thread IDs
11. Interrupt/human-in-the-loop
12. Resume after human approval
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import setup_logging
from app.workflow import (
    WorkflowRunner,
    create_workflow_graph,
    export_graph_mermaid,
)


def run_phase2_demonstration() -> None:
    """Execute complete Phase 2 LangGraph orchestration demonstration."""
    setup_logging(log_level="INFO", json_format=False)
    print("=" * 80)
    print("PHASE 2: LANGGRAPH INDEPENDENT ORCHESTRATION DEMONSTRATION")
    print("=" * 80)

    runner = WorkflowRunner()

    # --- 1. Graph Topology & Mermaid Visualization ---
    print("\n[1] Exporting Graph Topology (Mermaid Syntax)...")
    graph = create_workflow_graph()
    mermaid_diagram = export_graph_mermaid(graph)
    print("\n--- MERMAID GRAPH DIAGRAM ---")
    print(mermaid_diagram)
    print("-----------------------------\n")

    # --- 2. Demonstrating Successful Path & Retry Cycle ---
    print("\n[2] Demonstrating Retry Path & Quality Gate Routing...")
    thread_1 = "demo-thread-retry"
    result_1 = runner.start_workflow(
        user_query="Distributed Cache Consensus",
        thread_id=thread_1,
        max_iterations=3,
        quality_threshold=0.8,
    )

    print("\n--- Thread 1 Output ---")
    print(f"Thread ID: {thread_1}")
    print(f"Request ID: {result_1.get('request_id')}")
    print(f"Final Status: {result_1.get('status')}")
    print(f"Iterations Executed: {result_1.get('iteration')}")
    print(f"Quality Score: {result_1.get('quality_score')}")
    print(f"Critique: {result_1.get('critique')}")
    print(f"Final Answer: {result_1.get('final_answer')}")

    # --- 3. Demonstrating Iteration Limit & Human Interrupt ---
    print("\n[3] Demonstrating Iteration Limit & Human-in-the-Loop Interrupt...")
    thread_2 = "demo-thread-interrupt"
    result_2 = runner.start_workflow(
        user_query="always_fail: High-Risk Infrastructure Change",
        thread_id=thread_2,
        max_iterations=2,
        quality_threshold=0.8,
    )

    print("\n--- Thread 2 Output (Interrupted) ---")
    print(f"Thread ID: {thread_2}")
    if "__interrupt__" in result_2:
        interrupt_info = result_2["__interrupt__"][0].value
        print("Graph Execution Interrupted at Node: 'human_review'")
        print(f"Interrupt Reason: {interrupt_info.get('reason')}")
        print(f"Current Iteration: {interrupt_info.get('iteration')}")
        print(f"Quality Score: {interrupt_info.get('quality_score')}")

    # Inspect checkpoint snapshot state
    snapshot = runner.get_state(thread_2)
    iter_val = snapshot.get("iteration") if snapshot else "N/A"
    print(f"Checkpointed Thread State: Iteration {iter_val}")

    # --- 4. Demonstrating Resume After Human Approval ---
    print("\n[4] Resuming Interrupted Thread with Human Approval...")
    resumed_result = runner.resume_workflow(thread_id=thread_2, action="approve", approved=True)

    print("\n--- Thread 2 Output (Resumed & Completed) ---")
    print(f"Final Status: {resumed_result.get('status')}")
    print(f"Final Quality Score: {resumed_result.get('quality_score')}")
    print(f"Final Answer: {resumed_result.get('final_answer')}")
    print("\n" + "=" * 80)
    print("PHASE 2 DEMONSTRATION SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2_demonstration()
