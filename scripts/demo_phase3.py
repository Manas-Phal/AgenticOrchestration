"""Phase 3 DSPy Declarative Optimization Demonstration Script.

Demonstrates:
1. DSPy Signatures (ResearchSignature, CriticSignature)
2. DSPy Modules (ResearcherModule, CriticModule)
3. Custom Evaluation Dataset (20+ Examples)
4. Evaluation Metrics (Relevance, Completeness, Structure, Support Quality)
5. Baseline Evaluation vs DSPy Optimizer (LabeledFewShot) Compilation
6. Performance comparison and JSON artifact export
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dspy_lab import run_optimization_pipeline
from app.logging_config import setup_logging


def run_phase3_demonstration() -> None:
    """Execute complete Phase 3 DSPy optimization demonstration."""
    setup_logging(log_level="INFO", json_format=False)
    print("=" * 80)
    print("PHASE 3: DSPY DECLARATIVE OPTIMIZATION DEMONSTRATION")
    print("=" * 80)

    output_dir = Path(__file__).resolve().parent.parent / "artifacts_phase3"

    print("\n[1] Running Baseline vs Optimized DSPy Pipeline...")
    summary = run_optimization_pipeline(output_dir=output_dir, k_shots=2)

    print("\n--- EVALUATION & OPTIMIZATION REPORT ---")
    print(json.dumps(summary, indent=2))
    print("---------------------------------------\n")

    print(f"Artifacts saved to: {output_dir / 'evaluation_comparison.json'}")
    print("\n" + "=" * 80)
    print("PHASE 3 DEMONSTRATION SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3_demonstration()
