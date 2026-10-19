"""DSPy Optimization workflow comparing baseline vs optimized program performances."""

import json
from pathlib import Path
from typing import Any

from dspy.teleprompt import LabeledFewShot

from app.dspy_lab.dataset import load_dataset
from app.dspy_lab.evaluate import evaluate_program
from app.dspy_lab.metrics import research_eval_metric
from app.dspy_lab.modules import ResearcherModule, configure_dspy_lm
from app.logging_config import get_logger

logger = get_logger("app.dspy_lab.optimize")


def run_optimization_pipeline(
    output_dir: Path | None = None,
    k_shots: int = 2,
) -> dict[str, Any]:
    """Run baseline evaluation, apply LabeledFewShot DSPy optimizer, and compare performances.

    Args:
        output_dir: Directory path to persist JSON result artifacts.
        k_shots: Number of few-shot demonstration examples for LabeledFewShot optimizer.

    Returns:
        Structured empirical comparison summary dictionary.
    """
    logger.info("Initializing DSPy Optimization Pipeline", extra={"k_shots": k_shots})
    configure_dspy_lm()

    trainset, devset = load_dataset()

    # 1. Instantiate and Evaluate Baseline Program
    baseline_module = ResearcherModule()
    baseline_report = evaluate_program(
        program=baseline_module,
        devset=devset,
        metric_fn=research_eval_metric,
        program_name="Baseline Researcher",
    )

    baseline_score = baseline_report["overall_composite_score"]
    logger.info("Baseline Program Evaluation Completed", extra={"baseline_score": baseline_score})

    # 2. Compile Optimized Program using LabeledFewShot Optimizer
    optimizer = LabeledFewShot(k=k_shots)
    optimized_module = optimizer.compile(baseline_module, trainset=trainset)

    # 3. Evaluate Optimized Program
    optimized_report = evaluate_program(
        program=optimized_module,
        devset=devset,
        metric_fn=research_eval_metric,
        program_name=f"Optimized Researcher (LabeledFewShot k={k_shots})",
    )

    optimized_score = optimized_report["overall_composite_score"]
    logger.info(
        "Optimized Program Evaluation Completed", extra={"optimized_score": optimized_score}
    )

    # 4. Compare Baseline vs Optimized Performance
    score_delta = round(optimized_score - baseline_score, 4)

    comparison_summary = {
        "dataset_info": {
            "num_train_examples": len(trainset),
            "num_dev_examples": len(devset),
        },
        "optimizer": {
            "type": "LabeledFewShot",
            "k_shots": k_shots,
        },
        "baseline": {
            "program_name": baseline_report["program_name"],
            "composite_score": baseline_score,
            "metric_breakdown": baseline_report["metric_breakdown"],
        },
        "optimized": {
            "program_name": optimized_report["program_name"],
            "composite_score": optimized_score,
            "metric_breakdown": optimized_report["metric_breakdown"],
        },
        "comparison": {
            "score_delta": score_delta,
            "demonstrated_improvement": score_delta > 0,
            "summary_claim": (
                f"Optimization demonstrated a empirical score change of {score_delta:+.4f} "
                f"(Baseline: {baseline_score:.4f} -> Optimized: {optimized_score:.4f})."
            ),
        },
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "evaluation_comparison.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(comparison_summary, f, indent=2)
        logger.info("Saved comparison report", extra={"path": str(report_file)})

    return comparison_summary
