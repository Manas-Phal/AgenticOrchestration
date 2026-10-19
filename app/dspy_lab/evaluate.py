"""Evaluation module for running DSPy evaluation benchmarks on datasets."""

import json
from pathlib import Path
from typing import Any

import dspy
from dspy.evaluate import Evaluate

from app.dspy_lab.metrics import (
    compute_completeness_score,
    compute_relevance_score,
    compute_structural_correctness,
    compute_support_quality,
)
from app.logging_config import get_logger

logger = get_logger("app.dspy_lab.evaluate")


def evaluate_program(
    program: dspy.Module,
    devset: list[dspy.Example],
    metric_fn: Any,
    program_name: str = "Baseline Researcher",
    output_json_path: Path | None = None,
) -> dict[str, Any]:
    """Execute DSPy evaluation on a given module and devset, returning detailed metrics dictionary.

    Args:
        program: DSPy module instance to evaluate.
        devset: Validation evaluation dataset (list of dspy.Example).
        metric_fn: Composite metric evaluation function.
        program_name: Name label for reporting (e.g. 'Baseline' vs 'Optimized').
        output_json_path: Optional file path to persist JSON evaluation results.

    Returns:
        Structured evaluation report dictionary.
    """
    logger.info(
        "Starting DSPy Program Evaluation",
        extra={"program": program_name, "num_examples": len(devset)},
    )

    # Initialize official DSPy Evaluator
    evaluator = Evaluate(
        devset=devset,
        metric=metric_fn,
        num_threads=1,
        display_progress=False,
        display_table=False,
    )

    total_composite_score = evaluator(program)

    # Detailed per-example metric analysis
    per_example_results = []
    rel_scores = []
    comp_scores = []
    struct_scores = []
    supp_scores = []

    res_fields = ["summary", "key_findings", "confidence", "sources"]
    for idx, example in enumerate(devset):
        question = getattr(example, "question", "")
        pred = program(question=question)

        rel = compute_relevance_score(example, pred)
        comp = compute_completeness_score(pred, res_fields)
        struct = compute_structural_correctness(pred)
        supp = compute_support_quality(pred)
        composite = metric_fn(example, pred)

        rel_scores.append(rel)
        comp_scores.append(comp)
        struct_scores.append(struct)
        supp_scores.append(supp)

        per_example_results.append(
            {
                "id": idx + 1,
                "question": question,
                "relevance": round(rel, 4),
                "completeness": round(comp, 4),
                "structural_correctness": round(struct, 4),
                "support_quality": round(supp, 4),
                "composite_score": round(composite, 4),
            }
        )

    num_ex = max(len(devset), 1)
    raw_score = float(total_composite_score)
    overall_score = round(raw_score / 100.0 if raw_score > 1.0 else raw_score, 4)

    report = {
        "program_name": program_name,
        "num_eval_examples": len(devset),
        "overall_composite_score": overall_score,
        "metric_breakdown": {
            "mean_relevance": round(sum(rel_scores) / num_ex, 4),
            "mean_completeness": round(sum(comp_scores) / num_ex, 4),
            "mean_structural_correctness": round(sum(struct_scores) / num_ex, 4),
            "mean_support_quality": round(sum(supp_scores) / num_ex, 4),
        },
        "per_example_results": per_example_results,
    }

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved evaluation report to JSON", extra={"path": str(output_json_path)})

    return report
