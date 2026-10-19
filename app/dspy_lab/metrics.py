"""Evaluation metrics for Researcher and Critic DSPy programs."""

from typing import Any

import dspy

from app.logging_config import get_logger

logger = get_logger("app.dspy_lab.metrics")


def compute_relevance_score(example: dspy.Example, pred: Any) -> float:
    """Measure relevance of prediction against expected topic and query keywords."""
    question = getattr(example, "question", "").lower()
    expected_keywords = getattr(example, "expected_keywords", [])

    pred_text = str(getattr(pred, "summary", getattr(pred, "answer", str(pred)))).lower()

    if not pred_text:
        return 0.0

    hits = 0.0
    total_checks = max(len(expected_keywords), 1)

    for kw in expected_keywords:
        if kw.lower() in pred_text:
            hits += 1

    # Bonus if core question terms are present
    q_words = [w for w in question.split() if len(w) > 4]
    if q_words and any(w in pred_text for w in q_words):
        hits += 0.5

    return min(1.0, (hits / total_checks))


def compute_completeness_score(pred: Any, required_fields: list[str]) -> float:
    """Measure structural completeness of output fields."""
    present_fields = 0
    total_fields = len(required_fields)

    for field in required_fields:
        val = getattr(pred, field, None)
        if val is not None:
            if isinstance(val, (list, tuple, str)) and len(val) > 0:
                present_fields += 1
            elif isinstance(val, (int, float)):
                present_fields += 1

    return present_fields / total_fields if total_fields > 0 else 1.0


def compute_structural_correctness(pred: Any) -> float:
    """Measure structural correctness, valid data types, and numerical bounds."""
    score = 1.0

    # Quality score bound check (if present)
    quality_score = getattr(pred, "quality_score", None)
    if quality_score is not None:
        if isinstance(quality_score, (int, float)):
            if not (0.0 <= quality_score <= 1.0):
                score -= 0.5
        else:
            score -= 0.5

    # Confidence bound check (if present)
    confidence = getattr(pred, "confidence", None)
    if confidence is not None:
        if isinstance(confidence, (int, float)):
            if not (0.0 <= confidence <= 1.0):
                score -= 0.5
        else:
            score -= 0.5

    # List structure checks
    check_fields = [
        "key_findings",
        "sources",
        "strengths",
        "weaknesses",
        "improvement_suggestions",
    ]
    for list_field in check_fields:
        val = getattr(pred, list_field, None)
        if val is not None and not isinstance(val, (list, tuple)):
            score -= 0.2

    return max(0.0, score)


def compute_support_quality(pred: Any) -> float:
    """Measure support quality, source citations, and factual density."""
    sources = getattr(pred, "sources", [])
    findings = getattr(pred, "key_findings", [])
    summary = getattr(pred, "summary", "")

    score = 0.5  # Base score

    # Check for non-empty sources/citations
    if sources and isinstance(sources, list) and len(sources) > 0:
        score += 0.3

    # Check for empirical/numerical density in summary or findings
    full_text = str(summary) + " ".join(str(f) for f in findings)
    if any(char.isdigit() for char in full_text):
        score += 0.2

    return min(1.0, score)


def research_eval_metric(example: dspy.Example, pred: Any, trace: Any = None) -> float:
    """Composite metric for evaluating Researcher program outputs.

    Combines:
    - Relevance (40%)
    - Completeness (30%)
    - Structural Correctness (15%)
    - Factual/Support Quality (15%)
    """
    rel = compute_relevance_score(example, pred)
    comp = compute_completeness_score(pred, ["summary", "key_findings", "confidence", "sources"])
    struct = compute_structural_correctness(pred)
    supp = compute_support_quality(pred)

    composite = (0.40 * rel) + (0.30 * comp) + (0.15 * struct) + (0.15 * supp)

    logger.debug(
        "Evaluated research_eval_metric",
        extra={"rel": rel, "comp": comp, "struct": struct, "supp": supp, "composite": composite},
    )

    return composite


def critic_eval_metric(example: dspy.Example, pred: Any, trace: Any = None) -> float:
    """Composite metric for evaluating Critic program outputs.

    Combines:
    - Completeness (40%)
    - Structural Correctness (30%)
    - Support Quality (30%)
    """
    critic_fields = [
        "strengths",
        "weaknesses",
        "factual_concerns",
        "quality_score",
        "improvement_suggestions",
    ]
    comp = compute_completeness_score(pred, critic_fields)
    struct = compute_structural_correctness(pred)
    supp = compute_support_quality(pred)

    composite = (0.40 * comp) + (0.30 * struct) + (0.30 * supp)
    return composite
