"""Evaluation metrics computation module for baseline vs optimized system benchmarks."""

from typing import Any

from app.schemas.domain import FinalResponse, QualityAssessment, ResearchResult


def compute_structured_output_validity(run_result: dict[str, Any]) -> float:
    """Measure whether graph output generated a valid FinalResponse object (Objective)."""
    final_resp = run_result.get("final_response")
    if isinstance(final_resp, FinalResponse):
        return 1.0
    elif isinstance(final_resp, dict) and "final_answer" in final_resp:
        return 1.0
    return 0.0


def compute_research_relevance(run_result: dict[str, Any], expected_keywords: list[str]) -> float:
    """Measure keyword presence and topic alignment ratio in research results (Objective)."""
    final_resp = run_result.get("final_response")
    if not final_resp:
        return 0.0

    if isinstance(final_resp, FinalResponse):
        summary = final_resp.research_result.summary
        findings = " ".join(final_resp.research_result.key_findings)
    else:
        research = run_result.get("research_result", {})
        if isinstance(research, ResearchResult):
            summary = research.summary
            findings = " ".join(research.key_findings)
        elif isinstance(research, dict):
            summary = research.get("summary", "")
            findings = " ".join(research.get("key_findings", []))
        else:
            summary, findings = "", ""

    full_text = (summary + " " + findings).lower()
    if not full_text or not expected_keywords:
        return 0.5  # Default baseline alignment

    hits = sum(1 for kw in expected_keywords if kw.lower() in full_text)
    return round(hits / len(expected_keywords), 4)


def compute_completeness(run_result: dict[str, Any]) -> float:
    """Measure presence and density of required structural output fields (Objective)."""
    final_resp = run_result.get("final_response")
    if not final_resp:
        return 0.0

    required_checks = 0
    total_checks = 5

    if isinstance(final_resp, FinalResponse):
        if len(final_resp.final_answer.strip()) >= 10:
            required_checks += 1
        if final_resp.plan and len(final_resp.plan.steps) > 0:
            required_checks += 1
        if final_resp.research_result and len(final_resp.research_result.summary.strip()) >= 10:
            required_checks += 1
        if final_resp.research_result and len(final_resp.research_result.sources) > 0:
            required_checks += 1
        if final_resp.quality_assessment and final_resp.quality_assessment.quality_score >= 0.0:
            required_checks += 1
    elif isinstance(final_resp, dict):
        if len(str(final_resp.get("final_answer", "")).strip()) >= 10:
            required_checks += 1
        if final_resp.get("plan"):
            required_checks += 1
        if final_resp.get("research_result"):
            required_checks += 1
        if final_resp.get("quality_assessment"):
            required_checks += 2

    return round(required_checks / total_checks, 4)


def compute_citation_source_coverage(run_result: dict[str, Any]) -> float:
    """Measure quantity and validity of cited sources (Objective)."""
    final_resp = run_result.get("final_response")
    if not final_resp:
        return 0.0

    sources = []
    if isinstance(final_resp, FinalResponse):
        sources = final_resp.research_result.sources
    elif isinstance(final_resp, dict):
        res = final_resp.get("research_result", {})
        if isinstance(res, dict):
            sources = res.get("sources", [])

    if not sources:
        return 0.0

    valid_sources = 0
    for s in sources:
        url_str = getattr(s, "url", s.get("url", "") if isinstance(s, dict) else "")
        if url_str.startswith("http"):
            valid_sources += 1

    coverage = min(1.0, (len(sources) * 0.3) + (valid_sources / len(sources) * 0.7))
    return round(coverage, 4)


def compute_critic_quality(run_result: dict[str, Any]) -> float:
    """Measure completeness and granularity of critic evaluation output (Hybrid/Objective)."""
    final_resp = run_result.get("final_response")
    qa = None

    if isinstance(final_resp, FinalResponse):
        qa = final_resp.quality_assessment
    elif isinstance(run_result.get("quality_assessment"), QualityAssessment):
        qa = run_result.get("quality_assessment")

    if not qa:
        return 0.5

    critique = getattr(qa, "critique", None)
    if not critique:
        return 0.5

    strengths = getattr(critique, "strengths", [])
    weaknesses = getattr(critique, "weaknesses", [])
    suggestions = getattr(critique, "improvement_suggestions", [])

    items_count = len(strengths) + len(weaknesses) + len(suggestions)
    score_bounded = 0.0 <= qa.quality_score <= 1.0

    quality = min(1.0, (items_count * 0.2) + (0.5 if score_bounded else 0.0))
    return round(quality, 4)


def compute_final_answer_quality(run_result: dict[str, Any]) -> float:
    """Measure structural completeness and length quality of final synthesized answer (Hybrid)."""
    final_resp = run_result.get("final_response")
    if not final_resp:
        return 0.0

    if isinstance(final_resp, FinalResponse):
        answer = final_resp.final_answer
        score = final_resp.quality_assessment.quality_score
    elif isinstance(final_resp, dict):
        answer = str(final_resp.get("final_answer", ""))
        qa = final_resp.get("quality_assessment", {})
        score = qa.get("quality_score", 0.8) if isinstance(qa, dict) else 0.8
    else:
        answer, score = "", 0.0

    if len(answer.strip()) < 10:
        return 0.0

    length_score = min(1.0, len(answer) / 100.0)
    composite = round((length_score * 0.4) + (score * 0.6), 4)
    return composite


def evaluate_single_run(run_result: dict[str, Any], expected_keywords: list[str]) -> dict[str, Any]:
    """Calculate all 12 metrics for a single execution run."""
    final_resp = run_result.get("final_response")

    # Extract objective operational metrics directly from state
    latency_ms = run_result.get("total_duration_ms", 0.0)
    if latency_ms == 0.0 and isinstance(final_resp, FinalResponse):
        latency_ms = final_resp.execution_time_ms

    iteration = run_result.get("iteration", 1)
    if iteration == 0 and isinstance(final_resp, FinalResponse):
        iteration = final_resp.total_iterations

    retries = run_result.get("retry_count", 0)
    val_failures = run_result.get("validation_failures", 0)
    status = run_result.get("status", "completed")
    escalated_statuses = ("human_review", "human_approved", "human_edited", "rejected")
    human_escalated = 1.0 if status in escalated_statuses else 0.0

    token_dict = run_result.get("token_usage", {"total_tokens": 150})
    total_tokens = token_dict.get("total_tokens", 150)
    est_cost = run_result.get("estimated_cost", total_tokens * 0.000002)

    metrics = {
        "structured_output_validity": compute_structured_output_validity(run_result),
        "research_relevance": compute_research_relevance(run_result, expected_keywords),
        "completeness": compute_completeness(run_result),
        "citation_source_coverage": compute_citation_source_coverage(run_result),
        "critic_quality": compute_critic_quality(run_result),
        "final_answer_quality": compute_final_answer_quality(run_result),
        "retry_rate": float(retries),
        "validation_failure_rate": 1.0 if val_failures > 0 else 0.0,
        "average_latency_ms": round(float(latency_ms), 2),
        "iteration_count": float(iteration),
        "human_escalation_rate": human_escalated,
        "estimated_token_usage_cost": round(float(est_cost), 6),
        "total_tokens": int(total_tokens),
    }

    return metrics


def aggregate_metrics(run_metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean averages across a list of evaluation run metric dicts."""
    if not run_metrics_list:
        return {}

    metric_keys = [
        "structured_output_validity",
        "research_relevance",
        "completeness",
        "citation_source_coverage",
        "critic_quality",
        "final_answer_quality",
        "retry_rate",
        "validation_failure_rate",
        "average_latency_ms",
        "iteration_count",
        "human_escalation_rate",
        "estimated_token_usage_cost",
    ]

    aggregated: dict[str, float] = {}
    total_runs = len(run_metrics_list)

    for key in metric_keys:
        total_val = sum(m.get(key, 0.0) for m in run_metrics_list)
        aggregated[key] = round(total_val / total_runs, 4)

    return aggregated
