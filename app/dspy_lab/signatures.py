"""DSPy Signatures for Researcher and Critic programs."""

import dspy


class ResearchSignature(dspy.Signature):
    """Transform a technical research question into a structured, evidence-backed answer."""

    question: str = dspy.InputField(desc="Technical research question or prompt")
    topic: str = dspy.OutputField(desc="Core technical topic classification")
    summary: str = dspy.OutputField(desc="Comprehensive technical summary of findings")
    key_findings: list[str] = dspy.OutputField(desc="List of key takeaway findings")
    confidence: float = dspy.OutputField(desc="Overall confidence rating between 0.0 and 1.0")
    sources: list[str] = dspy.OutputField(desc="List of reference sources or citations")


class CriticSignature(dspy.Signature):
    """Critique a research output and evaluate technical quality and correctness."""

    research_result: str = dspy.InputField(desc="Structured research output text to evaluate")
    strengths: list[str] = dspy.OutputField(desc="List of identified strengths in research")
    weaknesses: list[str] = dspy.OutputField(desc="List of identified weaknesses or gaps")
    factual_concerns: list[str] = dspy.OutputField(desc="List of potential factual errors")
    quality_score: float = dspy.OutputField(desc="Overall quality rating score between 0.0 and 1.0")
    improvement_suggestions: list[str] = dspy.OutputField(
        desc="Actionable improvement recommendations"
    )
