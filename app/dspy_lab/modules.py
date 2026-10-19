"""DSPy Modules for Researcher and Critic programs."""

import json
from typing import Any

import dspy

from app.dspy_lab.signatures import CriticSignature, ResearchSignature
from app.logging_config import get_logger

logger = get_logger("app.dspy_lab.modules")


class MockDSPyLM(dspy.LM):
    """Deterministic mock LM for offline test execution and evaluation without API keys."""

    def __init__(self, model: str = "mock-dspy-model") -> None:
        """Initialize mock LM."""
        super().__init__(model=model)

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate deterministic structured JSON responses matching requested signature schema."""
        prompt_str = str(prompt or messages or "").lower()

        # Check if call is evaluating a Critic signature or Research signature
        is_critic = any(
            k in prompt_str for k in ["strengths", "factual_concerns", "research_result"]
        )
        if is_critic:
            response_payload = {
                "strengths": [
                    "Clear technical overview of core components",
                    "Well-structured explanation of operational mechanics",
                ],
                "weaknesses": [
                    "Could include quantitative latency benchmarks",
                    "Needs more edge-case failure mode discussion",
                ],
                "factual_concerns": [],
                "quality_score": 0.85,
                "improvement_suggestions": [
                    "Add real-world production metrics",
                    "Include architecture diagram references",
                ],
            }
        else:
            response_payload = {
                "topic": "Distributed AI Systems Architecture",
                "summary": (
                    "Comprehensive analysis of distributed system design patterns, "
                    "consensus mechanisms, and scalable state orchestration."
                ),
                "key_findings": [
                    "Consensus protocols ensure state machine replication safety across nodes.",
                    "Explicit state bounds prevent infinite execution cycles in workflow graphs.",
                    "Type-validated schemas guarantee input-output interface contracts.",
                ],
                "confidence": 0.90,
                "sources": [
                    "https://dspy-docs.vercel.app/",
                    "https://langchain-ai.github.io/langgraph/",
                ],
            }

        return [json.dumps(response_payload)]


def configure_dspy_lm(lm: dspy.LM | None = None) -> dspy.LM:
    """Configure active DSPy LM setting, defaulting to MockDSPyLM if none provided."""
    target_lm = lm or MockDSPyLM()
    try:
        dspy.settings.configure(lm=target_lm)
    except RuntimeError:
        # DSPy settings may already be configured by another thread in multi-threaded test runner
        pass
    logger.info("Configured DSPy LM", extra={"model": getattr(target_lm, "model", "custom")})
    return target_lm


class ResearcherModule(dspy.Module):
    """DSPy Researcher Module using ResearchSignature."""

    def __init__(self) -> None:
        """Initialize predictor component."""
        super().__init__()
        self.predictor = dspy.Predict(ResearchSignature)

    def forward(self, question: str) -> dspy.Prediction:
        """Execute research prediction for a given question string."""
        logger.info("Executing ResearcherModule", extra={"question": question})
        return self.predictor(question=question)


class CriticModule(dspy.Module):
    """DSPy Critic Module using CriticSignature."""

    def __init__(self) -> None:
        """Initialize predictor component."""
        super().__init__()
        self.predictor = dspy.Predict(CriticSignature)

    def forward(self, research_result: str) -> dspy.Prediction:
        """Execute critic evaluation for a given research result string."""
        logger.info("Executing CriticModule", extra={"input_length": len(research_result)})
        return self.predictor(research_result=research_result)
