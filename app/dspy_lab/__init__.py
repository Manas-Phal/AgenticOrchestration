"""Phase 3 DSPy Declarative Optimization package."""

from app.dspy_lab.dataset import RAW_DATASET_EXAMPLES, load_dataset
from app.dspy_lab.evaluate import evaluate_program
from app.dspy_lab.metrics import (
    compute_completeness_score,
    compute_relevance_score,
    compute_structural_correctness,
    compute_support_quality,
    critic_eval_metric,
    research_eval_metric,
)
from app.dspy_lab.modules import (
    CriticModule,
    MockDSPyLM,
    ResearcherModule,
    configure_dspy_lm,
)
from app.dspy_lab.optimize import run_optimization_pipeline
from app.dspy_lab.signatures import CriticSignature, ResearchSignature

__all__ = [
    "ResearchSignature",
    "CriticSignature",
    "ResearcherModule",
    "CriticModule",
    "MockDSPyLM",
    "configure_dspy_lm",
    "RAW_DATASET_EXAMPLES",
    "load_dataset",
    "compute_relevance_score",
    "compute_completeness_score",
    "compute_structural_correctness",
    "compute_support_quality",
    "research_eval_metric",
    "critic_eval_metric",
    "evaluate_program",
    "run_optimization_pipeline",
]
