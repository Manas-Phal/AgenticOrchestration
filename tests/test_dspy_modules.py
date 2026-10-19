"""Unit test suite for Phase 3 DSPy signatures, modules, metrics, dataset, and optimizer."""

from pathlib import Path

from app.dspy_lab import (
    CriticModule,
    CriticSignature,
    ResearcherModule,
    ResearchSignature,
    compute_completeness_score,
    compute_relevance_score,
    compute_structural_correctness,
    compute_support_quality,
    configure_dspy_lm,
    critic_eval_metric,
    evaluate_program,
    load_dataset,
    research_eval_metric,
    run_optimization_pipeline,
)


def test_signatures_structure() -> None:
    """Test ResearchSignature and CriticSignature input and output field definitions."""
    res_inputs = ResearchSignature.input_fields
    res_outputs = ResearchSignature.output_fields

    assert "question" in res_inputs
    assert "summary" in res_outputs
    assert "key_findings" in res_outputs
    assert "confidence" in res_outputs
    assert "sources" in res_outputs

    critic_inputs = CriticSignature.input_fields
    critic_outputs = CriticSignature.output_fields

    assert "research_result" in critic_inputs
    assert "strengths" in critic_outputs
    assert "weaknesses" in critic_outputs
    assert "factual_concerns" in critic_outputs
    assert "quality_score" in critic_outputs
    assert "improvement_suggestions" in critic_outputs


def test_researcher_module_execution() -> None:
    """Test ResearcherModule forward execution with MockDSPyLM."""
    configure_dspy_lm()
    module = ResearcherModule()

    pred = module(question="What is Raft consensus?")
    assert hasattr(pred, "summary") or hasattr(pred, "topic")
    assert getattr(pred, "confidence", 0.0) >= 0.0


def test_critic_module_execution() -> None:
    """Test CriticModule forward execution with MockDSPyLM."""
    configure_dspy_lm()
    module = CriticModule()

    pred = module(research_result="Raft is a consensus algorithm with leader election.")
    assert hasattr(pred, "strengths")
    assert hasattr(pred, "weaknesses")
    assert hasattr(pred, "quality_score")
    assert 0.0 <= getattr(pred, "quality_score", 0.0) <= 1.0


def test_dataset_loading() -> None:
    """Test load_dataset loads at least 20 examples split into trainset (12) and devset (8)."""
    trainset, devset = load_dataset()

    total_examples = len(trainset) + len(devset)
    assert total_examples >= 20
    assert len(trainset) == 12
    assert len(devset) == 8

    # Verify input keys
    example = trainset[0]
    assert example.inputs().question is not None


def test_metrics_calculation() -> None:
    """Test metric computation functions on mock predictions."""
    trainset, _ = load_dataset()
    example = trainset[0]

    configure_dspy_lm()
    module = ResearcherModule()
    pred = module(question=example.question)

    rel = compute_relevance_score(example, pred)
    comp = compute_completeness_score(pred, ["summary", "key_findings", "confidence", "sources"])
    struct = compute_structural_correctness(pred)
    supp = compute_support_quality(pred)

    assert 0.0 <= rel <= 1.0
    assert 0.0 <= comp <= 1.0
    assert 0.0 <= struct <= 1.0
    assert 0.0 <= supp <= 1.0

    comp_metric = research_eval_metric(example, pred)
    assert 0.0 <= comp_metric <= 1.0

    critic_module = CriticModule()
    critic_pred = critic_module(research_result="Sample research output text.")
    critic_comp = critic_eval_metric(example, critic_pred)
    assert 0.0 <= critic_comp <= 1.0


def test_evaluate_program_runner(tmp_path: Path) -> None:
    """Test evaluate_program benchmarking and JSON output persistence."""
    configure_dspy_lm()
    _, devset = load_dataset()
    module = ResearcherModule()

    output_json = tmp_path / "test_baseline_eval.json"
    report = evaluate_program(
        program=module,
        devset=devset[:3],  # Run on slice for fast unit test
        metric_fn=research_eval_metric,
        program_name="Test Baseline",
        output_json_path=output_json,
    )

    assert report["program_name"] == "Test Baseline"
    assert "overall_composite_score" in report
    assert output_json.exists()


def test_optimization_pipeline(tmp_path: Path) -> None:
    """Test run_optimization_pipeline baseline vs optimized comparison flow."""
    summary = run_optimization_pipeline(output_dir=tmp_path, k_shots=1)

    assert "baseline" in summary
    assert "optimized" in summary
    assert "comparison" in summary
    assert (tmp_path / "evaluation_comparison.json").exists()
