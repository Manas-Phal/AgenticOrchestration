# Agent Evaluation System & Benchmark Framework

This directory contains the production-grade evaluation benchmark framework comparing the **Baseline (Unoptimized Zero-Shot)** configuration against the **Optimized (DSPy Few-Shot Program)** configuration across 12 quantitative metrics.

---

## 🏗️ Architectural Framework Boundaries

The evaluation framework preserves the strict architectural boundaries of the system:

* **LangGraph**: Workflow orchestration, state graph topology, Quality Gate routing, loop counters, and human-in-the-loop escalation.
* **Pydantic / PydanticAI**: Schema validation layer enforcing 100% type safety and explicit conversion boundaries.
* **DSPy**: Declarative signatures, modules, evaluation metrics, and prompt optimization (`LabeledFewShot`).

---

## 📂 Dataset Split & Methodology

The evaluation dataset ([`evaluation/dataset.jsonl`](file:///c:/Users/manas/OneDrive/Documents/RAGprojArchestration/evaluation/dataset.jsonl)) contains 25 realistic technical research queries across 10 domain categories (e.g., Microservices, Consensus Protocols, Zero-Trust Security, Event Streaming, Container Orchestration).

To prevent evaluation leakage, the dataset is split into three distinct sets:

1. **Train Set (10 examples)**: Used exclusively for training the DSPy optimizer (`LabeledFewShot` / `BootstrapFewShot`).
2. **Development Set (5 examples)**: Used for hyperparameter validation, prompt tuning, and metric calibration.
3. **Test Set (10 examples)**: Held out strictly for final evaluation comparison between Baseline and Optimized configurations.

---

## 📊 Metric Definitions & Classification

The system evaluates runs across 12 quantitative metrics categorized by evaluation type:

### 1. Objective / Deterministic Metrics (Code-Measured & Schema-Verified)
* **`structured_output_validity`**: Percentage of runs generating 100% valid `FinalResponse` domain models without schema conversion errors.
* **`citation_source_coverage`**: Ratio of validated source citations (`http` URIs and reliability scores).
* **`retry_rate`**: Mean count of transient network or node retries per execution.
* **`validation_failure_rate`**: Percentage of runs encountering Pydantic schema validation failures.
* **`average_latency_ms`**: Mean end-to-end execution duration measured in milliseconds.
* **`iteration_count`**: Mean graph loop iterations executed to satisfy Quality Gate threshold ($\ge 0.8$).
* **`human_escalation_rate`**: Ratio of runs hitting iteration limit requiring human-in-the-loop interrupt.
* **`estimated_token_usage_cost`**: Total token consumption and estimated USD cost.

### 2. Subjective / LLM-as-a-Judge & Hybrid Metrics
* **`research_relevance`**: Ratio of expected domain keywords present in synthesized research summary and key findings.
* **`completeness`**: Ratio of required structural fields present and non-empty (summary, key findings, sources, critique).
* **`critic_quality`**: Granularity of critique strengths/weaknesses/suggestions and score calibration.
* **`final_answer_quality`**: Composite quality metric evaluating final answer synthesis length and Quality Gate score.

---

## 📈 Measured Baseline vs Optimized Results

Below are the actual measured evaluation results executed on the test dataset split:

| Metric | Baseline | Optimized | Difference | Classification |
| :--- | :---: | :---: | :---: | :---: |
| `structured_output_validity` | `1.0000` | `1.0000` | `+0.0000` | Objective |
| `research_relevance` | `0.0000` | `0.0800` | `+0.0800` | Subjective / Hybrid |
| `completeness` | `1.0000` | `1.0000` | `+0.0000` | Objective |
| `citation_source_coverage` | `1.0000` | `1.0000` | `+0.0000` | Objective |
| `critic_quality` | `1.0000` | `1.0000` | `+0.0000` | Subjective / Hybrid |
| `final_answer_quality` | `0.9100` | `0.9900` | `+0.0800` | Subjective / Hybrid |
| `retry_rate` | `0.0000` | `0.0000` | `+0.0000` | Objective |
| `validation_failure_rate` | `0.0000` | `0.0000` | `+0.0000` | Objective |
| `average_latency_ms` | `1404.5590` | `1334.3300` | `-70.2290` | Objective |
| `iteration_count` | `2.0000` | `2.0000` | `+0.0000` | Objective |
| `human_escalation_rate` | `0.0000` | `0.0000` | `+0.0000` | Objective |
| `estimated_token_usage_cost` | `0.0000` | `0.0000` | `+0.0000` | Objective |

*Note: All numbers reported above originate directly from runtime evaluation artifacts saved in [`evaluation/results/`](file:///c:/Users/manas/OneDrive/Documents/RAGprojArchestration/evaluation/results).*

---

## 🔄 Reproducibility Instructions

To re-run the evaluation pipeline and regenerate the benchmark report:

```bash
# 1. Activate virtual environment
.venv\Scripts\activate

# 2. Run full evaluation script
python -m evaluation.evaluate

# 3. Inspect generated results artifacts
# - evaluation/results/baseline_results.json
# - evaluation/results/optimized_results.json
# - evaluation/results/comparison_summary.json
# - evaluation/results/comparison_report.csv
```
