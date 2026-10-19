"""FastAPI application server exposing orchestration endpoints and interactive UI."""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.exceptions import OrchestrationBaseError
from app.core.health import HealthCheckReport, check_system_health
from app.logging_config import get_logger, setup_logging
from app.orchestration.runner import OrchestrationEngine
from app.schemas.domain import HumanReviewDecision

logger = get_logger("app.server")

settings = get_settings()
setup_logging(log_level=settings.log_level, json_format=settings.log_json)

app = FastAPI(
    title=settings.app_name,
    description=(
        "FastAPI interface for Production-Grade Agentic Orchestration System "
        "(LangGraph + PydanticAI + DSPy)."
    ),
    version="1.0.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine singleton instance
engine = OrchestrationEngine()


class StartRequestPayload(BaseModel):
    """Payload model for starting a new orchestration request."""

    user_query: str = Field(..., min_length=1, description="Research query or topic prompt")
    thread_id: str | None = Field(default=None, description="Optional thread identifier")
    max_iterations: int | None = Field(
        default=None, ge=1, le=10, description="Max iteration limit"
    )
    quality_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Quality gate threshold"
    )


class ResumeRequestPayload(BaseModel):
    """Payload model for resuming an interrupted workflow thread."""

    thread_id: str = Field(..., min_length=1, description="Target interrupted thread ID")
    decision: HumanReviewDecision = Field(..., description="Human review decision payload")


@app.get("/health", response_model=HealthCheckReport)
async def health_check() -> HealthCheckReport:
    """System health check endpoint returning subsystem readiness report."""
    return check_system_health()


@app.post("/api/orchestration/start")
async def start_orchestration(payload: StartRequestPayload) -> dict[str, Any]:
    """Start a new LangGraph orchestration request thread."""
    try:
        result = engine.start_request(
            user_query=payload.user_query,
            thread_id=payload.thread_id,
            max_iterations=payload.max_iterations,
            quality_threshold=payload.quality_threshold,
        )
        return sanitize_orchestration_result(result)
    except OrchestrationBaseError as err:
        logger.error("Orchestration error during start_request", extra={"error": str(err)})
        raise HTTPException(status_code=400, detail=err.to_dict()) from err
    except Exception as err:
        logger.error("Unexpected exception during start_request", extra={"error": str(err)})
        raise HTTPException(status_code=500, detail={"error": str(err)}) from err


@app.post("/api/orchestration/resume")
async def resume_orchestration(payload: ResumeRequestPayload) -> dict[str, Any]:
    """Resume an interrupted workflow thread following a human decision."""
    try:
        result = engine.resume_request(
            thread_id=payload.thread_id,
            decision=payload.decision,
        )
        return sanitize_orchestration_result(result)
    except OrchestrationBaseError as err:
        logger.error("Orchestration error during resume_request", extra={"error": str(err)})
        raise HTTPException(status_code=400, detail=err.to_dict()) from err
    except Exception as err:
        logger.error("Unexpected exception during resume_request", extra={"error": str(err)})
        raise HTTPException(status_code=500, detail={"error": str(err)}) from err


@app.get("/api/orchestration/state/{thread_id}")
async def get_thread_state(thread_id: str) -> dict[str, Any]:
    """Fetch current state snapshot for a given workflow thread ID."""
    state = engine.get_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    return sanitize_orchestration_result(dict(state))


@app.get("/api/evaluation/summary")
async def get_evaluation_summary() -> dict[str, Any]:
    """Return benchmark evaluation summary results comparing Baseline vs DSPy Optimized."""
    results_dir = Path(__file__).parent.parent / "evaluation" / "results"
    summary_path = results_dir / "comparison_summary.json"
    if not summary_path.exists():
        return {
            "status": "not_found",
            "message": (
                "Evaluation results summary not generated yet. "
                "Run 'python -m evaluation.evaluate'."
            ),
        }
    with open(summary_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def sanitize_orchestration_result(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure result dict is JSON-serializable by converting Pydantic models to dicts."""
    output: dict[str, Any] = {}
    for key, val in result.items():
        if hasattr(val, "model_dump"):
            output[key] = val.model_dump()
        elif isinstance(val, list):
            output[key] = [
                item.model_dump() if hasattr(item, "model_dump") else item for item in val
            ]
        else:
            output[key] = val
    return output


# Serve static web frontend
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_web_ui() -> HTMLResponse:
    """Serve single-page interactive web UI frontend."""
    index_html_path = STATIC_DIR / "index.html"
    if index_html_path.exists():
        with open(index_html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Agentic Orchestration System Backend Active</h1>")
