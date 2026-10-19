"""Integration test suite for FastAPI application server and REST endpoints."""

from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test /health endpoint returns 200 OK with HealthCheckReport JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    assert "components" in data
    assert "app_name" in data


def test_serve_web_ui() -> None:
    """Test GET / serves the interactive single-page application frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Agentic Orchestration Control Center" in response.text


def test_start_orchestration_api() -> None:
    """Test POST /api/orchestration/start executes workflow thread and returns result."""
    payload = {
        "user_query": "instant_pass: API Integration Test Request",
        "quality_threshold": 0.80,
        "max_iterations": 3,
    }
    response = client.post("/api/orchestration/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert "final_response" in data
    assert data["final_response"]["status"] == "completed"


def test_resume_orchestration_api() -> None:
    """Test POST /api/orchestration/resume handles human-in-the-loop decisions."""
    # 1. Start thread that hits human review interrupt
    start_payload = {
        "user_query": "always_fail: API Interrupt Request",
        "thread_id": "thread-api-interrupt-101",
        "max_iterations": 2,
    }
    start_res = client.post("/api/orchestration/start", json=start_payload)
    assert start_res.status_code == 200
    assert "__interrupt__" in start_res.json()

    # 2. Resume with approval decision
    resume_payload = {
        "thread_id": "thread-api-interrupt-101",
        "decision": {
            "action": "approve",
            "reviewer_id": "test_operator",
        },
    }
    resume_res = client.post("/api/orchestration/resume", json=resume_payload)
    assert resume_res.status_code == 200
    data = resume_res.json()
    assert data["status"] == "human_approved"


def test_get_thread_state_api() -> None:
    """Test GET /api/orchestration/state/{thread_id} returns state snapshot."""
    # Start request first
    tid = "thread-state-api-202"
    client.post(
        "/api/orchestration/start",
        json={"user_query": "instant_pass: State Fetch Request", "thread_id": tid},
    )

    response = client.get(f"/api/orchestration/state/{tid}")
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] is not None
    assert data["status"] == "completed"


def test_evaluation_summary_api() -> None:
    """Test GET /api/evaluation/summary returns benchmark metrics summary."""
    response = client.get("/api/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data or "status" in data
