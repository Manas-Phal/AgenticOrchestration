/* Agentic Orchestration Client Logic */

let currentThreadId = null;

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    loadEvaluationSummary();
});

function setQuery(text) {
    document.getElementById("query-input").value = text;
}

async function checkHealth() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        
        const dot = document.getElementById("system-status-dot");
        const text = document.getElementById("system-status-text");
        
        if (data.status === "healthy") {
            dot.style.background = "var(--emerald)";
            dot.style.boxShadow = "0 0 10px var(--emerald)";
            text.innerText = `System Healthy (${data.environment})`;
        } else {
            dot.style.background = "var(--amber)";
            text.innerText = `System ${data.status}`;
        }
    } catch (err) {
        console.error("Health check failed:", err);
    }
}

async function startOrchestration() {
    const query = document.getElementById("query-input").value.trim();
    if (!query) {
        alert("Please enter a research query prompt.");
        return;
    }

    const threshold = parseFloat(document.getElementById("threshold-slider").value);
    const maxIter = parseInt(document.getElementById("iterations-slider").value, 10);
    const btn = document.getElementById("btn-submit-request");

    btn.disabled = true;
    btn.innerHTML = "<span>⏳ Executing Graph Nodes...</span>";
    resetNodeHighlighting();
    highlightNode("node-planner", "active");

    document.getElementById("human-review-portal").classList.add("hidden");
    document.getElementById("res-status").innerText = "executing";

    try {
        const response = await fetch("/api/orchestration/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_query: query,
                quality_threshold: threshold,
                max_iterations: maxIter,
            }),
        });

        const data = await response.json();
        btn.disabled = false;
        btn.innerHTML = "<span>🚀 Launch Agent Workflow</span>";

        if (response.ok) {
            renderOrchestrationResult(data);
        } else {
            alert(`Execution failed: ${JSON.stringify(data.detail)}`);
        }
    } catch (err) {
        btn.disabled = false;
        btn.innerHTML = "<span>🚀 Launch Agent Workflow</span>";
        console.error("Request failed:", err);
        alert("Execution request failed. Check server log.");
    }
}

function renderOrchestrationResult(data) {
    currentThreadId = data.request_id || "thread-active";
    document.getElementById("active-thread-id").innerText = `Thread: ${currentThreadId}`;
    document.getElementById("res-status").innerText = data.status || "completed";

    // Handle Interrupted Graph State (Human Review Node)
    if (data.__interrupt__ && data.__interrupt__.length > 0) {
        const interruptInfo = data.__interrupt__[0].value;
        highlightNode("node-quality-gate", "interrupted");
        
        const portal = document.getElementById("human-review-portal");
        portal.classList.remove("hidden");
        document.getElementById("human-reason-text").innerText = interruptInfo.reason || "Iteration limit reached. Human review required.";
        
        document.getElementById("res-score").innerText = (interruptInfo.quality_score || 0.50).toFixed(2);
        document.getElementById("res-iterations").innerText = interruptInfo.iteration || 2;
        return;
    }

    // Graph Completed or Resumed
    resetNodeHighlighting();
    highlightNode("node-planner", "completed");
    highlightNode("node-research", "completed");
    highlightNode("node-validation1", "completed");
    highlightNode("node-critic", "completed");
    highlightNode("node-quality-gate", "completed");
    highlightNode("node-finalizer", "completed");

    const finalResp = data.final_response || {};
    document.getElementById("res-final-answer").innerText = finalResp.final_answer || data.final_answer || "Completed.";

    const qa = finalResp.quality_assessment || data.quality_assessment || {};
    document.getElementById("res-score").innerText = (qa.quality_score || 0.85).toFixed(2);
    document.getElementById("res-iterations").innerText = data.total_iterations || data.iteration || 1;
    document.getElementById("res-latency").innerText = `${Math.round(data.total_duration_ms || finalResp.execution_time_ms || 120)} ms`;

    // Render Execution Plan Steps
    const planObj = finalResp.plan || data.plan || {};
    const steps = planObj.steps || [];
    const planContainer = document.getElementById("res-plan-steps");
    planContainer.innerHTML = steps.map((s, idx) => `
        <div class="plan-item">
            <span style="color: var(--cyan); font-weight: 600;">Step ${idx + 1}:</span>
            <span>${s}</span>
        </div>
    `).join("");

    // Render Research Findings & Sources
    const researchObj = finalResp.research_result || data.research_result || {};
    const findings = researchObj.key_findings || [];
    const findingsContainer = document.getElementById("res-research-findings");
    findingsContainer.innerHTML = findings.map(f => `
        <div class="plan-item">
            <span style="color: var(--emerald);">✓</span>
            <span>${f}</span>
        </div>
    `).join("");

    const sources = researchObj.sources || [];
    const sourcesContainer = document.getElementById("res-sources-list");
    sourcesContainer.innerHTML = sources.map(src => `
        <div class="source-item">
            <a href="${src.url}" target="_blank" class="source-link">🔗 ${src.title}</a>
            <span class="source-reliability">Reliability: ${Math.round((src.reliability_score || 0.9) * 100)}%</span>
        </div>
    `).join("");
}

async function resumeHumanDecision(action) {
    if (!currentThreadId) {
        alert("No active interrupted thread found.");
        return;
    }

    try {
        const response = await fetch("/api/orchestration/resume", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                thread_id: currentThreadId,
                decision: { action: action, reviewer_id: "human_operator" },
            }),
        });

        const data = await response.json();
        document.getElementById("human-review-portal").classList.add("hidden");
        if (response.ok) {
            renderOrchestrationResult(data);
        } else {
            alert(`Resume failed: ${JSON.stringify(data.detail)}`);
        }
    } catch (err) {
        console.error("Resume failed:", err);
    }
}

function showEditPortal() {
    document.getElementById("edit-portal-inputs").classList.remove("hidden");
}

async function submitHumanEdit() {
    const rawSteps = document.getElementById("edit-steps-input").value;
    const scoreVal = parseFloat(document.getElementById("edit-score-input").value);
    const stepsList = rawSteps.split(",").map(s => s.trim()).filter(s => s.length > 0);

    try {
        const response = await fetch("/api/orchestration/resume", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                thread_id: currentThreadId,
                decision: {
                    action: "edit",
                    reviewer_id: "lead_architect",
                    edited_plan: stepsList,
                    edited_quality_score: scoreVal,
                },
            }),
        });

        const data = await response.json();
        document.getElementById("human-review-portal").classList.add("hidden");
        if (response.ok) {
            renderOrchestrationResult(data);
        } else {
            alert(`Submit edit failed: ${JSON.stringify(data.detail)}`);
        }
    } catch (err) {
        console.error("Submit edit failed:", err);
    }
}

async function loadEvaluationSummary() {
    try {
        const res = await fetch("/api/evaluation/summary");
        const data = await res.json();
        const tbody = document.getElementById("benchmark-table-body");

        if (data.comparison) {
            tbody.innerHTML = Object.entries(data.comparison).map(([metric, values]) => {
                const diff = values.difference;
                const diffClass = diff > 0 ? "diff-positive" : diff < 0 ? "diff-negative" : "";
                const typeTag = ["structured_output_validity", "completeness", "citation_source_coverage", "retry_rate", "average_latency_ms", "iteration_count"].includes(metric) ? "Objective" : "Subjective";
                
                return `
                    <tr>
                        <td style="font-family: var(--font-mono); color: var(--text-main);">${metric}</td>
                        <td>${values.baseline.toFixed(4)}</td>
                        <td style="font-weight: 600; color: var(--emerald);">${values.optimized.toFixed(4)}</td>
                        <td class="${diffClass}">${diff >= 0 ? "+" + diff.toFixed(4) : diff.toFixed(4)}</td>
                        <td style="color: var(--text-dim); font-size: 11px;">${typeTag}</td>
                    </tr>
                `;
            }).join("");
        }
    } catch (err) {
        console.error("Failed to load evaluation summary:", err);
    }
}

function resetNodeHighlighting() {
    const nodes = ["node-planner", "node-research", "node-validation1", "node-critic", "node-quality-gate", "node-finalizer"];
    nodes.forEach(n => {
        const el = document.getElementById(n);
        if (el) el.className = "flow-node";
    });
}

function highlightNode(nodeId, stateClass) {
    const el = document.getElementById(nodeId);
    if (el) el.classList.add(`node-${stateClass}`);
}
