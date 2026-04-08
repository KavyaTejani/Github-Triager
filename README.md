---
title: GitHub Triager RL
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# GitHub Triager OpenEnv Environment

An automated environment for training AI agents to triage GitHub issues. This project follows the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification.

##  Overview
The **GitHub Triager** environment simulates a high-traffic repository where agents must learn to categorize issues, assign priorities, and manage workloads across batches. It is designed to evaluate an agent's ability to handle single-task classification, multi-task trajectory-level decision-making, and multi-turn clarification.

### Why it exists
Maintainer burnout is a critical issue in open-source software. By building a standardized environment for training triage agents, we can develop models that automatically handle the "noisy" first phase of the software lifecycle—filtering spam, identifying critical bugs, and suggesting team assignments.

##  Features
- **4 Tasks of Increasing Difficulty**: Easy (Labeling), Medium (Full Triage), Hard (Batch Triage), and Expert (Clarification).
- **Context-Aware**: Agents receive a `project_map` to make intelligent component and team assignments.
- **WebSocket & HTTP Support**: Fully implemented high-speed WebSocket layer for high-throughput RL training.
- **Scalable Architecture**: Support for Redis-backed session management and multi-worker deployments.
- **Deterministic Grading**: Transparent scoring logic for every task with structured feedback.
- **Type-Safe**: Built with Pydantic v2 for robust data validation.

---

##  Project Structure

```text
GitHub-Triager/
├── server/                     # The "Backend" or "Environment"
│   ├── app.py                  # FastAPI server and API endpoints
│   ├── environment.py          # Core RL logic (Tasks and IssueStore)
│   ├── graders.py              # Deterministic scoring logic
│   ├── session_store.py        # Redis/In-memory session management
│   ├── ws_handler.py           # WebSocket message routing
│   ├── logging_config.py       # Structured JSON logging (structlog)
│   └── Dockerfile              # Production-ready containerization
├── data/                       
│   ├── simulated_issues.json   # Dataset (120 issues with Gold labels)
│   └── project_structure.json  # Map of components to files and teams
├── models.py                   # Pydantic models (Schema definitions)
├── client.py                   # HTTP & WebSocket client wrappers
├── inference.py                # Baseline agent script (Compliant with Eval)
├── openenv.yaml                # OpenEnv specification metadata
├── pyproject.toml              # Dependencies and entry points
└── tests/                      # Full test suite (Environment, API, WS)
```

---

##  Task Definitions

| Task ID | Name | Difficulty | Description |
|---------|------|------------|-------------|
| `label_classification` | Label Classification | Easy | Classify an issue into: bug, feature, documentation, question, enhancement. |
| `full_triage` | Full Triage | Medium | Assign label, priority, team assignee, and component using the `project_map`. |
| `batch_triage_with_context` | Contextual Batch Triage | Hard | Triage 10 issues, detecting duplicates and balancing workload across teams. |
| `clarification_triage` | Multi-Turn Clarification | Expert | Ask up to 3 clarifying questions before submitting triage. Penalizes extra turns. |

---

##  Reward Shaping Philosophy

### Easy/Medium Tasks
- **Binary/Weighted Scoring**: Task 1 is binary. Task 2 rewards are split: Label (40%), Priority (30%), Assignee (15%), Component (15%).

### Hard Task (Trajectory Rewards)
- **Workload Balance Bonus (+0.15 max)**: Rewards distributing issues evenly across teams.
- **Consistency Penalty (-0.05 per violation)**: Penalizes inconsistent labeling for similar issues.
- **Duplicate Detection Bonus (+0.2)**: Rewards correct identification of duplicate issues.

### Expert Task (Clarification)
- **Turn Penalty (-0.08 per turn)**: Encourages efficiency. Agents must decide if the information gain from a question outweighs the score reduction.

---

##  Quick Start

### Installation
```bash
pip install -e ".[dev,redis,inference]"
```

### Running the Server
```bash
# Start the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Running Baseline Inference
The `inference.py` script is fully compliant with the evaluation criteria, emitting structured logs (`[START]`, `[STEP]`, `[END]`).

**Mandatory Environment Variables:**
- `HF_TOKEN`: Your API Key (used for LLM calls).
- `MODEL_NAME`: The model to use (default: `gpt-4o-mini`).
- `API_BASE_URL`: The LLM API endpoint.

```bash
export HF_TOKEN="sk-..."
python inference.py
```

---

##  Testing

```bash
# Run all tests
pytest

# Test specific components
pytest tests/test_environment.py
pytest tests/test_websocket.py
```

---

##  API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task_id=...` | POST | Initialize a new session. (Rate limited: 60/min) |
| `/step` | POST | Execute an action. (Rate limited: 300/min) |
| `/health` | GET | System health, version, and session store status. |
| `/metrics` | GET | Training metrics (avg reward, throughput). |
| `/ws` | WS | High-speed WebSocket interface. |

---

##  Technical Details

### Context-Awareness
Every observation includes a `project_map` derived from `data/project_structure.json`. This allows agents to understand the repository's architecture and assign issues to the correct teams and components based on the files they own.

### Performance & Scalability
- **WebSockets**: Reduce overhead by 10x for high-speed training loops.
- **Redis Support**: Set `REDIS_URL` to enable horizontal scaling across multiple workers/containers.
- **Statelessness**: Task state is serialized/deserialized automatically between steps when using Redis.

### Evaluation Compliance
`inference.py` follows the mandatory stdout format:
- `[START] task_id="..."`
- `[STEP] step=..., score=..., done=...`
- `[END] total_reward=...`
