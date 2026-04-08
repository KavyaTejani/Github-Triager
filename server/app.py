from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
from typing import Dict, Any, Optional, List

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from server.environment import (
    IssueStore, LabelClassificationTask, FullTriageTask, BatchTriageTask, ClarificationTask
)
from models import (
    LabelClassificationAction, FullTriageAction, BatchTriageAction, StepResult
)
from server.ws_handler import parse_action, make_error_message, make_observation_message, make_step_result_message
from server.session_store import create_session_store, RedisSessionStore
from server.logging_config import configure_logging
import structlog

configure_logging()
logger = structlog.get_logger()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="GitHub Triager Environment",
    description="OpenEnv-compliant issue triage simulation with 4 tasks",
    version="1.1.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize IssueStore
issue_store = IssueStore("data/simulated_issues.json")

# Session storage
session_store = create_session_store()

# Global metrics
total_resets = 0
total_steps = 0
recent_rewards: List[float] = []

TASK_REGISTRY = {
    "label_classification": LabelClassificationTask,
    "full_triage": FullTriageTask,
    "batch_triage_with_context": BatchTriageTask,
    "clarification_triage": ClarificationTask,
}

def _get_task_instance(session_data: Dict) -> Any:
    """Helper to recreate a task instance and restore its state."""
    task_id = session_data["task_id"]
    task_cls = TASK_REGISTRY[task_id]
    task_instance = task_cls(store=issue_store)
    if "task_state" in session_data:
        task_instance.restore_state(session_data["task_state"])
    return task_instance

@app.get("/health")
async def health_check():
    store_type = "redis" if isinstance(session_store, RedisSessionStore) else "in_memory"
    return {
        "status": "healthy",
        "version": "1.1.0",
        "tasks": list(TASK_REGISTRY.keys()),
        "session_store": store_type,
        "active_sessions": session_store.count()
    }

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics for monitoring training runs."""
    avg_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
    return {
        "total_resets": total_resets,
        "total_steps": total_steps,
        "avg_reward_recent": round(avg_reward, 4),
        "active_sessions": session_store.count()
    }

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "label_classification", "difficulty": "easy",
             "description": "Classify a single issue into one of 5 labels"},
            {"id": "full_triage", "difficulty": "medium",
             "description": "Full triage: label, priority, assignee, and component"},
            {"id": "batch_triage_with_context", "difficulty": "hard",
             "description": "Triage a batch of 10 issues with inter-issue context"},
            {"id": "clarification_triage", "difficulty": "expert",
             "description": "Multi-turn triage: ask up to 3 clarifying questions before submitting"},
        ]
    }


@app.post("/reset")
@limiter.limit("60/minute")
async def reset_endpoint(request: Request, task_id: str = "label_classification"):
    global total_resets
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")

    session_id = str(uuid.uuid4())
    task_cls = TASK_REGISTRY[task_id]
    task_instance = task_cls(store=issue_store)
    observation = task_instance.reset()

    session_store.set(session_id, {
        "task_id": task_id,
        "task_state": task_instance.get_state(),
        "task": task_instance  # Keep in memory for InMemoryStore
    })

    total_resets += 1
    return {
        "session_id": session_id,
        "task_id": task_id,
        "observation": observation.model_dump()
    }


@app.post("/step")
@limiter.limit("5000/minute")
async def step_endpoint(request: Request, session_id: str, action: Dict[str, Any]):
    global total_steps, recent_rewards
    session_data = session_store.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    task_id = session_data["task_id"]
    task = session_data.get("task") or _get_task_instance(session_data)

    try:
        parsed_action = parse_action(task_id, action)
        result: StepResult = task.step(parsed_action)

        total_steps += 1
        reward_score = float(result.reward.get("score", result.reward.get("step_score", 0.0)))
        recent_rewards.append(reward_score)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        if result.done:
            session_store.delete(session_id)
        else:
            session_data["task_state"] = task.get_state()
            session_store.set(session_id, session_data)

        return result.model_dump()

    except Exception as e:
        logger.error("step_error", error=str(e), session_id=session_id)
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
async def state_endpoint(session_id: str):
    session_data = session_store.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "task_id": session_data["task_id"],
        "total_active_sessions": session_store.count()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_sessions: Dict[str, Dict] = {} 
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "reset":
                task_id = data.get("task_id", "label_classification")
                if task_id not in TASK_REGISTRY:
                    await websocket.send_json(make_error_message(f"Unknown task: {task_id}"))
                    continue
                
                session_id = str(uuid.uuid4())
                task_cls = TASK_REGISTRY[task_id]
                task_instance = task_cls(store=issue_store)
                observation = task_instance.reset()
                
                ws_sessions[session_id] = {"task_id": task_id, "task": task_instance}
                await websocket.send_json(make_observation_message(session_id, observation))

            elif msg_type == "step":
                session_id = data.get("session_id")
                action_data = data.get("action", {})
                
                if not session_id or session_id not in ws_sessions:
                    await websocket.send_json(make_error_message("Invalid session_id"))
                    continue
                
                session = ws_sessions[session_id]
                try:
                    parsed_action = parse_action(session["task_id"], action_data)
                    result = session["task"].step(parsed_action)
                    if result.done:
                        del ws_sessions[session_id]
                    await websocket.send_json(make_step_result_message(result))
                except Exception as e:
                    await websocket.send_json(make_error_message(str(e)))

            elif msg_type == "state":
                session_id = data.get("session_id")
                if session_id and session_id in ws_sessions:
                    await websocket.send_json({
                        "type": "state",
                        "session_id": session_id,
                        "task_id": ws_sessions[session_id]["task_id"],
                        "active_ws_sessions": len(ws_sessions)
                    })
                else:
                    await websocket.send_json(make_error_message("Session not found"))
            
    except WebSocketDisconnect:
        ws_sessions.clear()
    except Exception as e:
        logger.error("ws_error", error=str(e))


def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()

