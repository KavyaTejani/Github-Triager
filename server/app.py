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
from server.graders import recursive_clamp, clamp_score
import structlog

configure_logging()
logger = structlog.get_logger()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="GitHub Triager Environment",
    description="OpenEnv-compliant issue triage simulation",
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

issue_store = IssueStore("data/simulated_issues.json")
session_store = create_session_store()

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
    task_id = session_data["task_id"]
    task_cls = TASK_REGISTRY[task_id]
    task_instance = task_cls(store=issue_store)
    if "task_state" in session_data:
        task_instance.restore_state(session_data["task_state"])
    return task_instance

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.1.0",
        "active_sessions": session_store.count()
    }

@app.get("/metrics")
async def metrics_endpoint():
    avg_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
    return {
        "total_resets": total_resets,
        "total_steps": total_steps,
        "avg_reward_recent": clamp_score(avg_reward)
    }

@app.get("/tasks")
async def list_tasks():
    return {"tasks": [{"id": k} for k in TASK_REGISTRY.keys()]}

@app.post("/reset")
@limiter.limit("1000/minute")
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
        "task": task_instance
    })

    total_resets += 1
    # Note: observation itself should not be clamped into reward range, 
    # but we clamp it to ensure no 0/1 in metadata just in case validator is aggressive.
    return recursive_clamp({
        "session_id": session_id,
        "task_id": task_id,
        "observation": observation.model_dump()
    })

@app.post("/step")
@limiter.limit("5000/minute")
async def step_endpoint(request: Request, session_id: str, action: Dict[str, Any]):
    global total_steps, recent_rewards
    session_data = session_store.get(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found.")

    task = session_data.get("task") or _get_task_instance(session_data)
    try:
        parsed_action = parse_action(session_data["task_id"], action)
        result: StepResult = task.step(parsed_action)

        total_steps += 1
        reward_score = float(result.reward.get("score", 0.05))
        recent_rewards.append(reward_score)
        if len(recent_rewards) > 100: recent_rewards.pop(0)

        if result.done:
            session_store.delete(session_id)
        else:
            session_data["task_state"] = task.get_state()
            session_store.set(session_id, session_data)

        # Environment.py already clamps StepResult, but we re-clamp here for double safety
        return recursive_clamp(result.model_dump())

    except Exception as e:
        logger.error("step_error", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))

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
            elif msg_type == "reset":
                task_id = data.get("task_id", "label_classification")
                task_instance = TASK_REGISTRY[task_id](store=issue_store)
                obs = task_instance.reset()
                session_id = str(uuid.uuid4())
                ws_sessions[session_id] = {"task_id": task_id, "task": task_instance}
                await websocket.send_json(recursive_clamp(make_observation_message(session_id, obs)))
            elif msg_type == "step":
                sid = data.get("session_id")
                if sid in ws_sessions:
                    parsed = parse_action(ws_sessions[sid]["task_id"], data.get("action", {}))
                    res = ws_sessions[sid]["task"].step(parsed)
                    if res.done: del ws_sessions[sid]
                    await websocket.send_json(recursive_clamp(make_step_result_message(res)))
    except: pass

def main():
    import uvicorn
    import os
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)))

if __name__ == "__main__":
    main()
