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
from server.session_store import create_session_store
from server.logging_config import configure_logging
from server.graders import clamp_score
import structlog

configure_logging()
logger = structlog.get_logger()

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="GitHub Triager RL", version="1.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

issue_store = IssueStore("data/simulated_issues.json")
session_store = create_session_store()

TASK_REGISTRY = {
    "label_classification": LabelClassificationTask,
    "full_triage": FullTriageTask,
    "batch_triage_with_context": BatchTriageTask,
    "clarification_triage": ClarificationTask,
}

@app.get("/health")
async def health(): return {"status": "healthy"}

@app.post("/reset")
@limiter.limit("1000/minute")
async def reset(request: Request, task_id: str = "label_classification"):
    if task_id not in TASK_REGISTRY: raise HTTPException(400, "Unknown task")
    sid = str(uuid.uuid4())
    task = TASK_REGISTRY[task_id](store=issue_store)
    obs = task.reset()
    session_store.set(sid, {"task_id": task_id, "task": task})
    return {"session_id": sid, "task_id": task_id, "observation": obs.model_dump()}

@app.post("/step")
@limiter.limit("5000/minute")
async def step(request: Request, session_id: str, action: Dict[str, Any]):
    s = session_store.get(session_id)
    if not s: raise HTTPException(404, "Not found")
    task = s["task"]
    try:
        res = task.step(parse_action(s["task_id"], action))
        if res.done: session_store.delete(session_id)
        return res.model_dump()
    except Exception as e: raise HTTPException(422, str(e))

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await websocket.accept()
    sessions = {}
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "reset":
                tid = data.get("task_id", "label_classification")
                task = TASK_REGISTRY[tid](store=issue_store)
                sid = str(uuid.uuid4())
                sessions[sid] = {"task": task, "tid": tid}
                await websocket.send_json(make_observation_message(sid, task.reset()))
            elif data["type"] == "step":
                sid = data["session_id"]
                if sid in sessions:
                    res = sessions[sid]["task"].step(parse_action(sessions[sid]["tid"], data["action"]))
                    if res.done: del sessions[sid]
                    await websocket.send_json(make_step_result_message(res))
    except: pass

def main():
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

if __name__ == "__main__": main()
