# server/ws_handler.py

import logging
from typing import Dict, Any, Optional
from models import (
    LabelClassificationAction, FullTriageAction, BatchTriageAction,
    ClarificationRequest, ClarificationTriageAction
)

logger = logging.getLogger(__name__)

TASK_ACTION_MAP = {
    "label_classification": LabelClassificationAction,
    "full_triage": FullTriageAction,
    "batch_triage_with_context": BatchTriageAction,
}

def parse_action(task_id: str, action_data: Dict) -> Any:
    """Parse raw action dict into the correct Pydantic action model."""
    if task_id == "clarification_triage":
        action_type = action_data.get("action_type", "submit_triage")
        if action_type == "ask_clarification":
            return ClarificationRequest(**action_data)
        return ClarificationTriageAction(**action_data)

    action_cls = TASK_ACTION_MAP.get(task_id)
    if not action_cls:
        raise ValueError(f"Unknown task_id: {task_id}")
    return action_cls(**action_data)

def make_error_message(error: str) -> Dict:
    return {"type": "error", "error": error}

def make_observation_message(session_id: str, observation: Any) -> Dict:
    return {
        "type": "observation",
        "session_id": session_id,
        "observation": observation.model_dump() if hasattr(observation, 'model_dump') else observation
    }

def make_step_result_message(result: Any) -> Dict:
    return {
        "type": "step_result",
        **result.model_dump()
    }
