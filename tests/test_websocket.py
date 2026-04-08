import pytest
from fastapi.testclient import TestClient
from server.app import app

def test_websocket_ping():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        data = ws.receive_json()
        assert data["type"] == "pong"

def test_websocket_reset_and_step():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        # Reset
        ws.send_json({"type": "reset", "task_id": "label_classification"})
        obs_msg = ws.receive_json()
        assert obs_msg["type"] == "observation"
        assert "session_id" in obs_msg
        session_id = obs_msg["session_id"]
        
        # Step
        ws.send_json({
            "type": "step",
            "session_id": session_id,
            "task_id": "label_classification",
            "action": {"label": "bug"}
        })
        result_msg = ws.receive_json()
        assert result_msg["type"] == "step_result"
        assert "reward" in result_msg
        assert result_msg["done"] is True

def test_websocket_unknown_task():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "task_id": "nonexistent_task"})
        error_msg = ws.receive_json()
        assert error_msg["type"] == "error"
