import httpx
from typing import Optional, Dict, Any, Union

class GitHubTriagerClient:
    """
    HTTP client for the GitHub Triager OpenEnv environment.
    Follows the standard OpenEnv interface (reset, step, state).
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session_id: Optional[str] = None
        self.task_id: Optional[str] = None
        # Use a longer timeout for LLM-based agents
        self.http = httpx.Client(base_url=self.base_url, timeout=30.0)

    def reset(self, task_id: str = "label_classification") -> Dict[str, Any]:
        """Reset the environment and start a new episode."""
        resp = self.http.post("/reset", params={"task_id": task_id})
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        self.task_id = data["task_id"]
        return data["observation"]

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return the result."""
        if not self.session_id:
            raise ValueError("No active session. Call reset() first.")
        
        resp = self.http.post(
            "/step",
            params={"session_id": self.session_id},
            json=action
        )
        resp.raise_for_status()
        data = resp.json()
        
        # If the episode is done, clear the session ID
        if data.get("done"):
            self.session_id = None
            
        return data

    def state(self) -> Dict[str, Any]:
        """Get the current metadata state of the session."""
        if not self.session_id:
            raise ValueError("No active session.")
            
        resp = self.http.get("/state", params={"session_id": self.session_id})
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check the health status of the server."""
        resp = self.http.get("/health")
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> Dict[str, Any]:
        """List available tasks and their metadata."""
        resp = self.http.get("/tasks")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP client."""
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class GitHubTriagerWSClient:
    """
    Async WebSocket client for high-speed training loops.
    Usage:
        async with GitHubTriagerWSClient() as client:
            obs = await client.reset("full_triage")
            result = await client.step({"label": "bug", "priority": "high"})
    """
    
    def __init__(self, base_url: str = "ws://localhost:8000/ws"):
        import json
        self.json = json
        self.base_url = base_url
        self.session_id: Optional[str] = None
        self.task_id: Optional[str] = None
        self._ws = None

    async def connect(self):
        import websockets
        self._ws = await websockets.connect(self.base_url)

    async def disconnect(self):
        if self._ws:
            await self._ws.close()

    async def reset(self, task_id: str = "label_classification") -> Dict[str, Any]:
        await self._ws.send(self.json.dumps({"type": "reset", "task_id": task_id}))
        response = self.json.loads(await self._ws.recv())
        if response.get("type") == "error":
            raise RuntimeError(response["error"])
        self.session_id = response["session_id"]
        self.task_id = task_id
        return response["observation"]

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session_id:
            raise ValueError("No active session. Call reset() first.")
        await self._ws.send(self.json.dumps({
            "type": "step",
            "session_id": self.session_id,
            "task_id": self.task_id,
            "action": action
        }))
        response = self.json.loads(await self._ws.recv())
        if response.get("type") == "error":
            raise RuntimeError(response["error"])
        if response.get("done"):
            self.session_id = None
        return response

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()
