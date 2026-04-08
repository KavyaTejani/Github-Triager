# server/session_store.py

import json
import os
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 3600  # 1 hour

class BaseSessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def set(self, session_id: str, data: Dict) -> None:
        pass

    @abstractmethod
    def delete(self, session_id: str) -> None:
        pass

    @abstractmethod
    def count(self) -> int:
        pass


class InMemorySessionStore(BaseSessionStore):
    """Default store for development and testing."""
    
    def __init__(self):
        self._store: Dict[str, Dict] = {}

    def get(self, session_id: str) -> Optional[Dict]:
        return self._store.get(session_id)

    def set(self, session_id: str, data: Dict) -> None:
        self._store[session_id] = data

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def count(self) -> int:
        return len(self._store)


class RedisSessionStore(BaseSessionStore):
    """Production store using Redis for multi-worker deployments."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Connected to Redis at {redis_url}")

    def get(self, session_id: str) -> Optional[Dict]:
        raw = self._client.get(f"session:{session_id}")
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, session_id: str, data: Dict) -> None:
        # Serialize only JSON-safe fields (task state handled by caller)
        serializable = {k: v for k, v in data.items() if k != "task"}
        self._client.setex(
            f"session:{session_id}",
            SESSION_TTL_SECONDS,
            json.dumps(serializable)
        )

    def delete(self, session_id: str) -> None:
        self._client.delete(f"session:{session_id}")

    def count(self) -> int:
        # Note: keys() is O(N), but fine for typical OpenEnv batch sizes
        return len(self._client.keys("session:*"))


def create_session_store() -> BaseSessionStore:
    """
    Factory function. Returns RedisSessionStore if REDIS_URL is set,
    otherwise returns InMemorySessionStore.
    """
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            store = RedisSessionStore(redis_url)
            store._client.ping()  # Test connection
            logger.info("Using Redis session store")
            return store
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), falling back to in-memory store")
    
    logger.info("Using in-memory session store")
    return InMemorySessionStore()
