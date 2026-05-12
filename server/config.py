# server/config.py

from pydantic_settings import BaseSettings
from typing import Dict

class TriageConfig(BaseSettings):
    # Reward weights for Full Triage and Batch Triage
    LABEL_WEIGHT: float = 0.4
    PRIORITY_WEIGHT: float = 0.3
    ASSIGNEE_WEIGHT: float = 0.15
    COMPONENT_WEIGHT: float = 0.15
    
    # Range for clamp_score
    REWARD_MIN: float = 0.01
    REWARD_MAX: float = 0.95
    
    # Penalties
    TURN_PENALTY: float = 0.1  # Per turn in ClarificationTask
    
    class Config:
        env_prefix = "TRIAGER_"

config = TriageConfig()
