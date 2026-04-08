from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum

class LabelEnum(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    DOCUMENTATION = "documentation"
    QUESTION = "question"
    ENHANCEMENT = "enhancement"

class PriorityEnum(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class IssueObservation(BaseModel):
    """A single GitHub issue presented to the agent."""
    issue_id: str
    title: str = Field(..., max_length=200)
    body: str = Field(..., max_length=5000)
    author: str
    created_at: str
    labels: List[str] = Field(default_factory=list)
    project_map: Optional[Dict[str, Any]] = Field(default_factory=dict)
    repository_context: Dict[str, Any] = Field(default_factory=dict)
    # Should include: primary_language, open_issues, stars, forks, last_commit_at, top_contributors

    @field_validator('title')
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v

# Task 1 — Label Classification (Easy)
class LabelClassificationObservation(BaseModel):
    """Observation for Task 1: single issue to label."""
    task_id: str = "label_classification"
    issue: IssueObservation

class LabelClassificationAction(BaseModel):
    """Action for Task 1: just pick a label."""
    label: LabelEnum

class LabelClassificationReward(BaseModel):
    """Reward for Task 1: binary correctness."""
    score: float = Field(..., gt=0.0, lt=1.0)
    correct: bool
    expected_label: str
    predicted_label: str

# Task 2 — Full Triage (Medium)
class FullTriageObservation(BaseModel):
    """Observation for Task 2: single issue, full triage expected."""
    task_id: str = "full_triage"
    issue: IssueObservation

class FullTriageAction(BaseModel):
    """Action for Task 2: label + priority + assignee + component."""
    label: LabelEnum
    priority: PriorityEnum
    suggested_assignee: Optional[str] = None
    suggested_component: Optional[str] = None

class FullTriageReward(BaseModel):
    """Reward for Task 2: weighted component scores."""
    score: float = Field(..., gt=0.0, lt=1.0)
    label_correct: bool
    priority_correct: bool
    assignee_correct: bool
    component_correct: bool
    breakdown: Dict[str, float]

# Task 3 — Contextual Batch Triage (Hard)
class BatchTriageObservation(BaseModel):
    """Observation for Task 3: current issue + batch context."""
    task_id: str = "batch_triage_with_context"
    issue: IssueObservation
    batch_position: int
    batch_size: int
    prior_triage_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    duplicate_candidates: List[str] = Field(default_factory=list)

class BatchTriageAction(BaseModel):
    """Action for Task 3: full triage + batch-aware decisions."""
    label: LabelEnum
    priority: PriorityEnum
    suggested_assignee: Optional[str] = None
    suggested_component: Optional[str] = None
    is_duplicate_of: Optional[str] = None
    priority_justification: Optional[str] = None

class BatchTriageReward(BaseModel):
    """
    Reward for Task 3: trajectory-level score computed at episode end.
    Includes per-step partial scores AND trajectory bonuses/penalties.
    """
    score: float = Field(..., gt=0.0, lt=1.0)
    step_score: float = Field(..., gt=0.0, lt=1.0)
    trajectory_score: Optional[float] = Field(None, gt=0.0, lt=1.0)
    duplicate_detection_bonus: float = 0.0
    workload_balance_bonus: float = 0.0
    consistency_penalty: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    is_trajectory_final: bool = False

# Unified Step Result
class StepResult(BaseModel):
    """Unified response from env.step() across all tasks."""
    observation: Optional[Dict[str, Any]] = None
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

# Task 4 — Clarification Triage (Expert)

class ClarificationRequest(BaseModel):
    """Agent asks for more information before triaging."""
    model_config = {"extra": "allow"}
    action_type: str = "ask_clarification"
    question: str = Field(..., max_length=1000)

class ClarificationTriageAction(BaseModel):
    """Agent submits final triage after gathering enough info."""
    model_config = {"extra": "allow"}
    action_type: str = "submit_triage"
    label: LabelEnum
    priority: PriorityEnum
    suggested_assignee: Optional[str] = None
    suggested_component: Optional[str] = None
    confidence: float = Field(..., gt=0.0, lt=1.0,
        description="Agent's self-reported confidence in its triage (0.01 to 0.99)")

class ClarificationObservation(BaseModel):
    """Observation for Task 4: includes simulated user responses."""
    task_id: str = "clarification_triage"
    issue: IssueObservation
    turn: int = 0
    max_turns: int = 3
    clarification_history: List[Dict[str, str]] = Field(default_factory=list)
    # Each entry: {"question": "...", "answer": "..."}

class ClarificationReward(BaseModel):
    """Reward for Task 4: correct triage penalized by number of turns taken."""
    score: float = Field(..., gt=0.0, lt=1.0)
    base_triage_score: float
    turn_penalty: float
    turns_taken: int
    label_correct: bool
    priority_correct: bool
    breakdown: Dict[str, float]
