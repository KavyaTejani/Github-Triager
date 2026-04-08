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
    issue_id: str
    title: str = Field(..., max_length=200)
    body: str = Field(..., max_length=5000)
    author: str
    created_at: str
    labels: List[str] = Field(default_factory=list)
    project_map: Optional[Dict[str, Any]] = Field(default_factory=dict)
    repository_context: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('title')
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v

class LabelClassificationReward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)
    correct: bool
    expected_label: str
    predicted_label: str

class FullTriageReward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)
    label_correct: bool
    priority_correct: bool
    assignee_correct: bool
    component_correct: bool
    breakdown: Dict[str, float]

class BatchTriageReward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)
    step_score: float = Field(..., gt=0.0, lt=1.0)
    trajectory_score: float = Field(default=0.01, gt=0.0, lt=1.0)
    duplicate_detection_bonus: float = 0.0
    workload_balance_bonus: float = 0.0
    consistency_penalty: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    is_trajectory_final: bool = False

class StepResult(BaseModel):
    observation: Optional[Dict[str, Any]] = None
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class ClarificationReward(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0)
    base_triage_score: float
    turn_penalty: float
    turns_taken: int
    label_correct: bool
    priority_correct: bool
    breakdown: Dict[str, float]

# These remain unchanged as they are not rewards
class LabelClassificationObservation(BaseModel):
    task_id: str = "label_classification"
    issue: IssueObservation

class LabelClassificationAction(BaseModel):
    label: LabelEnum

class FullTriageObservation(BaseModel):
    task_id: str = "full_triage"
    issue: IssueObservation

class FullTriageAction(BaseModel):
    label: LabelEnum
    priority: PriorityEnum
    suggested_assignee: Optional[str] = None
    suggested_component: Optional[str] = None

class BatchTriageObservation(BaseModel):
    task_id: str = "batch_triage_with_context"
    issue: IssueObservation
    batch_position: int
    batch_size: int
    prior_triage_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    duplicate_candidates: List[str] = Field(default_factory=list)

class BatchTriageAction(BaseModel):
    label: LabelEnum
    priority: PriorityEnum
    suggested_assignee: Optional[str] = None
    suggested_component: Optional[str] = None
    is_duplicate_of: Optional[str] = None
    priority_justification: Optional[str] = None

class ClarificationRequest(BaseModel):
    model_config = {"extra": "allow"}
    action_type: str = "ask_clarification"
    question: str = Field(..., max_length=1000)

class ClarificationTriageAction(BaseModel):
    model_config = {"extra": "allow"}
    action_type: str = "submit_triage"
    label: LabelEnum
    priority: PriorityEnum
    suggested_assignee: Optional[str] = None
    suggested_component: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

class ClarificationObservation(BaseModel):
    task_id: str = "clarification_triage"
    issue: IssueObservation
    turn: int = 0
    max_turns: int = 3
    clarification_history: List[Dict[str, str]] = Field(default_factory=list)
