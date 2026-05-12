import os
import pytest
from server.environment import IssueStore, LabelClassificationTask, FullTriageTask, BatchTriageTask, ClarificationTask
from models import LabelEnum, PriorityEnum, LabelClassificationAction, FullTriageAction, BatchTriageAction, ClarificationRequest, ClarificationTriageAction

@pytest.fixture
def store():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "simulated_issues.json")
    project_path = os.path.join(base_dir, "data", "project_structure.json")
    return IssueStore(data_path, project_path)

def test_label_classification_task(store):
    task = LabelClassificationTask(store)
    obs = task.reset()
    assert obs.issue.issue_id is not None
    assert obs.issue.project_map is not None
    
    action = LabelClassificationAction(label=LabelEnum.BUG)
    result = task.step(action)
    assert result.done is True
    assert "reward" in result.model_dump()

def test_full_triage_task(store):
    task = FullTriageTask(store)
    obs = task.reset()
    assert obs.issue.issue_id is not None
    
    action = FullTriageAction(
        label=LabelEnum.BUG,
        priority=PriorityEnum.HIGH,
        suggested_assignee="backend_team",
        suggested_component="api"
    )
    result = task.step(action)
    assert result.done is True
    assert result.reward["score"] >= 0.01

def test_batch_triage_task(store):
    task = BatchTriageTask(store, size=3)
    obs = task.reset()
    assert task.idx == 0
    
    for i in range(3):
        action = BatchTriageAction(
            label=LabelEnum.BUG,
            priority=PriorityEnum.MEDIUM
        )
        result = task.step(action)
        if i < 2:
            assert result.done is False
            assert result.observation is not None
        else:
            assert result.done is True
            assert result.reward["is_trajectory_final"] is True

def test_clarification_task(store):
    task = ClarificationTask(store)
    obs = task.reset()
    assert task.turn == 0
    
    # Ask a question
    action = ClarificationRequest(question="What is the error?")
    result = task.step(action)
    assert result.done is False
    assert task.turn == 1
    assert len(task.clarification_history) == 2 # Agent question + User response
    
    # Submit triage
    action = ClarificationTriageAction(
        label=LabelEnum.BUG,
        priority=PriorityEnum.CRITICAL,
        suggested_assignee="backend_team",
        suggested_component="api",
        confidence=0.9
    )
    result = task.step(action)
    assert result.done is True
    assert result.reward["turns_taken"] == 1
    # Penalty should be applied in final score calculation
    assert result.reward["score"] < 0.95
