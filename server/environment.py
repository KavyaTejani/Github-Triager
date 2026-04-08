import json
import random
import logging
from typing import List, Dict, Optional, Any, Union

from models import (
    IssueObservation,
    LabelClassificationObservation, LabelClassificationAction,
    FullTriageObservation, FullTriageAction,
    BatchTriageObservation, BatchTriageAction,
    StepResult,
    ClarificationRequest, ClarificationTriageAction, ClarificationObservation, ClarificationReward
)
from server.graders import (
    LabelClassificationGrader, FullTriageGrader, BatchTriageGrader
)

logger = logging.getLogger(__name__)

class IssueStore:
    def __init__(self, data_path: str = "data/simulated_issues.json",
                 project_structure_path: str = "data/project_structure.json"):
        self.issues: List[Dict] = []
        self.project_map: Dict = {}
        self.load_issues(data_path)
        self.load_project_map(project_structure_path)

    def load_issues(self, path: str):
        try:
            with open(path, 'r') as f:
                self.issues = json.load(f)
            logger.info(f"Loaded {len(self.issues)} issues from {path}")
        except Exception as e:
            logger.error(f"Failed to load issues from {path}: {e}")
            self.issues = []

    def load_project_map(self, path: str):
        try:
            with open(path, 'r') as f:
                self.project_map = json.load(f)
            logger.info(f"Loaded project map from {path}")
        except Exception as e:
            logger.warning(f"Could not load project map: {e}")
            self.project_map = {}

    def get_random_issue(self) -> Dict:
        if not self.issues:
            raise ValueError("Issue store is empty")
        return random.choice(self.issues)

    def get_random_batch(self, batch_size: int = 10) -> List[Dict]:
        if not self.issues:
            raise ValueError("Issue store is empty")
        return random.sample(self.issues, min(batch_size, len(self.issues)))

    def get_batch_by_group(self, group: str) -> List[Dict]:
        return [i for i in self.issues if i.get('batch_group') == group]

    def get_issues_by_label(self, label: str) -> List[Dict]:
        return [i for i in self.issues if i.get('gold_label') == label]

    def get_balanced_batch(self, batch_size: int = 10) -> List[Dict]:
        """Returns a batch with balanced label distribution."""
        from collections import defaultdict
        by_label = defaultdict(list)
        for issue in self.issues:
            by_label[issue.get('gold_label', 'bug')].append(issue)
        
        result = []
        labels = list(by_label.keys())
        if not labels:
            return self.get_random_batch(batch_size)
            
        i = 0
        while len(result) < batch_size and any(by_label.values()):
            label = labels[i % len(labels)]
            if by_label[label]:
                result.append(by_label[label].pop(0))
            i += 1
        return result


class LabelClassificationTask:
    """Task 1 (Easy): Classify a single issue."""

    def __init__(self, store: IssueStore):
        self.store = store
        self.grader = LabelClassificationGrader()
        self.current_issue: Optional[Dict] = None

    def reset(self) -> LabelClassificationObservation:
        self.current_issue = self.store.get_random_issue()
        return LabelClassificationObservation(
            issue=self._to_obs(self.current_issue)
        )

    def step(self, action: LabelClassificationAction) -> StepResult:
        if self.current_issue is None:
            raise ValueError("Task not reset")
        reward = self.grader.grade(action, self.current_issue)
        return StepResult(
            observation=None,
            reward=reward.model_dump(),
            done=True,
            info={"issue_id": self.current_issue['issue_id']}
        )

    def get_state(self) -> Dict:
        return {"current_issue": self.current_issue}

    def restore_state(self, state: Dict) -> None:
        self.current_issue = state.get("current_issue")

    def _to_obs(self, issue: Dict) -> IssueObservation:
        return IssueObservation(
            issue_id=issue['issue_id'],
            title=issue['title'],
            body=issue['body'],
            author=issue['author'],
            created_at=issue['created_at'],
            labels=[],
            project_map=self.store.project_map,
            repository_context={
                "primary_language": "Python",
                "open_issues": 1234,
                "stars": 4821,
                "forks": 312,
                "last_commit_at": "2024-01-20T08:00:00Z",
                "top_contributors": ["alice", "bob", "carol"]
            }
        )


class FullTriageTask:
    """Task 2 (Medium): Full triage of a single issue."""

    def __init__(self, store: IssueStore):
        self.store = store
        self.grader = FullTriageGrader()
        self.current_issue: Optional[Dict] = None

    def reset(self) -> FullTriageObservation:
        self.current_issue = self.store.get_random_issue()
        return FullTriageObservation(
            issue=self._to_obs(self.current_issue)
        )

    def step(self, action: FullTriageAction) -> StepResult:
        if self.current_issue is None:
            raise ValueError("Task not reset")
        reward = self.grader.grade(action, self.current_issue)
        return StepResult(
            observation=None,
            reward=reward.model_dump(),
            done=True,
            info={"issue_id": self.current_issue['issue_id']}
        )

    def get_state(self) -> Dict:
        return {"current_issue": self.current_issue}

    def restore_state(self, state: Dict) -> None:
        self.current_issue = state.get("current_issue")

    def _to_obs(self, issue: Dict) -> IssueObservation:
        return IssueObservation(
            issue_id=issue['issue_id'],
            title=issue['title'],
            body=issue['body'],
            author=issue['author'],
            created_at=issue['created_at'],
            labels=[],
            project_map=self.store.project_map,
            repository_context={
                "primary_language": "Python",
                "open_issues": 1234,
                "stars": 4821,
                "forks": 312,
                "last_commit_at": "2024-01-20T08:00:00Z",
                "top_contributors": ["alice", "bob", "carol"]
            }
        )


class BatchTriageTask:
    """Task 3 (Hard): Triage a batch of 10 issues with context."""

    def __init__(self, store: IssueStore, batch_size: int = 10):
        self.store = store
        self.grader = BatchTriageGrader()
        self.batch: List[Dict] = []
        self.batch_index: int = 0
        self.batch_size = batch_size
        self.prior_decisions: List[Dict] = []

    def reset(self) -> BatchTriageObservation:
        self.grader.reset()
        self.batch = self.store.get_balanced_batch(self.batch_size)
        self.batch_index = 0
        self.prior_decisions = []
        return self._current_observation()

    def step(self, action: BatchTriageAction) -> StepResult:
        if not self.batch:
            raise ValueError("Task not reset")
        
        current_issue = self.batch[self.batch_index]
        step_reward = self.grader.grade_step(action, current_issue)

        self.prior_decisions.append({
            "issue_id": current_issue['issue_id'],
            "label": action.label.value,
            "priority": action.priority.value,
            "assignee": action.suggested_assignee
        })

        self.batch_index += 1
        done = self.batch_index >= len(self.batch)

        if done:
            trajectory_reward = self.grader.grade_trajectory()
            return StepResult(
                observation=None,
                reward=trajectory_reward.model_dump(),
                done=True,
                info={
                    "issues_triaged": len(self.batch),
                    "trajectory_final": True
                }
            )

        return StepResult(
            observation=self._current_observation().model_dump(),
            reward=step_reward.model_dump(),
            done=False,
            info={"issue_id": current_issue['issue_id']}
        )

    def get_state(self) -> Dict:
        return {
            "batch": self.batch,
            "batch_index": self.batch_index,
            "prior_decisions": self.prior_decisions,
            "trajectory_actions": self.grader.trajectory_actions,
            "trajectory_golds": self.grader.trajectory_golds
        }

    def restore_state(self, state: Dict) -> None:
        self.batch = state.get("batch", [])
        self.batch_index = state.get("batch_index", 0)
        self.prior_decisions = state.get("prior_decisions", [])
        self.grader.trajectory_actions = state.get("trajectory_actions", [])
        self.grader.trajectory_golds = state.get("trajectory_golds", [])

    def _current_observation(self) -> BatchTriageObservation:
        issue = self.batch[self.batch_index]
        dup_candidates = self._find_duplicate_candidates(issue)
        return BatchTriageObservation(
            issue=self._to_obs(issue),
            batch_position=self.batch_index,
            batch_size=len(self.batch),
            prior_triage_decisions=self.prior_decisions.copy(),
            duplicate_candidates=dup_candidates
        )

    def _to_obs(self, issue: Dict) -> IssueObservation:
        return IssueObservation(
            issue_id=issue['issue_id'],
            title=issue['title'],
            body=issue['body'],
            author=issue['author'],
            created_at=issue['created_at'],
            labels=[],
            project_map=self.store.project_map,
            repository_context={
                "primary_language": "Python",
                "open_issues": 1234,
                "stars": 4821,
                "forks": 312,
                "last_commit_at": "2024-01-20T08:00:00Z",
                "top_contributors": ["alice", "bob", "carol"]
            }
        )

    def _find_duplicate_candidates(self, issue: Dict) -> List[str]:
        candidates = []
        for prev in self.batch[:self.batch_index]:
            if prev.get('batch_group') == issue.get('batch_group'):
                candidates.append(prev['issue_id'])
        return candidates


class ClarificationTask:
    """
    Task 4 (Expert): Multi-turn triage with clarification.
    """
    
    MAX_TURNS = 3
    TURN_PENALTY = 0.08

    def __init__(self, store: IssueStore):
        self.store = store
        self.grader = FullTriageGrader()
        self.current_issue: Optional[Dict] = None
        self.turn = 0
        self.clarification_history: List[Dict] = []

    def reset(self) -> ClarificationObservation:
        # Select a vague issue (one with clarification_qa data)
        vague_issues = [i for i in self.store.issues if i.get('clarification_qa')]
        if not vague_issues:
            vague_issues = self.store.issues
        self.current_issue = random.choice(vague_issues)
        self.turn = 0
        self.clarification_history = []
        return self._current_observation()

    def step(self, action: Union[ClarificationRequest, ClarificationTriageAction]) -> StepResult:
        if self.current_issue is None:
            raise ValueError("Task not reset")
        
        # Check if action is dict (from API) and parse
        if isinstance(action, dict):
            action_type = action.get("action_type", "submit_triage")
            if action_type == "ask_clarification":
                action = ClarificationRequest(**action)
            else:
                action = ClarificationTriageAction(**action)
        
        if action.action_type == "ask_clarification":
            return self._handle_clarification(action)
        elif action.action_type == "submit_triage":
            return self._handle_triage(action)
        else:
            raise ValueError(f"Unknown action_type: {action.action_type}")

    def _handle_clarification(self, action: ClarificationRequest) -> StepResult:
        if self.turn >= self.MAX_TURNS:
            return StepResult(
                observation=None,
                reward={"score": 0.0, "reason": "max_turns_exceeded"},
                done=True,
                info={"turn": self.turn, "max_turns": self.MAX_TURNS}
            )
        
        answer = self._find_answer(action.question)
        self.clarification_history.append({
            "question": action.question,
            "answer": answer
        })
        self.turn += 1
        
        return StepResult(
            observation=self._current_observation().model_dump(),
            reward={"score": 0.0, "turn": self.turn},
            done=False,
            info={"turn": self.turn, "answer": answer}
        )

    def _handle_triage(self, action: ClarificationTriageAction) -> StepResult:
        base_action = FullTriageAction(
            label=action.label,
            priority=action.priority,
            suggested_assignee=action.suggested_assignee,
            suggested_component=action.suggested_component
        )
        base_reward = self.grader.grade(base_action, self.current_issue)
        
        turn_penalty = self.turn * self.TURN_PENALTY
        final_score = max(0.0, base_reward.score - turn_penalty)
        
        reward = ClarificationReward(
            score=round(final_score, 4),
            base_triage_score=base_reward.score,
            turn_penalty=round(turn_penalty, 4),
            turns_taken=self.turn,
            label_correct=base_reward.label_correct,
            priority_correct=base_reward.priority_correct,
            breakdown={
                "base_score": base_reward.score,
                "turn_penalty": -turn_penalty,
                "final_score": final_score
            }
        )
        
        return StepResult(
            observation=None,
            reward=reward.model_dump(),
            done=True,
            info={"turns_taken": self.turn}
        )

    def _find_answer(self, question: str) -> str:
        question_lower = question.lower()
        qa_pairs = self.current_issue.get('clarification_qa', [])
        for qa in qa_pairs:
            if any(kw in question_lower for kw in qa.get('keywords', [])):
                return qa['answer']
        return "I don't have additional information about that aspect of the issue."

    def _current_observation(self) -> ClarificationObservation:
        return ClarificationObservation(
            issue=self._to_obs(self.current_issue),
            turn=self.turn,
            max_turns=self.MAX_TURNS,
            clarification_history=self.clarification_history.copy()
        )
    
    def _to_obs(self, issue: Dict) -> IssueObservation:
        return IssueObservation(
            issue_id=issue['issue_id'],
            title=issue['title'],
            body=issue['body'],
            author=issue['author'],
            created_at=issue['created_at'],
            labels=[],
            project_map=self.store.project_map,
            repository_context={
                "primary_language": "Python",
                "open_issues": 1234,
                "stars": 4821,
                "forks": 312,
                "last_commit_at": "2024-01-20T08:00:00Z",
                "top_contributors": ["alice", "bob", "carol"]
            }
        )

    def get_state(self) -> Dict:
        return {
            "current_issue": self.current_issue,
            "turn": self.turn,
            "clarification_history": self.clarification_history
        }
    
    def restore_state(self, state: Dict) -> None:
        self.current_issue = state.get("current_issue")
        self.turn = state.get("turn", 0)
        self.clarification_history = state.get("clarification_history", [])
