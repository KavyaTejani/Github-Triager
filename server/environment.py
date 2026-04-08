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
    LabelClassificationGrader, FullTriageGrader, BatchTriageGrader, clamp_score, recursive_clamp
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
        except Exception as e:
            logger.error(f"Failed to load issues: {e}")
            self.issues = []

    def load_project_map(self, path: str):
        try:
            with open(path, 'r') as f:
                self.project_map = json.load(f)
        except Exception as e:
            self.project_map = {}

    def get_random_issue(self) -> Dict:
        return random.choice(self.issues)

    def get_balanced_batch(self, batch_size: int = 10) -> List[Dict]:
        from collections import defaultdict
        by_label = defaultdict(list)
        for issue in self.issues:
            by_label[issue.get('gold_label', 'bug')].append(issue)
        result = []
        labels = list(by_label.keys())
        i = 0
        while len(result) < batch_size and any(by_label.values()):
            label = labels[i % len(labels)]
            if by_label[label]: result.append(by_label[label].pop(0))
            i += 1
        return result

def to_issue_obs(issue: Dict, project_map: Dict) -> IssueObservation:
    return IssueObservation(
        issue_id=issue['issue_id'],
        title=issue['title'],
        body=issue['body'],
        author=issue['author'],
        created_at=issue['created_at'],
        labels=[],
        project_map=project_map,
        repository_context={"primary_language": "Python", "stars": 4821}
    )

class LabelClassificationTask:
    def __init__(self, store: IssueStore):
        self.store = store
        self.grader = LabelClassificationGrader()
        self.current_issue = None

    def reset(self):
        self.current_issue = self.store.get_random_issue()
        return LabelClassificationObservation(issue=to_issue_obs(self.current_issue, self.store.project_map))

    def step(self, action: LabelClassificationAction) -> StepResult:
        reward = self.grader.grade(action, self.current_issue)
        res = StepResult(reward=reward.model_dump(), done=True, info={"issue_id": self.current_issue['issue_id']})
        return StepResult(**recursive_clamp(res.model_dump()))

    def get_state(self): return {"current_issue": self.current_issue}
    def restore_state(self, state): self.current_issue = state.get("current_issue")

class FullTriageTask:
    def __init__(self, store: IssueStore):
        self.store = store
        self.grader = FullTriageGrader()
        self.current_issue = None

    def reset(self):
        self.current_issue = self.store.get_random_issue()
        return FullTriageObservation(issue=to_issue_obs(self.current_issue, self.store.project_map))

    def step(self, action: FullTriageAction) -> StepResult:
        reward = self.grader.grade(action, self.current_issue)
        res = StepResult(reward=reward.model_dump(), done=True, info={"issue_id": self.current_issue['issue_id']})
        return StepResult(**recursive_clamp(res.model_dump()))

    def get_state(self): return {"current_issue": self.current_issue}
    def restore_state(self, state): self.current_issue = state.get("current_issue")

class BatchTriageTask:
    def __init__(self, store: IssueStore, batch_size: int = 10):
        self.store = store
        self.grader = BatchTriageGrader()
        self.batch = []
        self.batch_index = 0
        self.prior_decisions = []

    def reset(self):
        self.grader.reset()
        self.batch = self.store.get_balanced_batch(10)
        self.batch_index = 0
        self.prior_decisions = []
        return self._current_observation()

    def step(self, action: BatchTriageAction) -> StepResult:
        current_issue = self.batch[self.batch_index]
        step_reward = self.grader.grade_step(action, current_issue)
        self.prior_decisions.append({
            "issue_id": current_issue['issue_id'],
            "label": action.label.value,
            "priority": action.priority.value
        })
        self.batch_index += 1
        done = self.batch_index >= len(self.batch)
        if done:
            reward = self.grader.grade_trajectory()
            res = StepResult(reward=reward.model_dump(), done=True, info={"traj": True})
        else:
            res = StepResult(observation=self._current_observation().model_dump(), reward=step_reward.model_dump(), done=False)
        return StepResult(**recursive_clamp(res.model_dump()))

    def _current_observation(self):
        issue = self.batch[self.batch_index]
        return BatchTriageObservation(
            issue=to_issue_obs(issue, self.store.project_map),
            batch_position=self.batch_index,
            batch_size=len(self.batch),
            prior_triage_decisions=self.prior_decisions.copy()
        )

    def get_state(self): return {"batch": self.batch, "idx": self.batch_index, "prior": self.prior_decisions}
    def restore_state(self, state):
        self.batch = state.get("batch", [])
        self.batch_index = state.get("idx", 0)
        self.prior_decisions = state.get("prior", [])

class ClarificationTask:
    MAX_TURNS = 3
    TURN_PENALTY = 0.08

    def __init__(self, store: IssueStore):
        self.store = store
        self.grader = FullTriageGrader()
        self.current_issue = None
        self.turn = 0
        self.history = []

    def reset(self):
        vague = [i for i in self.store.issues if i.get('clarification_qa')]
        self.current_issue = random.choice(vague) if vague else random.choice(self.store.issues)
        self.turn = 0
        self.history = []
        return self._current_observation()

    def step(self, action: Any) -> StepResult:
        if isinstance(action, dict):
            action = ClarificationRequest(**action) if action.get("action_type") == "ask_clarification" else ClarificationTriageAction(**action)
        
        if action.action_type == "ask_clarification":
            if self.turn >= self.MAX_TURNS:
                res = StepResult(reward={"score": 0.01}, done=True)
            else:
                answer = next((qa['answer'] for qa in self.current_issue.get('clarification_qa', []) if any(kw in action.question.lower() for kw in qa.get('keywords', []))), "No info.")
                self.history.append({"question": action.question, "answer": answer})
                self.turn += 1
                res = StepResult(observation=self._current_observation().model_dump(), reward={"score": 0.01}, done=False)
        else:
            base_reward = self.grader.grade(FullTriageAction(label=action.label, priority=action.priority, suggested_assignee=action.suggested_assignee, suggested_component=action.suggested_component), self.current_issue)
            raw_base = (base_reward.score - 0.01) / 0.98
            final_raw = max(0.0, raw_base - (self.turn * self.TURN_PENALTY))
            reward = ClarificationReward(score=clamp_score(final_raw), base_triage_score=base_reward.score, turn_penalty=self.turn * self.TURN_PENALTY, turns_taken=self.turn, label_correct=base_reward.label_correct, priority_correct=base_reward.priority_correct, breakdown={"final": final_raw})
            res = StepResult(reward=reward.model_dump(), done=True)
        return StepResult(**recursive_clamp(res.model_dump()))

    def _current_observation(self):
        return ClarificationObservation(issue=to_issue_obs(self.current_issue, self.store.project_map), turn=self.turn, max_turns=self.MAX_TURNS, clarification_history=self.history.copy())

    def get_state(self): return {"issue": self.current_issue, "turn": self.turn, "history": self.history}
    def restore_state(self, state):
        self.current_issue = state.get("issue")
        self.turn = state.get("turn", 0)
        self.history = state.get("history", [])
