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
        self.issues = []
        self.project_map = {}
        try:
            with open(data_path, 'r') as f: self.issues = json.load(f)
            with open(project_structure_path, 'r') as f: self.project_map = json.load(f)
        except: pass

    def get_random_issue(self): return random.choice(self.issues)
    def get_balanced_batch(self, size=10): return random.sample(self.issues, min(size, len(self.issues)))

def to_obs(issue, pm):
    return IssueObservation(
        issue_id=issue['issue_id'], title=issue['title'], body=issue['body'],
        author=issue['author'], created_at=issue['created_at'], labels=[],
        project_map=pm, repository_context={"primary_language": "Python", "stars": 4821}
    )

class LabelClassificationTask:
    def __init__(self, store):
        self.store = store
        self.grader = LabelClassificationGrader()
        self.current_issue = None

    def reset(self):
        self.current_issue = self.store.get_random_issue()
        return LabelClassificationObservation(issue=to_obs(self.current_issue, self.store.project_map))

    def step(self, action):
        reward = self.grader.grade(action, self.current_issue)
        return StepResult(reward=recursive_clamp(reward.model_dump()), done=True, info={"id": self.current_issue['issue_id']})

    def get_state(self): return {"issue": self.current_issue}
    def restore_state(self, state): self.current_issue = state.get("issue")

class FullTriageTask:
    def __init__(self, store):
        self.store = store
        self.grader = FullTriageGrader()
        self.current_issue = None

    def reset(self):
        self.current_issue = self.store.get_random_issue()
        return FullTriageObservation(issue=to_obs(self.current_issue, self.store.project_map))

    def step(self, action):
        reward = self.grader.grade(action, self.current_issue)
        return StepResult(reward=recursive_clamp(reward.model_dump()), done=True, info={"id": self.current_issue['issue_id']})

    def get_state(self): return {"issue": self.current_issue}
    def restore_state(self, state): self.current_issue = state.get("issue")

class BatchTriageTask:
    def __init__(self, store, batch_size=10):
        self.store = store
        self.grader = BatchTriageGrader()
        self.batch = []
        self.idx = 0

    def reset(self):
        self.grader.reset()
        self.batch = self.store.get_balanced_batch()
        self.idx = 0
        return self._obs()

    def step(self, action):
        current = self.batch[self.idx]
        step_reward = self.grader.grade_step(action, current)
        self.idx += 1
        done = self.idx >= len(self.batch)
        if done:
            reward = self.grader.grade_trajectory()
            return StepResult(reward=recursive_clamp(reward.model_dump()), done=True)
        return StepResult(observation=self._obs().model_dump(), reward=recursive_clamp(step_reward.model_dump()), done=False)

    def _obs(self):
        return BatchTriageObservation(issue=to_obs(self.batch[self.idx], self.store.project_map), batch_position=self.idx, batch_size=len(self.batch))

    def get_state(self): return {"batch": self.batch, "idx": self.idx}
    def restore_state(self, state): self.batch, self.idx = state.get("batch"), state.get("idx")

class ClarificationTask:
    def __init__(self, store):
        self.store = store
        self.grader = FullTriageGrader()
        self.issue = None
        self.turn = 0

    def reset(self):
        vague = [i for i in self.store.issues if i.get('clarification_qa')]
        self.issue = random.choice(vague) if vague else random.choice(self.store.issues)
        self.turn = 0
        return self._obs()

    def step(self, action):
        if isinstance(action, dict):
            action = ClarificationRequest(**action) if action.get("action_type") == "ask_clarification" else ClarificationTriageAction(**action)
        
        if action.action_type == "ask_clarification":
            if self.turn >= 3: return StepResult(reward={"score": 0.05}, done=True)
            self.turn += 1
            return StepResult(observation=self._obs().model_dump(), reward={"score": 0.05}, done=False)
        
        base_reward = self.grader.grade(action, self.issue)
        raw = (base_reward.score - 0.05) / 0.90
        final = clamp_score(max(0.0, raw - (self.turn * 0.08)))
        return StepResult(reward={"score": final}, done=True)

    def _obs(self):
        return ClarificationObservation(issue=to_obs(self.issue, self.store.project_map), turn=self.turn, max_turns=3)

    def get_state(self): return {"issue": self.issue, "turn": self.turn}
    def restore_state(self, state): self.issue, self.turn = state.get("issue"), state.get("turn")
