import json
import random
import logging
import copy
import os
from typing import List, Dict, Optional, Any, Union

from models import (
    IssueObservation,
    LabelClassificationObservation, LabelClassificationAction,
    FullTriageObservation, FullTriageAction,
    BatchTriageObservation, BatchTriageAction,
    StepResult,
    ClarificationRequest, ClarificationTriageAction, ClarificationObservation, ClarificationReward
)
from server.graders import LabelClassificationGrader, FullTriageGrader, BatchTriageGrader, clamp_score
from server.config import config

class IssueStore:
    def __init__(self, data_path: str = None,
                 project_structure_path: str = None):
        # Resolve paths relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_path is None:
            data_path = os.path.join(base_dir, "data", "simulated_issues.json")
        if project_structure_path is None:
            project_structure_path = os.path.join(base_dir, "data", "project_structure.json")
            
        self.issues, self.project_map = [], {}
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Issue data not found at {data_path}")
            
        with open(data_path, 'r') as f: 
            self.issues = json.load(f)
        with open(project_structure_path, 'r') as f: 
            self.project_map = json.load(f)
            
        if not self.issues:
            raise ValueError(f"Issue store is empty. Check {data_path}")

    def get_random_issue(self): 
        return copy.deepcopy(random.choice(self.issues))

    def get_balanced_batch(self, size=10): 
        batch = random.sample(self.issues, min(size, len(self.issues)))
        return copy.deepcopy(batch)

def to_obs(issue, pm, idx=0, size=1):
    return IssueObservation(
        issue_id=issue['issue_id'], title=issue['title'], body=issue['body'],
        author=issue['author'], created_at=issue['created_at'], labels=[],
        project_map=pm, repository_context={"primary_language": "Python", "stars": 4821}
    )

class LabelClassificationTask:
    def __init__(self, store):
        self.store, self.grader, self.issue = store, LabelClassificationGrader(), None
    def reset(self):
        self.issue = self.store.get_random_issue()
        return LabelClassificationObservation(issue=to_obs(self.issue, self.store.project_map))
    def step(self, action):
        reward = self.grader.grade(action, self.issue)
        return StepResult(reward=reward.model_dump(), done=True)
    def get_state(self): return {"issue": self.issue}
    def restore_state(self, state): self.issue = state.get("issue")

class FullTriageTask:
    def __init__(self, store):
        self.store, self.grader, self.issue = store, FullTriageGrader(), None
    def reset(self):
        self.issue = self.store.get_random_issue()
        return FullTriageObservation(issue=to_obs(self.issue, self.store.project_map))
    def step(self, action):
        reward = self.grader.grade(action, self.issue)
        return StepResult(reward=reward.model_dump(), done=True)
    def get_state(self): return {"issue": self.issue}
    def restore_state(self, state): self.issue = state.get("issue")

class BatchTriageTask:
    def __init__(self, store, size=10):
        self.store, self.grader, self.batch, self.idx, self.size = store, BatchTriageGrader(), [], 0, size
    def reset(self):
        self.grader.reset()
        self.batch, self.idx = self.store.get_balanced_batch(self.size), 0
        return self._obs()
    def step(self, action):
        _ = self.grader.grade_step(action, self.batch[self.idx])
        self.idx += 1
        done = self.idx >= len(self.batch)
        if done:
            reward = self.grader.grade_trajectory()
            return StepResult(reward=reward.model_dump(), done=True)
        return StepResult(observation=self._obs().model_dump(), reward={"score": 0.01}, done=False)
    def _obs(self):
        return BatchTriageObservation(issue=to_obs(self.batch[self.idx], self.store.project_map), batch_position=self.idx, batch_size=len(self.batch))
    def get_state(self): 
        return {
            "batch": self.batch, 
            "idx": self.idx, 
            "grader_state": self.grader.get_state()
        }
    def restore_state(self, state): 
        self.batch = state.get("batch")
        self.idx = state.get("idx")
        self.grader.restore_state(state.get("grader_state", {}))

class ClarificationTask:
    def __init__(self, store):
        self.store, self.grader, self.issue, self.turn = store, FullTriageGrader(), None, 0
        self.clarification_history = []

    def reset(self):
        v = [i for i in self.store.issues if i.get('clarification_qa')]
        self.issue, self.turn = random.choice(v) if v else random.choice(self.store.issues), 0
        self.clarification_history = []
        return self._obs()

    def step(self, action):
        if isinstance(action, dict):
            if action.get("action_type") == "ask_clarification":
                action = ClarificationRequest(**action)
            else:
                action = ClarificationTriageAction(**action)
        
        # Immediate penalty for asking questions
        if action.action_type == "ask_clarification":
            self.clarification_history.append({"role": "agent", "content": action.question})
            if self.turn >= 3: 
                return StepResult(reward={"score": 0.01}, done=True)
            self.turn += 1
            
            # Simulate a user response from the gold data if available
            user_response = "Please provide more details."
            qa = self.issue.get('clarification_qa', [])
            if len(qa) >= self.turn:
                user_response = qa[self.turn-1].get('answer', user_response)
            self.clarification_history.append({"role": "user", "content": user_response})

            return StepResult(observation=self._obs().model_dump(), reward={"score": 0.001}, done=False)
            
        base = self.grader.grade(action, self.issue)
        # Final score calculation with turn penalty applied to raw score
        raw = (base.score - config.REWARD_MIN) / (config.REWARD_MAX - config.REWARD_MIN) # Unclamp to [0, 1]
        final = clamp_score(max(0.0, raw - (self.turn * config.TURN_PENALTY)))
        
        # Breakdown for logging
        reward_info = {
            "score": final,
            "base_triage_score": base.score,
            "turn_penalty": self.turn * config.TURN_PENALTY,
            "turns_taken": self.turn,
            "label_correct": base.label_correct,
            "priority_correct": base.priority_correct
        }
        return StepResult(reward=reward_info, done=True)

    def _obs(self):
        return ClarificationObservation(
            issue=to_obs(self.issue, self.store.project_map), 
            turn=self.turn, 
            max_turns=3,
            clarification_history=self.clarification_history
        )
    def get_state(self): 
        return {
            "issue": self.issue, 
            "turn": self.turn, 
            "clarification_history": copy.deepcopy(self.clarification_history)
        }
    def restore_state(self, state): 
        self.issue = state.get("issue")
        self.turn = state.get("turn")
        self.clarification_history = state.get("clarification_history", [])
