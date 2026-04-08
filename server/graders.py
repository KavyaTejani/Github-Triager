from typing import Dict, List, Any, Optional, Union
from models import (
    LabelClassificationAction, LabelClassificationReward,
    FullTriageAction, FullTriageReward,
    BatchTriageAction, BatchTriageReward
)

def clamp_score(score: float) -> float:
    """Strictly maps [0, 1] to [0.05, 0.95] to avoid any boundary issues."""
    return round(0.05 + (max(0.0, min(1.0, score)) * 0.90), 4)

def recursive_clamp(data: Any) -> Any:
    """Recursively clamps all floats/ints in a dictionary or list to (0.05, 0.95)."""
    if isinstance(data, bool):
        return data
    if isinstance(data, (float, int)):
        return clamp_score(float(data))
    elif isinstance(data, dict):
        return {k: recursive_clamp(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_clamp(v) for v in data]
    return data

class LabelClassificationGrader:
    def grade(self, action: LabelClassificationAction, gold: Dict) -> LabelClassificationReward:
        correct = action.label.value == gold['gold_label']
        raw_score = 1.0 if correct else 0.0
        return LabelClassificationReward(
            score=clamp_score(raw_score),
            correct=correct,
            expected_label=gold['gold_label'],
            predicted_label=action.label.value
        )

class FullTriageGrader:
    PRIORITY_ORDER = ["critical", "high", "medium", "low"]

    def grade(self, action: FullTriageAction, gold: Dict) -> FullTriageReward:
        breakdown = {}
        label_match = action.label.value == gold['gold_label']
        breakdown['label'] = 0.4 if label_match else 0.0
        
        try:
            dist = abs(self.PRIORITY_ORDER.index(action.priority.value) - self.PRIORITY_ORDER.index(gold['gold_priority']))
            breakdown['priority'] = max(0.0, 0.3 - (dist * 0.1))
        except:
            breakdown['priority'] = 0.0
            
        breakdown['assignee'] = 0.15 if action.suggested_assignee == gold.get('gold_assignee') else 0.0
        breakdown['component'] = 0.15 if action.suggested_component == gold.get('gold_component') else 0.0
        
        raw_score = sum(breakdown.values())
        return FullTriageReward(
            score=clamp_score(raw_score),
            label_correct=label_match,
            priority_correct=(action.priority.value == gold['gold_priority']),
            assignee_correct=(action.suggested_assignee == gold.get('gold_assignee')),
            component_correct=(action.suggested_component == gold.get('gold_component')),
            breakdown=breakdown
        )

class BatchTriageGrader:
    def __init__(self):
        self.full_triage_grader = FullTriageGrader()
        self.trajectory_actions = []
        self.trajectory_golds = []

    def grade_step(self, action: BatchTriageAction, gold: Dict) -> BatchTriageReward:
        base_reward = self.full_triage_grader.grade(action, gold)
        dup_bonus = 0.2 if gold.get('duplicate_of') == action.is_duplicate_of else (
            -0.1 if action.is_duplicate_of else 0.0
        )
        raw_step = (base_reward.score - 0.05) / 0.90 + dup_bonus
        clamped_step = clamp_score(raw_step)
        
        self.trajectory_actions.append(action.model_dump(mode='json'))
        self.trajectory_golds.append(gold)

        return BatchTriageReward(
            score=clamped_step, step_score=clamped_step,
            trajectory_score=0.05, breakdown=base_reward.breakdown, is_trajectory_final=False
        )

    def grade_trajectory(self) -> BatchTriageReward:
        total = 0.0
        n = len(self.trajectory_actions)
        if n == 0: return BatchTriageReward(score=0.05, step_score=0.05, trajectory_score=0.05, is_trajectory_final=True)

        for act, gold in zip(self.trajectory_actions, self.trajectory_golds):
            s = 0.0
            if act['label'] == gold['gold_label']: s += 0.4
            if act['priority'] == gold['gold_priority']: s += 0.3
            if act.get('suggested_assignee') == gold.get('gold_assignee'): s += 0.15
            if act.get('suggested_component') == gold.get('gold_component'): s += 0.15
            total += s

        avg_step = total / n
        final_clamped = clamp_score(avg_step)
        return BatchTriageReward(
            score=final_clamped, step_score=clamp_score(avg_step),
            trajectory_score=final_clamped, is_trajectory_final=True,
            breakdown={"avg": avg_step}
        )

    def reset(self):
        self.trajectory_actions = []
        self.trajectory_golds = []
