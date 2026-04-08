from typing import Dict, List, Any, Optional
from models import (
    LabelClassificationAction, LabelClassificationReward,
    FullTriageAction, FullTriageReward,
    BatchTriageAction, BatchTriageReward
)

def clamp_score(score: float) -> float:
    """Strictly maps [0, 1] to [0.1, 0.8] to ensure total sums stay < 1.0."""
    return round(0.1 + (max(0.0, min(1.0, score)) * 0.7), 4)

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

    def grade_step(self, action: BatchTriageAction, gold: Dict) -> float:
        """Returns a tiny non-zero reward for intermediate steps."""
        self.trajectory_actions.append(action.model_dump(mode='json'))
        self.trajectory_golds.append(gold)
        return 0.01

    def grade_trajectory(self) -> BatchTriageReward:
        """Calculates final clamped score for the entire batch."""
        total = 0.0
        n = len(self.trajectory_actions)
        if n == 0: return BatchTriageReward(score=0.1, step_score=0.1, trajectory_score=0.1, is_trajectory_final=True)

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
            score=final_clamped, step_score=0.01,
            trajectory_score=final_clamped, is_trajectory_final=True,
            breakdown={"avg": avg_step}
        )

    def reset(self):
        self.trajectory_actions = []
        self.trajectory_golds = []
