from typing import Dict, List, Any, Optional, Union
from models import (
    LabelClassificationAction, LabelClassificationReward,
    FullTriageAction, FullTriageReward,
    BatchTriageAction, BatchTriageReward
)

def clamp_score(score: float) -> float:
    """
    Ensures the score is strictly between 0 and 1.
    Maps [0, 1] to [0.01, 0.99] to satisfy validation requirements.
    """
    return round(0.01 + (max(0.0, min(1.0, score)) * 0.98), 4)

def recursive_clamp(data: Any) -> Any:
    """
    Recursively clamps all floats in a dictionary or list to (0.01, 0.99).
    This is an absolute safety layer for the validator.
    """
    if isinstance(data, float):
        # Only clamp values that look like scores/probabilities
        # This prevents breaking non-score floats (though unlikely here)
        return clamp_score(data)
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
        label_correct = action.label.value == gold['gold_label']
        breakdown['label'] = 0.4 if label_correct else 0.0
        priority_correct = action.priority.value == gold['gold_priority']
        if priority_correct:
            breakdown['priority'] = 0.3
        else:
            try:
                distance = abs(
                    self.PRIORITY_ORDER.index(action.priority.value) -
                    self.PRIORITY_ORDER.index(gold['gold_priority'])
                )
                breakdown['priority'] = max(0.0, 0.3 - (distance * 0.1))
            except ValueError:
                breakdown['priority'] = 0.0
        assignee_correct = (action.suggested_assignee == gold.get('gold_assignee'))
        breakdown['assignee'] = 0.15 if assignee_correct else 0.0
        component_correct = (action.suggested_component == gold.get('gold_component'))
        breakdown['component'] = 0.15 if component_correct else 0.0
        
        raw_score = sum(breakdown.values())
        return FullTriageReward(
            score=clamp_score(raw_score),
            label_correct=label_correct,
            priority_correct=priority_correct,
            assignee_correct=assignee_correct,
            component_correct=component_correct,
            breakdown=breakdown
        )

class BatchTriageGrader:
    def __init__(self):
        self.full_triage_grader = FullTriageGrader()
        self.trajectory_actions: List[Dict] = []
        self.trajectory_golds: List[Dict] = []

    def grade_step(self, action: BatchTriageAction, gold: Dict) -> BatchTriageReward:
        base_action = FullTriageAction(
            label=action.label,
            priority=action.priority,
            suggested_assignee=action.suggested_assignee,
            suggested_component=action.suggested_component
        )
        base_reward = self.full_triage_grader.grade(base_action, gold)
        dup_bonus = 0.2 if gold.get('duplicate_of') == action.is_duplicate_of else (
            -0.1 if action.is_duplicate_of else 0.0
        )
        raw_step_score = (base_reward.score - 0.01) / 0.98 + dup_bonus
        clamped_step_score = clamp_score(raw_step_score)
        
        self.trajectory_actions.append(action.model_dump(mode='json'))
        self.trajectory_golds.append(gold)

        return BatchTriageReward(
            score=clamped_step_score,
            step_score=clamped_step_score,
            trajectory_score=0.01,
            duplicate_detection_bonus=dup_bonus,
            breakdown=base_reward.breakdown,
            is_trajectory_final=False
        )

    def grade_trajectory(self) -> BatchTriageReward:
        total_step_score = 0.0
        n = len(self.trajectory_actions)
        if n == 0:
            return BatchTriageReward(score=0.01, step_score=0.01, trajectory_score=0.01, is_trajectory_final=True)

        for action_data, gold in zip(self.trajectory_actions, self.trajectory_golds):
            s = 0.0
            if action_data['label'] == gold['gold_label']: s += 0.4
            if action_data['priority'] == gold['gold_priority']: s += 0.3
            if action_data.get('suggested_assignee') == gold.get('gold_assignee'): s += 0.15
            if action_data.get('suggested_component') == gold.get('gold_component'): s += 0.15
            total_step_score += s

        avg_step = total_step_score / n
        assignee_counts = {}
        for a in self.trajectory_actions:
            ass = a.get('suggested_assignee')
            if ass: assignee_counts[ass] = assignee_counts.get(ass, 0) + 1
        
        workload_bonus = 0.15 * (1.0 - (max(assignee_counts.values()) - min(assignee_counts.values()))/n) if len(assignee_counts) > 1 else 0.0
        consistency_penalty = 0.0
        for i in range(1, n):
            if self.trajectory_golds[i-1].get('gold_label') == self.trajectory_golds[i].get('gold_label'):
                if self.trajectory_actions[i-1].get('label') != self.trajectory_actions[i].get('label'):
                    consistency_penalty -= 0.05

        final_raw = avg_step + workload_bonus + consistency_penalty
        clamped_traj = clamp_score(final_raw)
        return BatchTriageReward(
            score=clamped_traj,
            step_score=clamp_score(avg_step),
            trajectory_score=clamped_traj,
            workload_balance_bonus=workload_bonus,
            consistency_penalty=consistency_penalty,
            is_trajectory_final=True,
            breakdown={"avg": avg_step, "workload": workload_bonus, "consistency": consistency_penalty}
        )

    def reset(self):
        self.trajectory_actions.clear()
        self.trajectory_golds.clear()
