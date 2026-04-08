from typing import Dict, List, Any, Optional
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

class LabelClassificationGrader:
    """Deterministic grader for Task 1 (Easy)."""

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
    """Deterministic grader for Task 2 (Medium)."""

    PRIORITY_ORDER = ["critical", "high", "medium", "low"]

    def grade(self, action: FullTriageAction, gold: Dict) -> FullTriageReward:
        breakdown = {}

        label_correct = action.label.value == gold['gold_label']
        breakdown['label'] = 0.4 if label_correct else 0.0

        priority_correct = action.priority.value == gold['gold_priority']
        if priority_correct:
            breakdown['priority'] = 0.3
        else:
            # Partial credit for being close in priority
            try:
                distance = abs(
                    self.PRIORITY_ORDER.index(action.priority.value) -
                    self.PRIORITY_ORDER.index(gold['gold_priority'])
                )
                breakdown['priority'] = max(0.0, 0.3 - (distance * 0.1))
            except ValueError:
                breakdown['priority'] = 0.0

        assignee_correct = (
            action.suggested_assignee is not None and
            action.suggested_assignee == gold.get('gold_assignee')
        )
        breakdown['assignee'] = 0.15 if assignee_correct else 0.0

        component_correct = (
            action.suggested_component is not None and
            action.suggested_component == gold.get('gold_component')
        )
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
    """
    Deterministic grader for Task 3 (Hard).
    Computes per-step scores AND trajectory-level bonuses/penalties.
    """

    def __init__(self):
        self.full_triage_grader = FullTriageGrader()
        self.trajectory_actions: List[Dict] = []
        self.trajectory_golds: List[Dict] = []

    def grade_step(self, action: BatchTriageAction, gold: Dict) -> BatchTriageReward:
        """Grade a single step within the batch. Returns partial score."""
        base_action = FullTriageAction(
            label=action.label,
            priority=action.priority,
            suggested_assignee=action.suggested_assignee,
            suggested_component=action.suggested_component
        )
        base_reward = self.full_triage_grader.grade(base_action, gold)
        
        dup_bonus = 0.0
        if gold.get('duplicate_of') is not None:
            if action.is_duplicate_of == gold['duplicate_of']:
                dup_bonus = 0.2
            elif action.is_duplicate_of is not None:
                dup_bonus = -0.1

        # Use 0.01-0.99 range for step_score too
        raw_step_score = (base_reward.score - 0.01) / 0.98 + dup_bonus
        clamped_step_score = clamp_score(raw_step_score)
        
        self.trajectory_actions.append(action.model_dump(mode='json'))
        self.trajectory_golds.append(gold)

        return BatchTriageReward(
            score=clamped_step_score,
            step_score=clamped_step_score,
            duplicate_detection_bonus=dup_bonus,
            breakdown=base_reward.breakdown,
            is_trajectory_final=False
        )

    def grade_trajectory(self) -> BatchTriageReward:
        """
        Compute trajectory-level bonuses/penalties at episode end.
        Called after all steps in the batch are complete.
        """
        total_step_score = 0.0
        n = len(self.trajectory_actions)
        if n == 0:
            return BatchTriageReward(
                score=0.01,
                step_score=0.01,
                trajectory_score=0.01,
                is_trajectory_final=True,
                breakdown={}
            )

        # Re-calculate avg step score
        for action_data, gold in zip(self.trajectory_actions, self.trajectory_golds):
            base_action = FullTriageAction(
                label=action_data['label'],
                priority=action_data['priority'],
                suggested_assignee=action_data.get('suggested_assignee'),
                suggested_component=action_data.get('suggested_component')
            )
            label_match = base_action.label.value == gold['gold_label']
            prio_match = base_action.priority.value == gold['gold_priority']
            s = 0.0
            if label_match: s += 0.4
            if prio_match: s += 0.3
            if base_action.suggested_assignee == gold.get('gold_assignee'): s += 0.15
            if base_action.suggested_component == gold.get('gold_component'): s += 0.15
            total_step_score += s

        avg_step = total_step_score / n

        # Workload balance bonus
        assignee_counts: Dict[str, int] = {}
        for a in self.trajectory_actions:
            assignee = a.get('suggested_assignee')
            if assignee:
                assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
        
        if len(assignee_counts) > 1:
            values = list(assignee_counts.values())
            imbalance = (max(values) - min(values)) / n
            workload_bonus = max(0.0, 0.15 * (1.0 - imbalance))
        else:
            workload_bonus = 0.0

        # Consistency penalty
        consistency_penalty = 0.0
        for i in range(1, len(self.trajectory_actions)):
            prev = self.trajectory_actions[i - 1]
            curr = self.trajectory_actions[i]
            prev_gold = self.trajectory_golds[i - 1]
            curr_gold = self.trajectory_golds[i]
            if prev_gold.get('gold_label') == curr_gold.get('gold_label'):
                if prev.get('label') != curr.get('label'):
                    consistency_penalty -= 0.05

        trajectory_score = avg_step + workload_bonus + consistency_penalty
        clamped_trajectory_score = clamp_score(trajectory_score)

        return BatchTriageReward(
            score=clamped_trajectory_score,
            step_score=clamp_score(avg_step),
            trajectory_score=clamped_trajectory_score,
            workload_balance_bonus=round(workload_bonus, 4),
            consistency_penalty=round(consistency_penalty, 4),
            is_trajectory_final=True,
            breakdown={
                "avg_step_score": round(avg_step, 4),
                "workload_bonus": round(workload_bonus, 4),
                "consistency_penalty": round(consistency_penalty, 4)
            }
        )

    def reset(self):
        """Clear trajectory state for a new episode."""
        self.trajectory_actions.clear()
        self.trajectory_golds.clear()
