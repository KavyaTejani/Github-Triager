import pytest
from server.graders import (
    LabelClassificationGrader, FullTriageGrader, BatchTriageGrader
)
from models import (
    LabelClassificationAction, FullTriageAction, BatchTriageAction,
    LabelEnum, PriorityEnum
)

class TestLabelClassificationGrader:
    def test_correct_label(self):
        grader = LabelClassificationGrader()
        action = LabelClassificationAction(label=LabelEnum.BUG)
        gold = {"gold_label": "bug"}
        reward = grader.grade(action, gold)
        # 0.01 + 1.0 * 0.98 = 0.99
        assert reward.score == 0.99
        assert reward.correct is True
        assert reward.expected_label == "bug"
        assert reward.predicted_label == "bug"

    def test_wrong_label(self):
        grader = LabelClassificationGrader()
        action = LabelClassificationAction(label=LabelEnum.FEATURE)
        gold = {"gold_label": "bug"}
        reward = grader.grade(action, gold)
        # 0.01 + 0.0 * 0.98 = 0.01
        assert reward.score == 0.01
        assert reward.correct is False
        assert reward.expected_label == "bug"
        assert reward.predicted_label == "feature"


class TestFullTriageGrader:
    def test_perfect_score(self):
        grader = FullTriageGrader()
        action = FullTriageAction(
            label=LabelEnum.BUG,
            priority=PriorityEnum.HIGH,
            suggested_assignee="backend_team",
            suggested_component="api"
        )
        gold = {
            "gold_label": "bug",
            "gold_priority": "high",
            "gold_assignee": "backend_team",
            "gold_component": "api"
        }
        reward = grader.grade(action, gold)
        assert reward.score == 0.99
        assert reward.label_correct is True
        assert reward.priority_correct is True
        assert reward.assignee_correct is True
        assert reward.component_correct is True

    def test_partial_credit_priority(self):
        grader = FullTriageGrader()
        # Correct label, but wrong priority (one step away)
        action = FullTriageAction(
            label=LabelEnum.BUG,
            priority=PriorityEnum.MEDIUM, # High is gold
            suggested_assignee=None,
            suggested_component=None
        )
        gold = {
            "gold_label": "bug",
            "gold_priority": "high",
            "gold_assignee": "backend_team",
            "gold_component": "api"
        }
        reward = grader.grade(action, gold)
        # raw: label (0.4) + priority partial (0.3 - 0.1 = 0.2) = 0.6
        # clamped: 0.01 + 0.6 * 0.98 = 0.598
        assert reward.score == 0.598
        assert reward.label_correct is True
        assert reward.priority_correct is False


class TestBatchTriageGrader:
    def test_trajectory_scoring(self):
        grader = BatchTriageGrader()
        gold_issues = [
            {
                "issue_id": "i1",
                "gold_label": "bug", "gold_priority": "high",
                "gold_assignee": "backend_team", "gold_component": "api",
                "duplicate_of": None, "batch_group": "bg1"
            },
            {
                "issue_id": "i2",
                "gold_label": "feature", "gold_priority": "low",
                "gold_assignee": "frontend_team", "gold_component": "ui",
                "duplicate_of": None, "batch_group": "bg1"
            },
        ]
        
        # Step 1
        action1 = BatchTriageAction(
            label=LabelEnum.BUG, priority=PriorityEnum.HIGH,
            suggested_assignee="backend_team", suggested_component="api"
        )
        reward1 = grader.grade_step(action1, gold_issues[0])
        assert reward1.step_score == 0.99
        
        # Step 2
        action2 = BatchTriageAction(
            label=LabelEnum.FEATURE, priority=PriorityEnum.LOW,
            suggested_assignee="frontend_team", suggested_component="ui"
        )
        reward2 = grader.grade_step(action2, gold_issues[1])
        assert reward2.step_score == 0.99
        
        # Trajectory
        final = grader.grade_trajectory()
        assert final.is_trajectory_final is True
        # Both steps perfect (1.0 each, avg 1.0)
        # Trajectory score will be clamped
        assert final.trajectory_score > 0.9
