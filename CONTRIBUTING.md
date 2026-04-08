# Contributing to GitHub Triager

## Setup
1. Clone the repo
2. `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Start server: `uvicorn server.app:app --reload`

## Adding New Issues to the Dataset
Edit `data/simulated_issues.json`. Each issue must have all required fields:
- `issue_id`, `title`, `body`, `author`, `created_at`
- `gold_label`, `gold_priority`, `gold_assignee`, `gold_component`
- `batch_group`
- `clarification_qa` (for Task 4)

Run `pytest tests/test_environment.py` to verify the store still loads correctly.

## Adding a New Task
1. Add Pydantic models to `models.py`.
2. Add task class to `server/environment.py` (implement `reset()`, `step()`, `get_state()`, `restore_state()`).
3. Add grader to `server/graders.py` if needed.
4. Register in `TASK_REGISTRY` in `server/app.py`.
5. Add system prompt and prompt builder logic to `inference.py`.
6. Add unit tests to `tests/`.
