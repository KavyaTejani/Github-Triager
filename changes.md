# GitHub-Triager Improvement Roadmap

This document details the proposed changes to transform the current project into a robust, high-signal reinforcement learning environment. These changes address the root causes of the recent training failure.

## 1. Engineering Stability & State Management

### Fix 1.1: Trajectory State Restoration
*   **Problem:** `BatchTriageTask` saves its current index but the `BatchTriageGrader` loses the history of actions for the first $N-1$ issues when a state is restored.
*   **Change:** Implement `get_state()` and `restore_state()` methods in `BatchTriageGrader`. Nest this state within `BatchTriageTask.get_state()`.
*   **Impact:** Ensures that rewards for batch tasks are calculated correctly even if the environment is serialized or moved between workers during a rollout.

### Fix 1.2: Enforce Data Immutability
*   **Problem:** `IssueStore` returns direct references to the issue dictionaries. If an agent or the environment logic mutates an issue (e.g., adding a temporary field), it pollutes the global dataset for all other training sessions.
*   **Change:** Wrap all issue returns in `copy.deepcopy()`.
*   **Impact:** Guarantees that each episode starts with "clean" gold data, preventing cross-episode data leakage.

### Fix 1.3: Unified Session Management
*   **Problem:** The `/ws` (WebSocket) endpoint uses a local memory dictionary, while HTTP endpoints use a global session store. This prevents "hybrid" training/evaluation and breaks horizontal scaling.
*   **Change:** Refactor `websocket_endpoint` in `app.py` to use the centralized `session_store`.
*   **Impact:** Allows sessions to be shared across transport layers and enables scaling the environment across multiple server pods.

---

## 2. RL Environment Logic (Reward & Signal)

### Fix 2.1: Stronger Reward Signal (Range Widening)
*   **Problem:** `clamp_score` squashes all performance into a tiny range [0.1, 0.8]. RL optimizers like GRPO struggle to differentiate between "good" and "great" when the delta is only 0.001.
*   **Change:** Widen the reward range to `[0.01, 0.95]`.
*   **Impact:** Provides a much steeper gradient for the optimizer to follow, accelerating convergence.

### Fix 2.2: Immediate Credit Assignment (Clarification Task)
*   **Problem:** The `turn_penalty` is only applied at the end of the episode. The model receives a positive 0.01 for "asking a question" and only sees the "bill" many steps later.
*   **Change:** Apply the turn penalty (or a "step cost") immediately on every turn where a question is asked.
*   **Impact:** Helps the model learn the cost-benefit trade-off of asking questions vs. triaging immediately.

### Fix 2.3: Configurable Reward Weights
*   **Problem:** Reward weights (Label=0.4, Priority=0.3, etc.) are hardcoded.
*   **Change:** Move these constants into a configuration class or the `openenv.yaml` file.
*   **Impact:** Allows for easy ablation studies and tuning of the environment's priorities without code changes.

---

## 3. Inference & Evaluation (The "Honest Baseline")

### Fix 3.1: Remove Hallucinated Fallbacks
*   **Problem:** `inference.py` catches JSON errors and submits a "standard" action. This masks model failures and makes the "Before training" score look artificially high.
*   **Change:** If JSON parsing fails, the script should submit a "MalformedAction" or simply terminate the episode with a minimum reward.
*   **Impact:** Forces the model to learn proper JSON formatting as part of the RL loop and ensures evaluation graphs represent actual model capability.

### Fix 3.2: Robust Log Parsing
*   **Problem:** The current print statements for `[STEP]` and `[END]` are slightly fragile and manually formatted.
*   **Change:** Use a structured logging approach or a dedicated helper to ensure compliance with the OpenEnv log parser requirements.
*   **Impact:** Prevents "bad graph" issues caused by the evaluator failing to parse output strings correctly.

---

## 4. Maintenance & Developer Experience

### Fix 4.1: Update .gitignore
*   **Problem:** Training checkpoints (`outputs/`) and Unsloth caches are flooding VS Code.
*   **Change:** Add these to `.gitignore`.
*   **Impact:** Cleaner developer experience and prevents accidental multi-gigabyte commits.

### Fix 4.2: Robust Data Loading
*   **Problem:** `IssueStore` silently fails if data is missing.
*   **Change:** Raise explicit `FileNotFoundError` or `ValueError` if the simulation JSON is empty or missing.
*   **Impact:** Prevents "ghost" training runs where the model is triaging empty data.
