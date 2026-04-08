"""
Baseline Inference Script for GitHub Triager
=============================================
MANDATORY REQUIREMENTS (OpenEnv spec):
- Uses the OpenAI API client (openai SDK)
- Reads credentials from mandatory environment variables
- File named 'inference.py' in project root
- Emits structured [START], [STEP], [END] logs
"""

import os
import json
import re
import textwrap
from typing import List, Optional, Dict, Any
from openai import OpenAI

from client import GitHubTriagerClient

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

SYSTEM_PROMPTS = {
    "label_classification": "Respond with ONLY a JSON object: {\"label\": \"bug|feature|documentation|question|enhancement\"}",
    "full_triage": "Triage the issue. Respond with ONLY a JSON object: {\"label\": \"...\", \"priority\": \"critical|high|medium|low\", \"suggested_assignee\": \"...\", \"suggested_component\": \"...\"}",
    "batch_triage_with_context": "Triage the batch issue. Respond with ONLY a JSON object: {\"label\": \"...\", \"priority\": \"...\", \"suggested_assignee\": \"...\", \"suggested_component\": \"...\", \"is_duplicate_of\": \"...\", \"priority_justification\": \"...\"}",
    "clarification_triage": "Vague issue. Ask question: {\"action_type\": \"ask_clarification\", \"question\": \"...\"} OR Submit triage: {\"action_type\": \"submit_triage\", \"label\": \"...\", \"priority\": \"...\", \"suggested_assignee\": \"...\", \"suggested_component\": \"...\", \"confidence\": 0.9}"
}

def safe_score(s: Any) -> str:
    try:
        val = float(s)
        return f"{max(0.01, min(0.99, val)):.4f}"
    except:
        return "0.0100"

def run_task(env: GitHubTriagerClient, task_id: str, max_steps: int = 15):
    print(f"[START] task_id=\"{task_id}\"")
    try:
        observation = env.reset(task_id=task_id)
        last_score = 0.01
        for step in range(1, max_steps + 1):
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": SYSTEM_PROMPTS[task_id]}, {"role": "user", "content": str(observation)}],
                    temperature=0.2, max_tokens=300
                )
                action = json.loads(re.search(r'\{[^}]+\}', response.choices[0].message.content, re.DOTALL).group(0))
            except:
                action = {"label": "question", "priority": "medium", "confidence": 0.5}

            result = env.step(action)
            reward_data = result.get("reward", {})
            last_score = float(reward_data.get("score", 0.01))
            
            print(f"[STEP] step={step}, score={safe_score(last_score)}, done={result.get('done')}")
            if result.get("done"): break
            observation = result.get("observation", observation)

        # For OpenEnv evaluation, the final log must be the clamped total reward
        # In this environment, 'score' in the final step is the trajectory/final total
        print(f"[END] total_reward={safe_score(last_score)}")
    except Exception as e:
        print(f"[END] total_reward=0.0100")

def main():
    with GitHubTriagerClient(base_url="http://localhost:8000") as env:
        try: env.health()
        except: return
        for tid in ["label_classification", "full_triage", "batch_triage_with_context", "clarification_triage"]:
            run_task(env, tid)

if __name__ == "__main__":
    main()
