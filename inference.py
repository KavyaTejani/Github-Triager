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
from typing import List, Optional, Dict, Any
from openai import OpenAI
from client import GitHubTriagerClient

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def safe_str(s: Any) -> str:
    """Bulletproof string formatting for validation compliance."""
    try:
        val = float(s)
        clamped = max(0.01, min(0.99, val))
        return f"{clamped:.4f}"
    except:
        return "0.0100"

def run_task(env: GitHubTriagerClient, task_id: str):
    # MANDATORY START LOG
    print(f"[START] task_id=\"{task_id}\"", flush=True)
    try:
        observation = env.reset(task_id=task_id)
        last_score = 0.01
        step_count = 0
        for step in range(1, 15):
            step_count = step
            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": "Respond with ONLY JSON."}, {"role": "user", "content": str(observation)}],
                    temperature=0.2, max_tokens=300
                )
                action = json.loads(re.search(r'\{[^}]+\}', response.choices[0].message.content, re.DOTALL).group(0))
            except:
                action = {"label": "bug", "priority": "high", "confidence": 0.5}

            result = env.step(action)
            reward_data = result.get("reward", {})
            
            # The server now returns clamped scores per my latest updates
            current_reward = float(reward_data.get("score", 0.01))
            
            # Mandatory Step Log
            print(f"[STEP] step={step}, score={safe_str(current_reward)}, done={result.get('done')}", flush=True)
            
            last_score = current_reward # In this env, final 'score' is the total performance
            
            if result.get("done"): break
            observation = result.get("observation", observation)

        # MANDATORY END LOG (requested format)
        # print(f"[END] task={task_id} score={final_score} steps={n}", flush=True)
        final_clamped = max(0.01, min(0.99, float(last_score)))
        print(f"[END] task={task_id} score={final_clamped:.4f} steps={step_count}", flush=True)
        
    except Exception as e:
        print(f"[END] task={task_id} score=0.0100 steps=0", flush=True)

def main():
    with GitHubTriagerClient(base_url="http://localhost:8000") as env:
        try: env.health()
        except: return
        for tid in ["label_classification", "full_triage", "batch_triage_with_context", "clarification_triage"]:
            run_task(env, tid)

if __name__ == "__main__":
    main()
