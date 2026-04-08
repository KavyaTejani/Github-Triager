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

# OpenEnv Mandatory Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

TEMPERATURE = 0.2
MAX_TOKENS = 300

llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


SYSTEM_PROMPTS = {
    "label_classification": textwrap.dedent("""
        You are a GitHub issue classifier. Given an issue, respond with ONLY a JSON object:
        {"label": "bug|feature|documentation|question|enhancement"}
    """).strip(),

    "full_triage": textwrap.dedent("""
        You are an expert GitHub maintainer. Triage the issue. Respond with ONLY a JSON object:
        {
            "label": "bug|feature|documentation|question|enhancement",
            "priority": "critical|high|medium|low",
            "suggested_assignee": "team_name or null",
            "suggested_component": "component_name or null"
        }
        Priority guide:
        - critical: System down, data loss, security vulnerability
        - high: Major functionality broken, many users affected
        - medium: Partial breakage, workaround exists
        - low: Minor, cosmetic, edge case
        
        The observation includes a `project_map` field showing which teams own which components.
        Use it to determine the correct `suggested_assignee` and `suggested_component`.
    """).strip(),

    "batch_triage_with_context": textwrap.dedent("""
        You are an expert GitHub maintainer triaging a batch of issues.
        Consider inter-issue context: detect duplicates, balance assignee workload,
        notice priority escalation patterns.
        Respond with ONLY a JSON object:
        {
            "label": "bug|feature|documentation|question|enhancement",
            "priority": "critical|high|medium|low",
            "suggested_assignee": "team_name or null",
            "suggested_component": "component_name or null",
            "is_duplicate_of": "issue_id or null",
            "priority_justification": "brief reason or null"
        }
        
        The observation includes a `project_map` field showing which teams own which components.
        Use it to determine the correct `suggested_assignee` and `suggested_component`.
    """).strip(),

    "clarification_triage": textwrap.dedent("""
        You are triaging a vague GitHub issue. You may ask up to 3 clarifying questions
        before submitting your final triage.
        
        To ask a question, respond with:
        {"action_type": "ask_clarification", "question": "your question here"}
        
        To submit final triage, respond with:
        {
            "action_type": "submit_triage",
            "label": "bug|feature|documentation|question|enhancement",
            "priority": "critical|high|medium|low",
            "suggested_assignee": "team_name or null",
            "suggested_component": "component_name or null",
            "confidence": 0.0-1.0
        }
        
        Strategy: Only ask questions when truly necessary. Each question costs 0.08 from your score.
        If you're already confident, submit immediately.
    """).strip()
}


def build_user_prompt(observation: dict, task_id: str) -> str:
    issue = observation.get("issue", observation)
    
    prompt = ""
    if task_id in ("label_classification", "full_triage"):
        prompt = textwrap.dedent(f"""
            Issue #{issue.get('issue_id')}
            Title: {issue.get('title')}
            Body:
            {issue.get('body', '')[:2000]}
            Author: {issue.get('author')}
            Created: {issue.get('created_at')}
        """).strip()
    elif task_id == "batch_triage_with_context":
        prior = observation.get("prior_triage_decisions", [])
        dups = observation.get("duplicate_candidates", [])
        prompt = textwrap.dedent(f"""
            Issue #{issue.get('issue_id')} (batch position {observation.get('batch_position', 0)+1}/{observation.get('batch_size', 10)})
            Title: {issue.get('title')}
            Body:
            {issue.get('body', '')[:2000]}
            Author: {issue.get('author')}

            Prior triage decisions this batch:
            {json.dumps(prior[-5:], indent=2) if prior else 'None (first issue)'}

            Possible duplicate candidates (match these IDs if you find a duplicate):
            {dups if dups else 'None detected in prior steps'}
        """).strip()
    elif task_id == "clarification_triage":
        history = observation.get("clarification_history", [])
        turn = observation.get("turn", 0)
        max_turns = observation.get("max_turns", 3)
        
        history_text = "\n".join(
            [f"Q: {h['question']}\nA: {h['answer']}" for h in history]
        ) if history else "None yet."
        
        prompt = textwrap.dedent(f"""
            Issue #{issue.get('issue_id')}
            Title: {issue.get('title')}
            Body:
            {issue.get('body', '')[:2000]}
            
            Turn: {turn}/{max_turns} (remaining questions: {max_turns - turn})
            Clarification History:
            {history_text}
        """).strip()

    if 'project_map' in issue:
        pm = issue['project_map']
        map_lines = []
        for comp, info in pm.get('components', {}).items():
            map_lines.append(f"- '{comp}' → {info['team']} ({info['description']})")
        prompt += "\n\nProject Map:\n" + "\n".join(map_lines)
    
    return prompt


def parse_response(text: str) -> dict:
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "question", "priority": "medium"}


def run_task(env: GitHubTriagerClient, task_id: str, max_steps: int = 15):
    # MANDATORY LOG: [START]
    print(f"[START] task_id=\"{task_id}\"")

    try:
        observation = env.reset(task_id=task_id)
        total_reward = 0.0
        steps = 0

        for step in range(1, max_steps + 1):
            prompt = build_user_prompt(observation, task_id)

            try:
                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                action = parse_response(response.choices[0].message.content)
            except Exception:
                action = {"label": "question", "priority": "medium"}

            result = env.step(action)
            reward_data = result.get("reward", {})
            
            score = float(reward_data.get("score", reward_data.get("step_score", 0.0)))
            done = result.get("done", True)
            steps += 1

            # MANDATORY LOG: [STEP]
            print(f"[STEP] step={step}, score={score:.4f}, done={done}")
            
            total_reward += score

            if done:
                # Handle Task 3 trajectory scoring if applicable
                if reward_data.get("is_trajectory_final") and "trajectory_score" in reward_data:
                    final_traj = float(reward_data["trajectory_score"])
                    # In Task 3, we often consider the trajectory score as the final word
                    total_reward = final_traj 
                break

            observation = result.get("observation", observation)

        # MANDATORY LOG: [END]
        print(f"[END] total_reward={total_reward:.4f}")
        return total_reward
        
    except Exception as e:
        print(f"Error running task {task_id}: {e}")
        print(f"[END] total_reward=0.0000")
        return 0.0


def main():
    with GitHubTriagerClient(base_url="http://localhost:8000") as env:
        try:
            env.health()
        except Exception:
            return

        run_task(env, "label_classification")
        run_task(env, "full_triage")
        run_task(env, "batch_triage_with_context", max_steps=10)
        run_task(env, "clarification_triage", max_steps=5)


if __name__ == "__main__":
    main()
