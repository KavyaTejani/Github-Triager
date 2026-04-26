# GitHub Triager: Training an LLM to Triage Issues with Reinforcement Learning

## The Problem
Open-source maintainers spend hours manually triaging incoming GitHub issues — deciding
if something is a bug or a feature request, figuring out which team should own it, and
assigning priority. As repositories scale, this becomes a major source of burnout.
GitHub Triager is an RL environment designed to train a language model to handle this
automatically.

## The Environment
GitHub Triager is an OpenEnv-compliant environment with **four tasks of increasing difficulty**:

| Task | Difficulty | What the agent must do |
|------|------------|------------------------|
| Label Classification | Easy | Classify into bug / feature / docs / question / enhancement |
| Full Triage | Medium | Assign label + priority + team + component using a project map |
| Batch Triage | Hard | Triage 10 issues, detect duplicates, balance team workload |
| Clarification Triage | Expert | Ask targeted questions before triaging; each extra question costs reward |

## Reward Design
We avoided a single binary reward signal to prevent reward hacking:
- **Task 2** splits reward: Label (40%), Priority (30%), Assignee (15%), Component (15%).
- **Task 3** adds workload balance bonuses (+0.15 max) and consistency penalties (-0.05 per violation).
- **Task 4** penalises each clarification turn by -0.08, teaching the model to ask only when necessary.

## Training
I trained `Llama-3.2-3B-Instruct` using **GRPO** (via HuggingFace TRL) with **Unsloth**
for 4-bit efficiency. GRPO (Group Relative Policy Optimization) was chosen over standard PPO because it lets the model directly compare several triage responses for the same issue. This approach teaches the agent to distinguish not just between correct and incorrect actions, but between mediocre and excellent triage, all without requiring enormous amounts of training data. The model rolls out episodes against the live OpenEnv environment
and receives structured reward feedback after each step.

![Loss Curve](results/loss_curve.png)
*Training loss over 200 GRPO steps.*

![Reward Curve](results/reward_curve.png)
*Average episode reward during training. The model improves steadily.*

## Results
![Before vs After](results/before_after_comparison.png)
*The trained model significantly outperforms the untrained baseline on label classification.*

## Links
- 🤗 **Live Environment:** https://huggingface.co/spaces/Kavya011/github-triager-rl
- 📓 **Training Notebook:** [Open in Colab](https://colab.research.google.com/drive/1example-link)
- 💻 **GitHub:** https://github.com/KavyaTejani/Github-Triager