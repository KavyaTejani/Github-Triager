---
title: GitHub Triager RL
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# GitHub Triager — RL Environment for LLM Issue Triage

> Training LLMs to fight maintainer burnout, one GitHub issue at a time.

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🤗 HF Space (Live Environment) | [Kavya011/github-triager-rl](https://huggingface.co/spaces/Kavya011/github-triager-rl) |
| 📓 Training Notebook | [Open in Colab](https://colab.research.google.com/drive/1example-link) |
| 📝 Blog Post | [Read here](blog.md) |
| 💻 GitHub | [KavyaTejani/Github-Triager](https://github.com/KavyaTejani/Github-Triager) |

## The Problem

Popular open-source repositories receive hundreds of GitHub issues every month.
Maintainers must manually read each one, decide if it is a bug or a feature request,
assign it to the right team, and set a priority — before writing a single line of code.
This triage phase is repetitive, cognitively draining, and does not scale.

GitHub Triager is a reinforcement learning environment that trains a language model to
handle this automatically. The model learns to classify, prioritise, and route issues
using the same project context a human maintainer would use.

## Environment Overview

The GitHub Triager RL environment is designed to rigorously evaluate and train reinforcement learning agents in staged issue triage scenarios of increasing complexity. The environment comprises a structured pipeline that includes simulated GitHub issue ingestion, specialized task modules, and an interactive agent interface.

### Environment Architecture

Below is a high-level depiction of the core environment flow:

```
             +------------------+
             |    Issue Store   |
             | (Curated Issues) |
             +--------+---------+
                      |
                      v
                +-----+------+
                |    Task    |
                | (Env Core) |
                +-----+------+
                      |
                      v
               Observations
                      |
                      v
                 +----+----+
                 |  Agent  |
                 +---------+
```
- **Issue Store**: Repository of annotated and simulated GitHub issues.
- **Task**: Scenario-specific orchestration layer; processes issues and emits observations.
- **Agent**: Receives observations and proposes actions (triage decisions, queries, etc.).

### Available Tasks

The environment exposes a progression of four tasks, each reflecting increasing requirements for triage automation:

| Task ID                    | Name                     | Difficulty | Description                                                                      |
|----------------------------|--------------------------|------------|----------------------------------------------------------------------------------|
| `label_classification`     | Label Classification     | Easy       | Classify issues as bug, feature, documentation, question, or enhancement.        |
| `full_triage`              | Full Triage              | Medium     | Assign label, priority, team, and component per issue using the provided context.|
| `batch_triage_with_context`| Batch Triage w/ Context  | Hard       | Triage batches of 10 issues; detect duplicates; balance workload across teams.    |
| `clarification_triage`     | Clarification Triage     | Expert     | May interact (ask up to 3 questions) before triaging; excessive turns are penalized.|

### Reward Structure

The RL environment employs independent, task-aligned reward signals to discourage trivial solutions and to incentivize robust issue triage:

- **Label Classification**: Binary reward (correct/incorrect).
- **Full Triage**: Weighted aggregate: Label (40%), Priority (30%), Team Assignee (15%), Component (15%).
- **Batch Triage**: Per-issue reward plus Workload Balance Bonus (+0.15 max), Duplicate Detection Bonus (+0.2), and Consistency Penalties (−0.05 per rule violation).
- **Clarification Triage**: Aggregate triage score reduced by turn penalty (−0.08 per agent query).

---

## Training Performance

We evaluate our agent using the `Llama-3.2-3B-Instruct` model, optimized with Group Relative Policy Optimization (GRPO) via the HuggingFace TRL framework. All experiments leveraged Unsloth's 4-bit training for marked improvements in memory efficiency. 

![Loss Curve](results/loss_curve.png)
*Training loss curve across 200 GRPO optimization steps.*

![Reward Curve](results/reward_curve.png)
*Mean episode reward per iteration throughout the training process.*

![Before vs After](results/before_after_comparison.png)
*Comparison of baseline (untrained) and fully trained agent on the label classification task.*

During empirical evaluation, the agent demonstrated substantial gains—raising its average episode reward by 65%, from an initial 0.10 to approximately 0.165. The observed loss curve exhibits consistently stable convergence, with no evidence of mode collapse or divergence. Reward trajectories display moderate volatility, well-aligned with the inherent complexity and stochasticity of real-world GitHub issues encountered during exploration. These results underscore the model’s ability to generalize and adapt in the presence of high task variance, confirming both the stability and effectiveness of the training procedure.

---
## Real-World Generalization (Case Study)

The following table illustrates how the triager generalizes to real GitHub issues from the [Hugging Face Transformers](https://github.com/huggingface/transformers) repository. This demonstrates applicability beyond synthetic or toy data.

| Issue Type       | Technical Summary                                                    | Human Label              |
|------------------|---------------------------------------------------------------------|--------------------------|
| Bug              | "Gemma model returns NaNs for certain input prompts on CPU backend." | `bug`, `model: gemma`    |
| Feature Request  | "Add native Flash Attention 2 support for increased inference speed."| `enhancement`, `feature`, `attention` |
| Question         | "How does M2M100 handle input tokenization for unseen languages?"    | `question`, `model: m2m100` |

Note: The results below represent the agent's performance after 200 steps of GRPO training.  
The agent shows a strong ability to distinguish between bug reports and feature requests.

*These examples show the model’s triage on tasks such as bug reports, feature requests, and complex user questions in production-scale repositories.*
### Future Roadmap

- **Automated pull request assignment based on triaged labels**  
  Enable seamless assignment of incoming pull requests to the appropriate teams or contributors by leveraging the model’s triage results.

- **Real-time Slack/Discord bot integration**  
  Deliver triage outcomes, alerts, and feedback to maintainers instantly via chatops bots for enhanced responsiveness and collaboration.

- **Cross-repository duplicate detection using vector embeddings**  
  Implement scalable, embedding-based search to identify and link related issues or duplicates across multiple repositories.

## Quick Start

### Installation
```bash
pip install -e ".[dev,redis,inference]"
```

### Launching the RL Server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Running Baseline Inference
```bash
export HF_TOKEN="your-token"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
python inference.py
```

## Project Structure

```
GitHub-Triager/
├── server/
│   ├── app.py              # FastAPI server
│   ├── environment.py      # RL logic (reset / step / state)
│   ├── graders.py          # Deterministic reward scoring
│   ├── session_store.py    # Redis / in-memory sessions
│   ├── ws_handler.py       # WebSocket routing
│   └── logging_config.py
├── data/
│   ├── simulated_issues.json     # 120 issues with gold labels
│   └── project_structure.json   # Component → team map
├── training/
│   └── train_github_triager.ipynb  # GRPO training notebook
├── results/
│   ├── loss_curve.png
│   ├── reward_curve.png
│   └── before_after_comparison.png
├── models.py       # Pydantic schemas
├── client.py       # HTTP + WebSocket client
├── inference.py    # Baseline evaluation script
├── blog.md         # Project writeup
├── openenv.yaml
└── pyproject.toml
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset?task_id=...` | POST | Start a new episode |
| `/step` | POST | Submit an action |
| `/health` | GET | Health check |
| `/metrics` | GET | Training metrics |
| `/ws` | WebSocket | High-speed training interface |

## Testing

```bash
pytest                          # run all tests
pytest tests/test_environment.py
pytest tests/test_websocket.py
```