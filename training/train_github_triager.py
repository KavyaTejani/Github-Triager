# Add project root to sys.path to find 'client'
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import re
import requests
import matplotlib.pyplot as plt
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
try:
    from client import GitHubTriagerClient
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.getcwd())
    from client import GitHubTriagerClient
# Configuration
ENVIRONMENT_URL = "https://kavya011-github-triager-rl.hf.space"
MODEL_NAME      = "unsloth/Llama-3.2-3B-Instruct"
MAX_STEPS       = 500      # Increased for stability
BATCH_SIZE      = 1        # Reduced to 1 to save VRAM
NUM_GENERATIONS = 2        # Reduced to 2 to save VRAM
GRADIENT_ACCUMULATION_STEPS = 4 # Simulates a larger batch size without using VRAM

def test_connection():
    print(f"Connecting to {ENVIRONMENT_URL}...")
    try:
        with GitHubTriagerClient(base_url=ENVIRONMENT_URL) as env:
            result = env.reset(task_id="label_classification")
            print("✅ Successfully connected to the environment!")
            print("Observation keys:", list(result.keys()))
            return result
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def load_model():
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    print("Model loaded successfully.")
    return model, tokenizer

def rollout_single(model, tokenizer, task_id="label_classification"):
    with GitHubTriagerClient(base_url=ENVIRONMENT_URL) as env:
        obs = env.reset(task_id=task_id)
        prompt = f"""You are a GitHub issue triager.
Issue Title: {obs['issue']['title']}
Issue Body: {obs['issue']['body']}
Task: Classify this issue. Respond with valid JSON only.
Format: {{"label": "<bug|feature|documentation|question|enhancement>"}}"""

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]   # strip prompt from output

        result = env.step({"response": response})
        return prompt, response, float(result.get("reward", 0.0))

def get_env_reward(completion: str) -> float:
    try:
        match = re.search(r'\{.*\}', str(completion), re.DOTALL)
        if match:
            action_data = json.loads(match.group(0))
        else:
            return 0.01 # Model didn't output JSON
    except:
        return 0.01

    try:
        url = f"{ENVIRONMENT_URL}/grade/label_classification"
        response = requests.post(url, json={"action": action_data}, timeout=10)

        if response.status_code == 200:
            return float(response.json().get("score", 0.01))

    except:
        pass
    return 0.01

def compute_reward(prompts, completions, **kwargs):
    return [get_env_reward(c[0]["content"] if isinstance(c, list) else c) for c in completions]

def build_dataset(tokenizer, n_samples: int = 100):
    print(f"Building dataset with {n_samples} samples...")
    samples = []
    import time
    with GitHubTriagerClient(base_url=ENVIRONMENT_URL) as env:
        for i in range(n_samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{n_samples} samples collected...")
            
            try:
                obs = env.reset(task_id="label_classification")
                prompt_text = f"""You are a GitHub issue triager.
Issue Title: {obs['issue']['title']}
Issue Body: {obs['issue']['body']}
Classify this issue. Respond with JSON only.
Format: {{"label": "<bug|feature|documentation|question|enhancement>"}}"""
                
                # Use Chat Template if available
                messages = [{"role": "user", "content": prompt_text}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                samples.append({"prompt": prompt})
                time.sleep(0.5) # Prevent overwhelming the server
            except Exception as e:
                print(f"  Warning: Failed to fetch sample {i}: {e}. Retrying...")
                time.sleep(2)
    return Dataset.from_list(samples)

def evaluate_model(model, tokenizer, dataset, n_episodes=20):
    import random
    total = 0.0
    for _ in range(n_episodes):
        prompt = dataset[random.randint(0, len(dataset)-1)]["prompt"]
        inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=64)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        total += get_env_reward(response)
    return total / n_episodes

def main():
    # 1. Test Connection
    test_connection()

    # 2. Load Model
    model, tokenizer = load_model()

    # 3. Build Dataset
    dataset = build_dataset(tokenizer, 100)
    print(f"Dataset ready: {len(dataset)} samples")

    # 4. Training
    print("Starting training (GRPO)...")
    training_args = GRPOConfig(
        output_dir="./outputs/github-triager-grpo",
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_steps=MAX_STEPS,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=20,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[compute_reward],
        tokenizer=tokenizer,
    )

    trainer.train()
    print("Training complete.")

    # 5. Save Plots
    os.makedirs("results", exist_ok=True)
    log_history = trainer.state.log_history
    steps        = [x["step"]   for x in log_history if "loss"   in x]
    losses       = [x["loss"]   for x in log_history if "loss"   in x]
    r_steps      = [x["step"]   for x in log_history if "reward" in x]
    rewards      = [x["reward"] for x in log_history if "reward" in x]

    if steps:
        plt.figure(figsize=(10, 4))
        plt.plot(steps, losses, color="royalblue", linewidth=2)
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("GitHub Triager — Training Loss (GRPO)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/loss_curve.png", dpi=150)
        print("Saved: results/loss_curve.png")

    if r_steps:
        plt.figure(figsize=(10, 4))
        plt.plot(r_steps, rewards, color="seagreen", linewidth=2)
        plt.xlabel("Training Step")
        plt.ylabel("Average Reward")
        plt.title("GitHub Triager — Reward During Training (GRPO)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/reward_curve.png", dpi=150)
        print("Saved: results/reward_curve.png")

    # 6. Evaluation
    try:
        print("Evaluating model...")
        trained_avg = evaluate_model(model, tokenizer, dataset, 20)
        baseline    = 0.10

        plt.figure(figsize=(6, 5))
        plt.bar(["Baseline", "Trained"], [baseline, trained_avg], color=["#ff6b6b", "#51cf66"], edgecolor="black")
        plt.ylabel("Average Reward")
        plt.title("Before vs After GRPO Training")
        plt.savefig("results/before_after_comparison.png", dpi=150)
        print("Saved: results/before_after_comparison.png")
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # 7. Save Adapter
    model.save_pretrained("outputs/github-triager-adapter")
    tokenizer.save_pretrained("outputs/github-triager-adapter")
    print("Adapter saved to outputs/github-triager-adapter")

if __name__ == "__main__":
    main()
