import os
import time
import csv
import requests
from router_cli import load_or_build, score

MODEL_PATH = "router_model.joblib"
OLLAMA_URL = "http://localhost:11434/api/generate"
BASELINE_LOG = "baseline_execution_log.csv"

# 3-Tier Model Mapping (Update exact model tags from 'ollama list')
MODELS = {
    0: "qwen2.5:0.5b",   # Tier 0: Easy
    1: "llama3.2:3b",    # Tier 1: Medium
    2: "llama3.1:8b"     # Tier 2: Hard
}

TEST_SUITE = [
    # Low Complexity (Target: Tier 0)
    "What is the capital of France?",
    "Hi, how are you doing today?",
    "List three red fruits.",
    "Translate 'Good morning' into Spanish.",
    
    # Medium Complexity (Target: Tier 1)
    "Write a Python function using numpy to calculate a moving average with a window of 5.",
    "Write a step-by-step guide on how to change a flat tire.",
    "Summarize photosynthesizing plants using bullet points for key facts.",
    
    # High Complexity & Structural Constraints (Target: Tier 2)
    "Generate a JSON object containing 5 movies. You must include 'title', 'year', 'director'.",
    "Analyze the main economic factors that led to the 2008 global financial crisis and critique the response.",
    "Calculate compound interest on $5,000 at 4.5% over 10 years compounded monthly using A = P(1 + r/n)^(nt)."
]


def query_ollama(model_name: str, prompt: str):
    """Executes a single-pass inference call with speed options enabled."""
    start = time.time()
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 128,  # Token output cap for fair CPU comparison
            "num_thread": 4      # Optimized for physical CPU core counts
        }
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=180)
        res.raise_for_status()
        data = res.json()
        elapsed = time.time() - start
        
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1) / 1e9
        tps = eval_count / eval_duration if eval_duration > 0 else 0.0
        
        return data.get("response", ""), elapsed, eval_count, tps, "SUCCESS"
    except Exception as e:
        return "", time.time() - start, 0, 0.0, f"ERROR: {e}"


def run_baseline_benchmark():
    bundle = load_or_build(MODEL_PATH, retrain=False, sample=None)
    
    headers = [
        "timestamp", "prompt", "difficulty_score", "assigned_tier",
        "assigned_model", "latency_sec", "tokens_generated", "tokens_per_sec", "status"
    ]
    
    file_exists = os.path.exists(BASELINE_LOG)
    
    print("\nStarting Baseline Router Benchmark (Single-Pass)...")
    print("=" * 60)

    with open(BASELINE_LOG, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)

        for i, prompt in enumerate(TEST_SUITE, 1):
            # Classify difficulty score and map to tier
            d_score, c5, assigned_tier, _ = score(bundle, prompt)
            model_name = MODELS[assigned_tier]
            
            print(f"[{i}/{len(TEST_SUITE)}] Processing: '{prompt[:35]}...'")
            print(f"   Score: {d_score:+.3f} | Assigned Tier {assigned_tier} ({model_name})")

            response_text, latency, tokens, tps, status = query_ollama(model_name, prompt)
            
            print(f"   Done | Latency: {latency:.2f}s | Speed: {tps:.1f} t/s | Status: {status}")

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                prompt,
                round(d_score, 4),
                assigned_tier,
                model_name,
                round(latency, 3),
                tokens,
                round(tps, 2),
                status
            ])

    print("\n" + "=" * 60)
    print(f"Baseline benchmark complete. Output saved to {BASELINE_LOG}")


if __name__ == "__main__":
    run_baseline_benchmark()