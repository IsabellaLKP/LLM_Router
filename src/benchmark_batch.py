import os
import time
import json
import csv
import re
import requests
from router_cli import load_or_build, score

MODEL_PATH = "router_model.joblib"
OLLAMA_URL = "http://localhost:11434/api/generate"
BENCHMARK_LOG = "agent_execution_log.csv"

# Updated 3-Tier Mapping matching your newly pulled local models
MODELS = {
    0: "qwen2.5:0.5b",   # Ultra-fast sub-1B tier
    1: "llama3.2:3b",    # Balanced middle tier
    2: "llama3.1:8b"     # High-capacity reasoning tier
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
    
    # High Complexity & Structural Constraints (Target: Tier 2 or Escalation)
    "Generate a JSON object containing 5 movies. You must include 'title', 'year', 'director'.",
    "Analyze the main economic factors that led to the 2008 global financial crisis and critique the response.",
    "Calculate compound interest on $5,000 at 4.5% over 10 years compounded monthly using A = P(1 + r/n)^(nt)."
]


def query_ollama(model_name: str, prompt: str):
    """Sends request to Ollama with token generation caps to prevent CPU hangs."""
    start = time.time()
    payload = {
        "model": model_name, 
        "prompt": prompt, 
        "stream": False,
        "options": {
            "num_predict": 128,  # Restricts generation to ~128 tokens for speed
            "num_thread": 4      # Restricts thread usage to physical CPU cores
        }
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=180)  # 3-minute timeout guard
        res.raise_for_status()
        data = res.json()
        elapsed = time.time() - start
        
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1) / 1e9
        tps = eval_count / eval_duration if eval_duration > 0 else 0.0
        
        return data.get("response", ""), elapsed, eval_count, tps, None
    except Exception as e:
        return "", time.time() - start, 0, 0.0, str(e)


def verify_output_quality(prompt: str, response: str) -> tuple[bool, str]:
    """Validates output structure and basic completion requirements."""
    if not response or len(response.strip()) < 5:
        return False, "Response empty or incomplete."

    if "json" in prompt.lower():
        json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return False, "Malformed JSON structure."
        else:
            return False, "Requested JSON format missing."

    return True, "Passed validation."


def run_agent_execution(bundle, prompt: str, writer):
    """Executes single prompt with active agent self-correction and logging."""
    d_score, c5, initial_tier, feats = score(bundle, prompt)
    current_tier = initial_tier
    attempts = 0
    max_attempts = 2
    total_latency = 0
    escalated = False

    while attempts < max_attempts:
        attempts += 1
        model_name = MODELS[current_tier]
        print(f"   Attempt {attempts}: Running {model_name} (Tier {current_tier})...")

        response_text, latency, tokens, tps, err = query_ollama(model_name, prompt)
        total_latency += latency

        if err:
            print(f"   Connection/Timeout error: {err}")
            log_row(writer, prompt, d_score, initial_tier, current_tier, escalated, model_name, total_latency, tokens, "TIMEOUT_ERROR")
            return
        
        is_valid, reason = verify_output_quality(prompt, response_text)

        if is_valid:
            print(f"   Passed ({reason}) | Latency: {latency:.2f}s | Speed: {tps:.1f} t/s")
            log_row(writer, prompt, d_score, initial_tier, current_tier, escalated, model_name, total_latency, tokens, "SUCCESS")
            return
        else:
            print(f"   Failed inspection: {reason}")
            if current_tier < 2:
                current_tier += 1
                escalated = True
                print(f"   Escalating to: {MODELS[current_tier]}...")
            else:
                print("   Max tier reached. Returning unverified response.")
                log_row(writer, prompt, d_score, initial_tier, current_tier, escalated, model_name, total_latency, tokens, "FAILED_INSPECTION")
                return


def log_row(writer, prompt, d_score, init_tier, final_tier, escalated, model, latency, tokens, status):
    """Helper function to log row data directly to CSV."""
    writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        prompt,
        round(d_score, 4),
        init_tier,
        final_tier,
        escalated,
        model,
        round(latency, 3),
        tokens,
        status
    ])


def run_batch():
    bundle = load_or_build(MODEL_PATH, retrain=False, sample=None)
    headers = [
        "timestamp", "prompt", "difficulty_score", "initial_tier", 
        "final_tier", "escalated", "final_model", "total_latency_sec", 
        "tokens_generated", "status"
    ]
    
    file_exists = os.path.exists(BENCHMARK_LOG)
    
    print("\nStarting Active Agent Batch Benchmark...")
    print("=" * 60)

    with open(BENCHMARK_LOG, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)

        for i, prompt in enumerate(TEST_SUITE, 1):
            print(f"\n[{i}/{len(TEST_SUITE)}] Processing: '{prompt[:35]}...'")
            run_agent_execution(bundle, prompt, writer)

    print("\n" + "=" * 60)
    print(f"Batch execution complete. Results updated in {BENCHMARK_LOG}")


if __name__ == "__main__":
    run_batch()