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

# 3-Tier Model Hierarchy
TIER_MODELS = {
    0: "qwen2.5:0.5b",   # Tier 0 (Easy): Sub-1B ultra-fast response
    1: "llama3.2:3b",    # Tier 1 (Medium): High-efficiency balanced model
    2: "llama3.1:8b"     # Tier 2 (Hard): High capacity reasoning tier
}


def query_ollama(model_name: str, prompt: str):
    """Sends request to local Ollama instance and returns response + metadata."""
    start = time.time()
    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"num_predict": 256}}
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        elapsed = time.time() - start
        
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 1) / 1e9
        tps = eval_count / eval_duration if eval_duration > 0 else 0.0
        
        return data.get("response", ""), elapsed, eval_count, tps, None
    except Exception as e:
        return "", time.time() - start, 0, 0.0, str(e)


# ---------------------------------------------------------------------------
# AGENT TOOL EXECUTION & VERIFICATION
# ---------------------------------------------------------------------------

def execute_python_tool(code_snippet: str) -> str:
    """Safely evaluates basic math or Python expressions."""
    try:
        allowed_globals = {"__builtins__": None, "abs": abs, "pow": pow, "round": round, "sum": sum}
        result = eval(code_snippet, allowed_globals)
        return f"[Tool Output]: {result}"
    except Exception as e:
        return f"[Tool Error]: {e}"


def verify_output_quality(prompt: str, response: str) -> tuple[bool, str]:
    """
    Validates output structure and basic completion requirements.
    Returns (is_valid, failure_reason).
    """
    if not response or len(response.strip()) < 5:
        return False, "Response was empty or too short."

    # Validate JSON requests if explicitly mentioned in prompt
    if "json" in prompt.lower():
        json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return False, "Response contained malformed JSON."
        else:
            return False, "Requested JSON format was not provided."

    # Validate output length for complex analytical prompts
    analytical_keywords = ["analyze", "prove", "critique", "evaluate"]
    if any(k in prompt.lower() for k in analytical_keywords) and len(response.split()) < 30:
        return False, "Response length insufficient for analytical request."

    return True, "Passed validation."


# ---------------------------------------------------------------------------
# ACTIVE ROUTING & ESCALATION ENGINE
# ---------------------------------------------------------------------------

def run_agent(bundle, prompt: str):
    """Executes prompt with dynamic routing, quality checks, and fallback tier escalation."""
    d_score, c5, initial_tier, feats = score(bundle, prompt)
    current_tier = initial_tier
    attempts = 0
    max_attempts = 2
    
    print("-" * 50)
    print(f"Prompt: {prompt}")
    print(f"Difficulty Score: {d_score:+.3f} | Initial Class: {current_tier} ({TIER_MODELS[current_tier]})")
    print("-" * 50)

    total_latency = 0
    escalated = False

    while attempts < max_attempts:
        attempts += 1
        model_name = TIER_MODELS[current_tier]
        print(f"\nAttempt {attempts}: Executing with {model_name}...")

        response_text, latency, tokens, tps, err = query_ollama(model_name, prompt)
        total_latency += latency

        if err:
            print("Execution timed out (exceeded 60s CPU limit).")
            log_agent_run(prompt, d_score, initial_tier, current_tier, escalated, model_name, 60.0, 0, "CPU_TIMEOUT")
            return  # Stop here rather than attempting an even larger model
        else:
            is_valid, reason = verify_output_quality(prompt, response_text)

        if is_valid:
            print(f"Verification status: Passed ({reason})")
            print(f"Latency: {latency:.2f}s | Throughput: {tps:.1f} tokens/sec")
            print("\nResponse:")
            print(response_text)
            print("-" * 50 + "\n")
            
            log_agent_run(prompt, d_score, initial_tier, current_tier, escalated, model_name, total_latency, tokens, "SUCCESS")
            return
        else:
            print(f"Verification status: Failed ({reason})")
            if current_tier < 2:
                current_tier += 1
                escalated = True
                print(f"Escalating request to next model tier: {TIER_MODELS[current_tier]}...")
            else:
                print("Max tier reached. Returning current output.")
                print("\nResponse (Unverified):")
                print(response_text)
                print("-" * 50 + "\n")
                log_agent_run(prompt, d_score, initial_tier, current_tier, escalated, model_name, total_latency, tokens, "FAILED_INSPECTION")
                return


def log_agent_run(prompt, d_score, init_tier, final_tier, escalated, model, latency, tokens, status):
    """Appends execution metrics to the CSV log."""
    file_exists = os.path.exists(BENCHMARK_LOG)
    headers = [
        "timestamp", "prompt", "difficulty_score", "initial_tier", 
        "final_tier", "escalated", "final_model", "total_latency_sec", 
        "tokens_generated", "status"
    ]
    with open(BENCHMARK_LOG, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), prompt, round(d_score, 4),
            init_tier, final_tier, escalated, model, round(latency, 3), tokens, status
        ])


def main():
    bundle = load_or_build(MODEL_PATH, retrain=False, sample=None)
    print("Router Agent initialized. Type your prompt below, or use ':q' to exit.\n")
    
    while True:
        try:
            prompt = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt in (":quit", ":q", ":exit"):
            print("Exiting.")
            break

        run_agent(bundle, prompt)


if __name__ == "__main__":
    main()