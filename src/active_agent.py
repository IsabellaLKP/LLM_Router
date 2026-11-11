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

TIER_MODELS = {
    0: "qwen2.5:0.5b",   # Tier 0 (Easy): Sub-1B parameter model (~72.9 J per gen)
    1: "llama3.2:3b",    # Tier 1 (Medium): Balanced reasoning (~221.8 J per gen)
    2: "llama3.1:8b"     # Tier 2 (Hard): High-capacity reasoning (~486.6 J per gen)
}

def query_ollama(model_name: str, prompt: str):
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

def execute_python_tool(code_snippet: str) -> str:
    """Safely evaluates numeric calculations or algorithmic queries."""
    try:
        allowed_globals = {"__builtins__": None, "abs": abs, "pow": pow, "round": round, "sum": sum, "max": max, "min": min}
        result = eval(code_snippet, allowed_globals)
        return f"[Tool Output]: {result}"
    except Exception as e:
        return f"[Tool Error]: {e}"

def verify_output_quality(prompt: str, response: str) -> tuple[bool, str]:
    if not response or len(response.strip()) < 5:
        return False, "Output empty or truncated."

    # JSON Structure Audit
    if "json" in prompt.lower():
        json_match = re.search(r"\{.*\}|\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return False, "Response contained malformed JSON."
        else:
            return False, "Requested JSON format missing."

    # Analytical Length Check
    analytical_keywords = ["analyze", "prove", "critique", "evaluate", "explain"]
    if any(k in prompt.lower() for k in analytical_keywords) and len(response.split()) < 25:
        return False, "Response length insufficient for analytical request."

    return True, "Passed validation."

def run_agent(bundle, prompt: str):
    d_score, c5, initial_tier, feats = score(bundle, prompt)
    current_tier = initial_tier
    attempts = 0
    max_attempts = 2
    total_latency = 0
    escalated = False

    print(f"\n--- Processing Prompt: '{prompt[:40]}...' ---")
    print(f"Router Score: {d_score:+.3f} | Initial Tier: Tier {initial_tier} ({TIER_MODELS[initial_tier]})")

    while attempts < max_attempts:
        attempts += 1
        model_name = TIER_MODELS[current_tier]
        response_text, latency, tokens, tps, err = query_ollama(model_name, prompt)
        total_latency += latency

        if err:
            log_agent_run(prompt, d_score, initial_tier, current_tier, escalated, model_name, 60.0, 0, "TIMEOUT")
            print("Execution timed out.")
            return

        is_valid, reason = verify_output_quality(prompt, response_text)

        if is_valid:
            print(f"Verification: Passed ({reason}) | Model: {model_name} | Latency: {latency:.2f}s")
            log_agent_run(prompt, d_score, initial_tier, current_tier, escalated, model_name, total_latency, tokens, "SUCCESS")
            return response_text
        else:
            print(f"Verification: Failed ({reason})")
            if current_tier < 2:
                current_tier += 1
                escalated = True
                print(f"Escalating to Tier {current_tier} ({TIER_MODELS[current_tier]})...")
            else:
                log_agent_run(prompt, d_score, initial_tier, current_tier, escalated, model_name, total_latency, tokens, "FAILED_INSPECTION")
                return response_text

def log_agent_run(prompt, d_score, init_tier, final_tier, escalated, model, latency, tokens, status):
    file_exists = os.path.exists(BENCHMARK_LOG)
    headers = ["timestamp", "prompt", "difficulty_score", "initial_tier", "final_tier", "escalated", "final_model", "total_latency_sec", "tokens_generated", "status"]
    with open(BENCHMARK_LOG, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), prompt, round(d_score, 4), init_tier, final_tier, escalated, model, round(latency, 3), tokens, status])