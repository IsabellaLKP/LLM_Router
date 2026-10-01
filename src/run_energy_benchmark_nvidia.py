"""
run_energy_benchmark_nvidia.py

Same experiment as run_energy_benchmark.py, but for the RTX 2070. CodeCarbon
reads the GPU natively through NVML, so there is no LibreHardwareMonitor hook
and no separate power sampler. Keep the AMD version around for the 6700 XT.

Also logs how Ollama split each model between GPU and CPU. If a model spills
layers to CPU, its energy numbers are not comparable to the others and the run
should be redone with a smaller model or shorter prompts.

Outputs:
  energy.csv    one row per (model, prompt, run)
  outputs.jsonl one row per (model, prompt, run) with the generated text

Deps: pip install codecarbon nvidia-ml-py requests
"""

import csv
import json
import time

import requests
from codecarbon import EmissionsTracker

# --- config ---------------------------------------------------------------
MODELS = [
    "qwen2.5:0.5b",
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen2.5:7b",
    "llama3.1:8b",
]
REFERENCE_MODEL = "llama3.1:8b"
PROMPTS = "prompts.jsonl"
RUNS = 3
NUM_PREDICT = 512
OLLAMA = "http://localhost:11434"
SEED = 42
COOLDOWN = 3.0


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate(model, prompt):
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": SEED, "num_predict": NUM_PREDICT},
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def gpu_placement(model):
    """Report whether Ollama put the whole model on the GPU. 100% is what you want."""
    try:
        r = requests.get(f"{OLLAMA}/api/ps", timeout=30)
        for m in r.json().get("models", []):
            if m.get("name", "").startswith(model.split(":")[0]):
                total = m.get("size", 0)
                vram = m.get("size_vram", 0)
                pct = 100 * vram / total if total else 0
                return round(pct, 1)
    except Exception:
        pass
    return None


def main():
    prompts = load_prompts(PROMPTS)
    print(f"{len(prompts)} prompts x {len(MODELS)} models x {RUNS} runs")

    ecsv = open("energy.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        ecsv,
        fieldnames=[
            "model", "prompt_id", "bin", "difficulty", "run",
            "cpu_ram_joules", "gpu_joules", "total_joules",
            "wall_seconds", "prompt_tokens", "output_tokens", "pct_on_gpu",
        ],
    )
    writer.writeheader()
    ojsonl = open("outputs.jsonl", "w", encoding="utf-8")

    for model in MODELS:
        print(f"\n=== {model} ===")
        generate(model, "hi")          # warm up, loads weights into VRAM
        time.sleep(COOLDOWN)
        pct = gpu_placement(model)
        print(f"  {pct}% of model resident on GPU"
              if pct is not None else "  GPU placement unknown")
        if pct is not None and pct < 99:
            print("  WARNING: model is partly on CPU, energy is not comparable")

        for run in range(RUNS):
            for p in prompts:
                tracker = EmissionsTracker(
                    measure_power_secs=1,
                    save_to_file=False,
                    log_level="error",
                )
                t0 = time.time()
                tracker.start()
                resp = generate(model, p["prompt"])
                tracker.stop()
                wall = time.time() - t0

                d = tracker.final_emissions_data
                cpu_ram_j = (d.cpu_energy + d.ram_energy) * 3.6e6   # kWh -> J
                gpu_j = d.gpu_energy * 3.6e6
                total = cpu_ram_j + gpu_j

                writer.writerow({
                    "model": model,
                    "prompt_id": p["id"],
                    "bin": p["bin"],
                    "difficulty": p["difficulty"],
                    "run": run,
                    "cpu_ram_joules": round(cpu_ram_j, 3),
                    "gpu_joules": round(gpu_j, 3),
                    "total_joules": round(total, 3),
                    "wall_seconds": round(wall, 3),
                    "prompt_tokens": resp.get("prompt_eval_count", 0),
                    "output_tokens": resp.get("eval_count", 0),
                    "pct_on_gpu": pct,
                })
                ecsv.flush()

                ojsonl.write(json.dumps({
                    "model": model,
                    "prompt_id": p["id"],
                    "run": run,
                    "output": resp.get("response", ""),
                }) + "\n")
                ojsonl.flush()

                time.sleep(COOLDOWN)
            print(f"  run {run + 1}/{RUNS} done")

    ecsv.close()
    ojsonl.close()
    print("\nwrote energy.csv and outputs.jsonl")


if __name__ == "__main__":
    main()
