"""
measure_gen.py -- per-tier LLM generation energy, measured with NVML.

Runs each model over the same set of prompts, integrates GPU power via NVML over
the whole run, and divides by the number of generations. This gives the per-tier
generation energy (J/prompt) that routing_experiment.py needs in TIERS.

Measure all four the SAME way (same prompts, same token cap) so the ratios between
tiers are valid -- the routing result depends on those ratios.

Requires: ollama running, the models pulled, pynvml, and the python client
(pip install ollama).
  ollama pull qwen2.5:0.5b llama3.2:1b llama3.2:3b llama3.1:8b   # if not pulled

Run:  python measure_gen.py --n 100
"""

import argparse
import threading
import time
from time import perf_counter

import pynvml

MODELS = ["qwen2.5:0.5b", "llama3.2:1b", "llama3.2:3b", "llama3.1:8b"]

# A fixed, representative prompt set. Same for every model so comparisons are fair.
PROMPTS = [
    "Explain how a bicycle stays upright when moving.",
    "Write a short paragraph about the ocean.",
    "What are three causes of the French Revolution?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Give me a recipe for a simple tomato soup.",
    "Explain the difference between weather and climate.",
    "Write a function in Python that reverses a string.",
    "What is compound interest and how does it work?",
    "Describe the water cycle.",
    "List some tips for improving sleep quality.",
]


class GpuSampler:
    def __init__(self, hz=50):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.dt = 1.0 / hz; self._stop = False; self.j = 0.0

    def _loop(self):
        last = perf_counter()
        while not self._stop:
            time.sleep(self.dt)
            now = perf_counter(); span = now - last; last = now
            self.j += pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0 * span

    def __enter__(self):
        self._stop = False; self.j = 0.0
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True; self._t.join()


def measure_model(model, prompts, num_predict):
    import ollama
    # warmup (load weights into VRAM; don't count this)
    ollama.generate(model=model, prompt="hello",
                    options={"num_predict": 16})
    with GpuSampler() as s:
        t0 = perf_counter()
        for p in prompts:
            ollama.generate(model=model, prompt=p,
                            options={"num_predict": num_predict})
        elapsed = perf_counter() - t0
    return s.j / len(prompts), elapsed / len(prompts)


def measure_idle_gpu(seconds):
    with GpuSampler() as s:
        time.sleep(seconds)
    return s.j / seconds  # watts


def run(args):
    prompts = (PROMPTS * ((args.n // len(PROMPTS)) + 1))[:args.n]
    print(f"{args.n} generations/model, num_predict={args.num_predict}\n")

    idle_w = measure_idle_gpu(10)
    print(f"idle GPU power: {idle_w:.1f} W\n")
    print(f"{'model':<16}{'gross J/gen':>13}{'net J/gen':>12}{'sec/gen':>10}")
    results = {}
    for m in MODELS:
        try:
            gross, sec = measure_model(m, prompts, args.num_predict)
        except Exception as e:
            print(f"{m:<16}  ERROR: {e}")
            continue
        net = max(gross - idle_w * sec, 0.0)   # subtract idle draw over gen time
        results[m] = net
        print(f"{m:<16}{gross:>13.2f}{net:>12.2f}{sec:>10.3f}")

    print("\nPaste these NET J/gen into routing_experiment.py TIERS:")
    for m in MODELS:
        if m in results:
            print(f'    ("{m}", {results[m]:.2f}),')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="generations per model")
    ap.add_argument("--num-predict", type=int, default=256,
                    help="max tokens per generation (keep fixed across models)")
    run(ap.parse_args())
