"""
score_similarity.py

Checks whether the small models actually said the same thing as the
reference model. Uses Ollama's own embedding endpoint so there is no new
dependency (pull once: `ollama pull nomic-embed-text`).

Reports three numbers per (model, prompt), because any one of them alone
is misleading:
  cosine      semantic closeness to the reference answer
  jaccard     word overlap; if cosine is high but jaccard is also high the
              model may just be copying, not agreeing
  len_ratio   output length vs reference; catches truncation and padding

Output: similarity.csv

Deps: pip install requests
"""

import csv
import json
import math
import re
from collections import defaultdict

import requests

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
REFERENCE_MODEL = "llama3.1:8b"   # must match run_energy_benchmark.py
OUTPUTS = "outputs.jsonl"
OUT = "similarity.csv"
MAX_CHARS = 6000                  # embedding context guard

_cache = {}


def embed(text):
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return None
    if text in _cache:
        return _cache[text]
    r = requests.post(
        f"{OLLAMA}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    v = r.json()["embedding"]
    _cache[text] = v
    return v


def cosine(a, b):
    if a is None or b is None:
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else float("nan")


def words(text):
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def jaccard(a, b):
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return float("nan")
    return len(wa & wb) / len(wa | wb)


def main():
    # keep run 0 only; repeated runs are for energy noise, not text variation
    # (temperature is 0, so outputs should be near-identical across runs)
    texts = {}
    with open(OUTPUTS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["run"] != 0:
                continue
            texts[(row["model"], row["prompt_id"])] = row["output"]

    models = sorted({m for m, _ in texts})
    prompt_ids = sorted({p for _, p in texts})
    print(f"{len(models)} models x {len(prompt_ids)} prompts")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt_id", "cosine", "jaccard", "len_ratio"])
        for pid in prompt_ids:
            ref = texts.get((REFERENCE_MODEL, pid))
            if ref is None:
                continue
            ref_vec = embed(ref)
            for m in models:
                cand = texts.get((m, pid))
                if cand is None:
                    continue
                if m == REFERENCE_MODEL:
                    w.writerow([m, pid, 1.0, 1.0, 1.0])
                    continue
                lr = len(cand) / len(ref) if ref else float("nan")
                w.writerow([
                    m, pid,
                    round(cosine(embed(cand), ref_vec), 4),
                    round(jaccard(cand, ref), 4),
                    round(lr, 4),
                ])
            print(f"  {pid} done")

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
