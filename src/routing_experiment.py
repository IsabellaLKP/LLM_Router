"""
routing_experiment.py -- does the router actually matter?

Ties accuracy to energy: each routing policy picks a model tier per prompt; we
tally the generation energy it spends and the routing quality it achieves, then
compare. This is the capstone test -- it decides whether cheap routing beats the
naive baselines, and whether the fine-tuned router's accuracy is worth its cost.

Ground truth difficulty = the fine-tuned model's score. That makes the fine-tuned
router the quality CEILING by construction; its value in the table is showing
what perfect routing costs to *run* (a GPU pass per prompt) vs what the CPU
router gives up to run for ~nothing.

Policies compared (as a quality-vs-energy frontier, sweeping how aggressively
each router pushes prompts toward cheap tiers):
  always-small / always-large  -- no routing (energy floor / quality ceiling)
  random                       -- sanity floor; a real router must beat this
  CPU router (0.650)           -- the deployable, near-free system
  Fine-tuned router (= oracle) -- perfect routing, GPU cost per prompt

Assumptions (state these in the paper):
  - difficulty rank maps monotonically to required model size (quantile bands)
  - per-tier generation energy from measured Ollama benchmarks (measure_gen.py)
  - routing by z-scored rank, so score scale/compression doesn't bias tiers

Needs router_model.joblib (built by router_cli.py).

Run:  python routing_experiment.py [--sample N]
Writes routing_frontier.csv (plot served vs total_J per policy).
"""

import argparse
import os
import re

import numpy as np
from scipy import sparse
import joblib

# ======================= CONFIG -- MEASURED NUMBERS ==========================
# (name, net generation energy in JOULES/prompt), smallest -> largest.
# From measure_gen.py on the RTX 2070 (100 generations/model, num_predict=256).
# qwen2.5:7b dropped (dominated by llama3.1:8b).
TIERS = [
    ("qwen2.5:0.5b",  72.90),
    ("llama3.2:1b",  128.03),
    ("llama3.2:3b",  221.77),
    ("llama3.1:8b",  486.60),
]
ROUTER_COST_J = {
    "cpu":       0.0088,   # measured: ~8.8 mJ/prompt (energy_benchmark.py)
    "finetuned": 0.322,    # measured: bge-small forward pass (energy_benchmark.py --embed, bge row)
}
# ============================================================================

DATASET = "agentlans/prompt-difficulty"
MODEL_PATH = "router_model.joblib"
FT_MODEL = "agentlans/bge-small-en-v1.5-prompt-difficulty"
N = len(TIERS)
TIER_E = np.array([e for _, e in TIERS])


# --- CPU router features (needed to score with the saved bundle) -----------
WH = {"what", "why", "how", "when", "where", "who", "whom", "whose", "which"}
BLOOM = {1: {"define", "list", "name", "state", "recall", "identify", "label"},
         2: {"describe", "explain", "summarize", "classify", "discuss", "interpret"},
         3: {"apply", "calculate", "solve", "compute", "use", "implement", "simulate"},
         4: {"analyze", "compare", "contrast", "differentiate", "examine", "categorize"},
         5: {"evaluate", "assess", "justify", "critique", "argue", "recommend", "judge"},
         6: {"create", "design", "develop", "compose", "write", "construct",
             "formulate", "devise", "plan", "generate", "build", "prove", "derive", "optimize"}}
LEVEL = {v: l for l, vs in BLOOM.items() for v in vs}
REASON = set().union(*(BLOOM[l] for l in (4, 5, 6)))
CONSTRAINT = {"must", "should", "only", "without", "exactly", "ensure", "avoid",
              "json", "table", "bullet", "paragraph", "words", "step-by-step"}
MATH_RE = re.compile(r"[=+\-*/^%]|\b\d+\b")


def _syll(w):
    w = w.lower(); n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1: n -= 1
    return max(n, 1)


def feats(text):
    text = text or ""
    words = re.findall(r"\b\w+\b", text); low = [w.lower() for w in words]
    n = max(len(words), 1)
    n_sent = max(len([s for s in re.split(r"[.!?]+", text) if s.strip()]), 1)
    syl = [_syll(w) for w in words]
    fk = 0.39 * (n / n_sent) + 11.8 * (sum(syl) / n) - 15.59
    ent = (len(re.findall(r"(?<!^)\b[A-Z][a-z]+", text))
           + len(re.findall(r"\b[A-Z]{2,}\b", text)) + len(re.findall(r"\b\d+\b", text))) / n
    return [np.log1p(n), fk, sum(s >= 3 for s in syl) / n,
            len(re.findall(r"[,;:()\-\u2014]", text)) / n, ent,
            max((LEVEL.get(w, 0) for w in low), default=0),
            sum(w in REASON for w in low), sum(w in CONSTRAINT for w in low),
            sum(w in WH for w in low), min(len(MATH_RE.findall(text)), 10)]


# --- data + scores ----------------------------------------------------------
def load_split(sample):
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    df = load_dataset(DATASET, split="train").to_pandas()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    df = df[df["prompt"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    _, te = train_test_split(df, test_size=0.2, random_state=42)
    return te["prompt"].tolist()


def cpu_scores(prompts):
    b = joblib.load(MODEL_PATH)
    b = {k: b[k] for k in ("vec", "sc", "ridge")}
    T = b["vec"].transform(prompts)
    E = b["sc"].transform(np.asarray([feats(p) for p in prompts]))
    return b["ridge"].predict(sparse.hstack([T, E]).tocsr())


def finetuned_scores(prompts):
    cache = f"ft_scores_{len(prompts)}.npy"
    if os.path.exists(cache):
        print(f"  loaded cached fine-tuned scores ({cache})")
        return np.load(cache)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(FT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FT_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), 64):
            enc = tok(prompts[i:i + 64], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(dev)
            out.extend(model(**enc).logits.squeeze(-1).cpu().numpy().tolist())
    arr = np.asarray(out); np.save(cache, arr)
    return arr


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def frontier(est_z, required, caps_z, run_cost, n, spread=4.0, n_pts=81):
    """Sweep routing aggressiveness; return array of (served_frac, total_J)."""
    pts = []
    for b in np.linspace(-spread, spread, n_pts):
        chosen = np.digitize(est_z - b, caps_z)          # smaller b => cheaper tiers
        served = np.mean(chosen >= required)
        total = TIER_E[chosen].sum() + run_cost * n
        pts.append((served, total))
    return np.array(pts)


def energy_at(pts, target):
    """Cheapest total energy that reaches >= target served; else None."""
    ok = pts[pts[:, 0] >= target]
    return ok[:, 1].min() if len(ok) else None


def run(args):
    prompts = load_split(args.sample)
    n = len(prompts)
    print(f"Held-out prompts: {n:,}\nScoring ...")
    cpu = cpu_scores(prompts)
    ft = finetuned_scores(prompts)

    # z-score everything so the comparison is about RANKING, not scale/calibration
    truth_z = zscore(ft)
    caps_z = np.quantile(truth_z, [k / N for k in range(1, N)])
    required = np.digitize(truth_z, caps_z)               # each prompt's needed tier
    rng = np.random.default_rng(20240719)

    policies = [
        ("Fine-tuned (=oracle)", zscore(ft), ROUTER_COST_J["finetuned"]),
        ("CPU router (0.650)",  zscore(cpu), ROUTER_COST_J["cpu"]),
        ("random",              zscore(rng.normal(size=n)), 0.0),
    ]
    curves = {name: frontier(est, required, caps_z, cost, n)
              for name, est, cost in policies}

    small_J = TIER_E[0] * n
    large_J = TIER_E[-1] * n
    print(f"\nReference (no routing):  always-small {small_J/1000:6.2f} kJ (serves "
          f"{np.mean(0 >= required)*100:.0f}%)   always-large {large_J/1000:6.2f} kJ (serves 100%)")

    print(f"\n{'policy':<22}{'router J':>10}{'E@90% kJ':>11}{'E@99% kJ':>11}{'max served':>12}")
    for name, _, cost in policies:
        c = curves[name]
        e90, e99 = energy_at(c, 0.90), energy_at(c, 0.99)
        s90 = f"{e90/1000:.2f}" if e90 else "  --"
        s99 = f"{e99/1000:.2f}" if e99 else "  --"
        print(f"{name:<22}{cost*n:>10.1f}{s90:>11}{s99:>11}{c[:,0].max()*100:>11.1f}%")

    # save the curves so you can plot served-vs-energy for the paper
    with open("routing_frontier.csv", "w") as f:
        f.write("policy,served,total_J\n")
        for name in curves:
            for served, tot in curves[name]:
                f.write(f"{name},{served:.4f},{tot:.2f}\n")
    print("\nfrontier points -> routing_frontier.csv  (plot served vs total_J)")

    # -- does it matter? --
    ora, cpu_c = curves["Fine-tuned (=oracle)"], curves["CPU router (0.650)"]
    e_cpu, e_ora = energy_at(cpu_c, 0.90), energy_at(ora, 0.90)
    print("\n-- does it matter? --")
    if e_cpu:
        print(f"CPU router hits 90% served for {e_cpu/1000:.2f} kJ vs always-large "
              f"{large_J/1000:.2f} kJ -> {(1-e_cpu/large_J)*100:.0f}% less energy, same quality")
    if e_cpu and e_ora:
        print(f"Oracle hits 90% for {e_ora/1000:.2f} kJ; CPU needs {(e_cpu/e_ora-1)*100:+.0f}% more "
              f"generation-side to cover its ranking errors -- but its router runs "
              f"{ROUTER_COST_J['finetuned']/ROUTER_COST_J['cpu']:.0f}x cheaper.")
    print("Compare the CPU and oracle curves in the CSV: the gap between them is what"
          "\ncheap estimation costs; the gap to always-large is what routing saves.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    run(ap.parse_args())
