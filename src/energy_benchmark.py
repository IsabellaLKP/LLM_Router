"""
energy_benchmark.py -- energy benchmark that needs NO root.

The lab box locks /sys/.../energy_uj (RAPL) behind root, so CodeCarbon can't read
CPU energy and silently fabricates it. This avoids RAPL entirely:

  * GPU: poll nvmlDeviceGetPowerUsage() in a background thread and integrate
    power over each window (a real measurement -- pynvml works as your user).
  * CPU: psutil gives utilization without root; multiply by a STATED package
    power to model CPU energy. This is a transparent estimate, not a hidden one.
    Set --cpu-tdp to your Xeon's package TDP so the assumption is explicit and
    auditable in the paper.

Reports, per workload: throughput, GPU energy/prompt (measured), CPU energy/prompt
(modelled), and both net of an idle baseline. One-time training energy separately.

Run:
  python energy_benchmark.py --cpu-tdp 85          # CPU router only
  python energy_benchmark.py --cpu-tdp 85 --embed  # + bge + union
"""

import argparse
import re
import statistics as stats
import threading
import time
from time import perf_counter

import numpy as np
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import pynvml
    pynvml.nvmlInit()
    GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAVE_GPU = True
except Exception:
    HAVE_GPU = False

DATASET = "agentlans/prompt-difficulty"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

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


def build_pipeline(sample):
    from datasets import load_dataset
    df = load_dataset(DATASET, split="train").to_pandas()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    df = df[df["prompt"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    y = tr["difficulty"].to_numpy(float)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(tr["prompt"])
    sc = StandardScaler().fit(np.asarray([feats(p) for p in tr["prompt"]]))
    Etr = sc.transform(np.asarray([feats(p) for p in tr["prompt"]]))
    ridge = Ridge(alpha=1.0).fit(sparse.hstack([Xtr, Etr]).tocsr(), y)
    return vec, sc, ridge, te["prompt"].tolist()


# --- sampler ---------------------------------------------------------------
class Sampler:
    """Background thread integrating GPU power (W) and CPU utilisation over time."""
    def __init__(self, cpu_tdp, hz=50):
        self.cpu_tdp = cpu_tdp; self.dt = 1.0 / hz
        self._stop = False; self.gpu_j = 0.0; self.cpu_util_area = 0.0; self.secs = 0.0

    def _loop(self):
        psutil.cpu_percent(None)      # prime
        last = perf_counter()
        while not self._stop:
            time.sleep(self.dt)
            now = perf_counter(); span = now - last; last = now
            if HAVE_GPU:
                w = pynvml.nvmlDeviceGetPowerUsage(GPU) / 1000.0   # mW -> W
                self.gpu_j += w * span
            self.cpu_util_area += (psutil.cpu_percent(None) / 100.0) * span
            self.secs += span

    def __enter__(self):
        self._stop = False; self.gpu_j = 0.0; self.cpu_util_area = 0.0; self.secs = 0.0
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True; self._t.join()

    def cpu_j(self):                  # modelled: util-seconds x package TDP
        return self.cpu_util_area * self.cpu_tdp


def measure(step, pool, duration, cpu_tdp, warmup=100):
    for i in range(min(warmup, len(pool))):
        step(pool[i])
    with Sampler(cpu_tdp) as s:
        t0 = perf_counter(); count = 0; n = len(pool)
        while perf_counter() - t0 < duration:
            step(pool[count % n]); count += 1
        elapsed = perf_counter() - t0
    return dict(count=count, elapsed=elapsed, thru=count / elapsed,
                gpu_j=s.gpu_j, cpu_j=s.cpu_j())


def measure_idle(duration, cpu_tdp):
    with Sampler(cpu_tdp) as s:
        time.sleep(duration)
    return s.gpu_j / duration, s.cpu_j() / duration   # (idle_gpu_W, idle_cpu_W)


def run(args):
    print(f"GPU sampling: {'on (NVML)' if HAVE_GPU else 'OFF (pynvml unavailable)'}")
    print(f"CPU energy model: utilisation x {args.cpu_tdp} W package (STATED estimate)\n")
    print("Building pipeline (one-time) ...")
    with Sampler(args.cpu_tdp) as s:
        vec, sc, ridge, pool = build_pipeline(args.sample)
    print(f"  one-time training energy ~ CPU {s.cpu_j():,.0f} J + GPU {s.gpu_j:,.0f} J")
    print(f"  {len(pool):,} held-out prompts\n")

    def s_formula(p): feats(p)
    def s_tfidf(p): vec.transform([p])
    prerow = sparse.hstack([vec.transform([pool[0]]),
                            sc.transform([feats(pool[0])])]).tocsr()
    def s_ridge(_): ridge.predict(prerow)
    def s_cpu(p):
        ridge.predict(sparse.hstack([vec.transform([p]),
                                     sc.transform([feats(p)])]).tocsr())

    workloads = [("formula only", s_formula), ("tfidf transform", s_tfidf),
                 ("ridge predict", s_ridge), ("FULL CPU router", s_cpu)]

    if args.embed:
        from sentence_transformers import SentenceTransformer
        with Sampler(args.cpu_tdp) as s:
            emb = SentenceTransformer(EMBED_MODEL)
        print(f"  one-time bge load ~ GPU {s.gpu_j:,.0f} J\n")
        def s_bge(p): emb.encode([p], show_progress_bar=False, normalize_embeddings=True)
        def s_union(p):
            z = emb.encode([p], show_progress_bar=False, normalize_embeddings=True)
            ridge.predict(sparse.hstack([vec.transform([p]), sparse.csr_matrix(z),
                                         sc.transform([feats(p)])]).tocsr())
        workloads += [("bge encode (GPU)", s_bge), ("FULL union", s_union)]

    print(f"Idle baseline ({args.duration}s) ...")
    ig, ic = measure_idle(args.duration, args.cpu_tdp)
    print(f"  idle: CPU {ic:.1f} W, GPU {ig:.1f} W\n")

    print(f"Per-prompt inference  ({args.trials} trials x {args.duration}s)  [mJ/prompt]")
    print(f"{'workload':<20}{'thru/s':>9}{'cpu net':>9}{'gpu net':>9}{'TOTAL net':>11}")
    for name, step in workloads:
        rows = [measure(step, pool, args.duration, args.cpu_tdp) for _ in range(args.trials)]
        thru = stats.mean(r["thru"] for r in rows)
        # net = subtract idle power over the window, per prompt, in mJ
        cpu = stats.mean(max(r["cpu_j"] - ic * r["elapsed"], 0) / r["count"] for r in rows) * 1000
        gpu = stats.mean(max(r["gpu_j"] - ig * r["elapsed"], 0) / r["count"] for r in rows) * 1000
        print(f"{name:<20}{thru:>9,.0f}{cpu:>9.3f}{gpu:>9.3f}{cpu+gpu:>11.3f}")

    print("\nCPU column is modelled (util x stated TDP); GPU column is measured (NVML).")
    print("Compare TOTAL net mJ/prompt against a single LLM generation (100s-1000s of J).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-tdp", type=float, default=85.0,
                    help="Xeon package TDP (W) for the CPU energy model; state it in the paper")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--embed", action="store_true")
    run(ap.parse_args())
