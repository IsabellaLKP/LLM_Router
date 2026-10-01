"""
error_analysis.py -- where and why does the CPU router miss, and which new
features would actually fix it?

Reproduces the held-out split, scores it with the trained router (loads
router_model.joblib if present, else builds it), then:

  1. Aggregate error (Spearman, MAE, residual stats).
  2. Worst misses overall, and split by direction:
       under-predicted (model too EASY)  -> Tetris / subjective live here
       over-predicted  (model too HARD)  -> verbose-but-trivial live here
  3. Residual diagnostics: correlation of each EXISTING feature with the
     residual, and -- the useful part -- for a set of CANDIDATE new features
     (opinion/subjective, build-task, code, creative), how much they line up
     with the misses. A candidate whose presence coincides with a systematic
     one-directional residual is a feature worth adding; one that doesn't is
     an anecdote. Add features against this, not against two hand-picked prompts.

Run:  python error_analysis.py [--sample N] [--top 15]
"""

import argparse
import os
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATASET = "agentlans/prompt-difficulty"
MODEL_PATH = "router_model.joblib"

FEATURE_NAMES = ["log word count", "FK grade", "complex-word ratio",
                 "punctuation density", "entity density", "max Bloom level",
                 "reasoning verbs", "constraint markers", "wh-words", "math load"]

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

# --- candidate NEW features (what we're testing the value of) --------------
SUPERLATIVE = re.compile(r"\b\w+est\b")
OPINION = {"best", "worst", "better", "should", "favorite", "favourite", "vs",
           "versus", "recommend", "opinion", "prefer", "pros", "cons", "worth",
           "most", "least", "top", "ideal"}
BUILD_VERBS = {"build", "implement", "create", "develop", "design", "make",
               "write", "code", "program", "generate", "construct"}
ARTIFACT = {"game", "app", "application", "website", "site", "parser", "compiler",
            "engine", "api", "bot", "tool", "algorithm", "system", "database",
            "server", "dashboard", "clone", "simulator", "chatbot", "framework"}
CODE = {"python", "javascript", "java", "function", "code", "sql", "html", "css",
        "react", "class", "script", "loop", "array"}
CREATIVE = {"story", "poem", "essay", "novel", "song", "lyrics", "screenplay",
            "narrative", "pantoum"}


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


def candidates(text):
    low = (text or "").lower(); toks = set(re.findall(r"\b\w+\b", low))
    return {
        "opinion/subjective": bool(toks & OPINION) or bool(SUPERLATIVE.search(low)),
        "build-task": bool(toks & BUILD_VERBS) and bool(toks & ARTIFACT),
        "code-task": bool(toks & CODE),
        "creative-task": bool(toks & CREATIVE),
    }


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


# ---------------------------------------------------------------------------
def build_bundle(prompts, y):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, strip_accents="unicode")
    T = vec.fit_transform(prompts)
    sc = StandardScaler().fit(np.asarray([feats(p) for p in prompts]))
    E = sc.transform(np.asarray([feats(p) for p in prompts]))
    ridge = Ridge(alpha=1.0).fit(sparse.hstack([T, E]).tocsr(), np.asarray(y, float))
    return dict(vec=vec, sc=sc, ridge=ridge)


def predict_all(bundle, prompts):
    T = bundle["vec"].transform(prompts)
    E = bundle["sc"].transform(np.asarray([feats(p) for p in prompts]))
    return bundle["ridge"].predict(sparse.hstack([T, E]).tocsr())


def load_data(sample):
    from datasets import load_dataset
    df = load_dataset(DATASET, split="train").to_pandas()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    df = df[df["prompt"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    return train_test_split(df, test_size=0.2, random_state=42)


def clip(s, w=68):
    s = " ".join((s or "").split())
    return s[:w] + ("…" if len(s) > w else "")


def run(args):
    tr, te = load_data(args.sample)
    if os.path.exists(MODEL_PATH):
        print(f"Loaded {MODEL_PATH} (predicting held-out split it wasn't trained on)")
        bundle = joblib.load(MODEL_PATH)
        if "vec" not in bundle:            # router_cli bundle has extra keys; same core
            bundle = {k: bundle[k] for k in ("vec", "sc", "ridge")}
    else:
        print("No cached model; building on the train split ...")
        bundle = build_bundle(tr["prompt"].tolist(), tr["difficulty"].to_numpy(float))

    prompts = te["prompt"].tolist()
    true = te["difficulty"].to_numpy(float)
    pred = predict_all(bundle, prompts)
    resid = true - pred                    # +ve => model UNDER-rated (too easy)
    F = np.asarray([feats(p) for p in prompts])
    cand = {k: np.asarray([candidates(p)[k] for p in prompts]) for k in
            ("opinion/subjective", "build-task", "code-task", "creative-task")}

    print(f"\nHeld-out n = {len(te):,}")
    print(f"Spearman {spearman(pred, true):.3f} | MAE {np.mean(np.abs(resid)):.3f} "
          f"| mean residual {resid.mean():+.3f}")

    def show(title, idx):
        print(f"\n== {title} ==")
        print(f"{'panel':>6}{'pred':>7}{'err':>7}  tags / prompt")
        for i in idx:
            tags = ",".join(k.split("/")[0] for k, v in cand.items() if v[i])
            print(f"{true[i]:>6.2f}{pred[i]:>7.2f}{resid[i]:>+7.2f}  "
                  f"[{tags}] {clip(prompts[i])}")

    order = np.argsort(-np.abs(resid))
    show(f"Worst {args.top} misses (any direction)", order[:args.top])
    under = np.argsort(-resid)             # model too EASY (Tetris/subjective)
    show(f"Top {args.top} UNDER-predicted  (model too easy)", under[:args.top])
    over = np.argsort(resid)               # model too HARD (verbose/trivial)
    show(f"Top {args.top} OVER-predicted  (model too hard)", over[:args.top])

    print("\n== Existing features vs residual (corr; +ve => feature marks under-rated prompts) ==")
    for name, col in zip(FEATURE_NAMES, F.T):
        if np.std(col) == 0:
            print(f"  {name:<20}   const (no variance on this split)")
        else:
            print(f"  {name:<20}{np.corrcoef(col, resid)[0, 1]:+.3f}")

    print("\n== Candidate NEW features: do they mark systematic misses? ==")
    print(f"{'candidate':<20}{'n':>7}{'mean resid (has)':>18}{'(not)':>9}{'Δ':>8}  verdict")
    for name, mask in cand.items():
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:<18}{n:>7}   (never fires on this split)"); continue
        rh, rn = resid[mask].mean(), resid[~mask].mean()
        delta = rh - rn
        verdict = "ADD - systematic under-rating" if delta > 0.30 and n >= 25 else \
                  "add - systematic over-rating" if delta < -0.30 and n >= 25 else \
                  "weak / noisy"
        print(f"  {name:<18}{n:>7}{rh:>+18.3f}{rn:>+9.3f}{delta:>+8.3f}  {verdict}")

    print("\nInterpretation: a candidate with large +Δ marks prompts the model rates"
          "\ntoo easy (adding it lets Ridge push them up); large -Δ, too hard. 'weak'"
          "\nmeans it won't move aggregate error -- don't add it just for a few prompts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--top", type=int, default=15)
    run(ap.parse_args())
