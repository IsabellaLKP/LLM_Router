"""
router_cli.py -- score individual prompts through the difficulty router, interactively.

First run trains the CPU router (TF-IDF + 10 formula features -> Ridge) on the
dataset and caches it to router_model.joblib. Every run after that loads the
cache and starts instantly. Then it drops you into a prompt where you type a
prompt and get its difficulty score + class back.

  python router_cli.py                 # load cache, or build it the first time
  python router_cli.py --retrain       # force a rebuild
  python router_cli.py --sample 20000  # faster first build (smaller train set)

Also importable as the deployable router:
  from router_cli import load_or_build, score
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
NAMES5 = ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"]
NAMES3 = ["Easy", "Medium", "Hard"]

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


# ---------------------------------------------------------------------------
def build_bundle(prompts, targets):
    """Fit vectorizer + scaler + Ridge and derive class bin edges."""
    y = np.asarray(targets, float)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, strip_accents="unicode")
    T = vec.fit_transform(prompts)
    sc = StandardScaler().fit(np.asarray([feats(p) for p in prompts]))
    E = sc.transform(np.asarray([feats(p) for p in prompts]))
    ridge = Ridge(alpha=1.0).fit(sparse.hstack([T, E]).tocsr(), y)
    edges5 = np.array([-np.inf, *np.quantile(y, [.2, .4, .6, .8]), np.inf])
    edges3 = np.array([-np.inf, *np.quantile(y, [1 / 3, 2 / 3]), np.inf])
    return dict(vec=vec, sc=sc, ridge=ridge, edges5=edges5, edges3=edges3)


def build_and_save(path, sample=None):
    from datasets import load_dataset
    print(f"Building router from {DATASET} (one-time) ...")
    df = load_dataset(DATASET, split="train").to_pandas()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)
    df = df[df["prompt"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    tr, _ = train_test_split(df, test_size=0.2, random_state=42)
    bundle = build_bundle(tr["prompt"].tolist(), tr["difficulty"].to_numpy(float))
    joblib.dump(bundle, path)
    print(f"  saved -> {path}\n")
    return bundle


def load_or_build(path, retrain, sample):
    if retrain or not os.path.exists(path):
        return build_and_save(path, sample)
    print(f"Loaded cached router ({path})\n")
    return joblib.load(path)


def score(bundle, prompt):
    f = feats(prompt)
    row = sparse.hstack([bundle["vec"].transform([prompt]),
                         bundle["sc"].transform([f])]).tocsr()
    d = float(bundle["ridge"].predict(row)[0])
    c5 = int(np.clip(np.digitize(d, bundle["edges5"][1:-1]), 0, 4))
    c3 = int(np.clip(np.digitize(d, bundle["edges3"][1:-1]), 0, 2))
    return d, c5, c3, f


# ---------------------------------------------------------------------------
BANNER = """
Difficulty Router - interactive
Type a prompt and press Enter to score it. Commands:
  :features   toggle the 10-feature breakdown (currently {})
  :retrain    rebuild the model from the dataset
  :help       show this menu
  :quit       exit
"""


def repl(bundle, path):
    show_features = False
    print(BANNER.format("off"))
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return

        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            print("bye"); return
        if line == ":help":
            print(BANNER.format("on" if show_features else "off")); continue
        if line == ":features":
            show_features = not show_features
            print(f"feature breakdown {'on' if show_features else 'off'}"); continue
        if line == ":retrain":
            bundle = build_and_save(path); continue

        d, c5, c3, f = score(bundle, line)
        print(f"  predicted difficulty : {d:+.3f}   (panel scale, ~ -8..9)")
        print(f"  5-class              : {NAMES5[c5]}   (class {c5}/4)")
        print(f"  3-class              : {NAMES3[c3]}   (class {c3}/2)")
        if show_features:
            print("  features:")
            for name, val in zip(FEATURE_NAMES, f):
                print(f"    {name:<20}{val:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    bundle = load_or_build(args.model_path, args.retrain, args.sample)
    repl(bundle, args.model_path)


if __name__ == "__main__":
    main()
