"""
improved_train.py -- train and evaluate the prompt-difficulty regressor.

Label-noise ceiling + regression on the continuous `difficulty` score, compared
across representations:
  * TF-IDF, TF-IDF + 10 handcrafted formula features (the deployed CPU router)
  * formula features alone (GBM head)
  * frozen bge-small embeddings with a linear (Ridge) and nonlinear (GBM) head
  * TF-IDF + bge + formula union (Ridge)
Embeddings are cached on disk (emb_bge_<n>.npy), so re-runs don't re-encode.

Predictions are also binned to 5 and 3 classes for comparison with the original
direct 5-way classifier (0.408 acc / 0.768 adjacent / 0.398 macro-F1).

Run:  python improved_train.py [--sample N] [--no-embed]
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET = "agentlans/prompt-difficulty"
META_COLS = {"prompt", "difficulty", "transformed_difficulty"}
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
    complex_r = sum(s >= 3 for s in syl) / n
    punct = len(re.findall(r"[,;:()\-\u2014]", text)) / n
    ent = (len(re.findall(r"(?<!^)\b[A-Z][a-z]+", text))
           + len(re.findall(r"\b[A-Z]{2,}\b", text))
           + len(re.findall(r"\b\d+\b", text))) / n
    bloom = max((LEVEL.get(w, 0) for w in low), default=0)
    return [np.log1p(n), fk, complex_r, punct, ent, bloom,
            sum(w in REASON for w in low), sum(w in CONSTRAINT for w in low),
            sum(w in WH for w in low), min(len(MATH_RE.findall(text)), 10)]


def eng_matrix(prompts):
    return np.asarray([feats(p) for p in prompts], dtype=float)


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def load_full(sample=None, seed=42):
    from datasets import load_dataset
    df = load_dataset(DATASET, split="train").to_pandas()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)
    model_cols = [c for c in df.columns if c not in META_COLS
                  and pd.api.types.is_numeric_dtype(df[c])]
    df = df[df["prompt"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    return df, model_cols


def noise_ceiling(df, model_cols, n_splits=50, seed=0):
    R = df[model_cols].to_numpy(float); rng = np.random.default_rng(seed)
    m, halves = R.shape[1], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_splits):
            idx = rng.permutation(m)
            ma = np.nanmean(R[:, idx[:m // 2]], axis=1)
            mb = np.nanmean(R[:, idx[m // 2:]], axis=1)
            ok = ~np.isnan(ma) & ~np.isnan(mb)
            halves.append(spearman(ma[ok], mb[ok]))
    r_half = float(np.mean(halves)); r_full = 2 * r_half / (1 + r_half)
    return r_half, r_full, np.sqrt(max(r_full, 0.0))


def cached_embed(prompts):
    path = f"emb_bge_{len(prompts)}.npy"
    if os.path.exists(path):
        Z = np.load(path)
        if len(Z) == len(prompts):
            print(f"  loaded cached embeddings ({path})")
            return Z
    from sentence_transformers import SentenceTransformer
    print(f"  encoding {len(prompts):,} prompts with {EMBED_MODEL} ...")
    Z = SentenceTransformer(EMBED_MODEL).encode(
        list(prompts), batch_size=256, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True)
    np.save(path, Z)
    return Z


def bin_report(yt_cont, yp_cont, nc, edges):
    cut = edges[1:-1]
    yt, yp = np.digitize(yt_cont, cut), np.clip(np.digitize(yp_cont, cut), 0, nc - 1)
    return (accuracy_score(yt, yp), float(np.mean(np.abs(yp - yt) <= 1)),
            f1_score(yt, yp, average="macro"))


def make_head(kind):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.07,
                                         max_leaf_nodes=31, random_state=42)


def run(args):
    print(f"Loading {DATASET} ...")
    df, model_cols = load_full(args.sample)
    print(f"  {len(df):,} prompts | panel columns: {len(model_cols)}")

    rh, rf, ceil = noise_ceiling(df, model_cols)
    print("\n== Label noise ceiling ==")
    print(f"  split-half {rh:.3f} | reliability {rf:.3f} | corr ceiling ≈ {ceil:.3f} "
          f"(R² ceiling ≈ {rf:.3f})")

    y = df["difficulty"].to_numpy(float)
    idx = np.arange(len(df))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=42)
    prompts = df["prompt"].tolist()
    tr_txt = [prompts[i] for i in itr]; te_txt = [prompts[i] for i in ite]
    ytr, yte = y[itr], y[ite]
    edges5 = np.array([-np.inf, *np.quantile(ytr, [.2, .4, .6, .8]), np.inf])
    edges3 = np.array([-np.inf, *np.quantile(ytr, [1 / 3, 2 / 3]), np.inf])

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, strip_accents="unicode")
    T_tr, T_te = vec.fit_transform(tr_txt), vec.transform(te_txt)
    sc = StandardScaler().fit(eng_matrix(tr_txt))
    E_tr, E_te = sc.transform(eng_matrix(tr_txt)), sc.transform(eng_matrix(te_txt))

    use_embed = not args.no_embed
    if use_embed:
        print("\nEmbeddings:")
        Z_all = cached_embed(prompts)
        Z_tr, Z_te = Z_all[itr], Z_all[ite]

    # (name, Xtr, Xte, head)  -- sparse for ridge-on-tfidf, dense for gbm
    hs = lambda *m: sparse.hstack(m).tocsr()
    configs = [
        ("TF-IDF",              T_tr, T_te, "ridge"),
        ("TF-IDF + formula",    hs(T_tr, E_tr), hs(T_te, E_te), "ridge"),
        ("formula only",        E_tr, E_te, "gbm"),
    ]
    if use_embed:
        configs += [
            ("bge  (ridge)",        Z_tr, Z_te, "ridge"),
            ("bge  (gbm)",          Z_tr, Z_te, "gbm"),
            ("bge + formula (gbm)", np.hstack([Z_tr, E_tr]), np.hstack([Z_te, E_te]), "gbm"),
            ("TF-IDF+bge+formula",  hs(T_tr, sparse.csr_matrix(Z_tr), E_tr),
                                    hs(T_te, sparse.csr_matrix(Z_te), E_te), "ridge"),
        ]

    print("\n== Regression on continuous difficulty ==")
    print(f"{'features':<22}{'head':>7}{'Spearman':>10}{'Pearson':>9}{'R2':>8}{'MAE':>8}")
    preds = {}
    for name, Xtr, Xte, head in configs:
        p = make_head(head).fit(Xtr, ytr).predict(Xte)
        preds[name] = p
        print(f"{name:<22}{head:>7}{spearman(p, yte):>10.3f}"
              f"{float(np.corrcoef(p, yte)[0,1]):>9.3f}{r2_score(yte, p):>8.3f}"
              f"{mean_absolute_error(yte, p):>8.3f}")

    print("\n== Predict-then-bin  (direct 5-way classifier was 0.408 / 0.768 / 0.398) ==")
    print(f"{'features':<22}{'classes':>8}{'acc':>8}{'adjacent':>10}{'macroF1':>9}")
    for name, p in preds.items():
        for nc, ed in ((5, edges5), (3, edges3)):
            acc, adj, mf1 = bin_report(yte, p, nc, ed)
            print(f"{name:<22}{nc:>8}{acc:>8.3f}{adj:>10.3f}{mf1:>9.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--no-embed", action="store_true")
    run(ap.parse_args())
