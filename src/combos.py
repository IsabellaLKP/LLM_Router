"""
test_h1_combinations.py -- Evaluate custom combinations of H1 feature groups
"""

import re
import time
import itertools
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from router_cli import feats as orig_feats

# ---------------------------------------------------------------------------
# 1. H1 REGEX EXTRACTORS
# ---------------------------------------------------------------------------
FORMAT_RE  = re.compile(r"\b(json|xml|yaml|table|markdown|csv|bullet|latex|schema)\b", re.I)
EXPLAIN_RE = re.compile(r"\b(step-by-step|detailed|elaborate|explain|walkthrough|thorough)\b", re.I)
REASON_RE  = re.compile(r"\b(analyze|prove|derive|critique|optimize|eval|synthesize|compare)\b", re.I)
TASK_RE    = re.compile(r"\b(must|should|without|only|exactly|ensure|avoid|constraint)\b", re.I)

H1_FUNCS = {
    "F_format":  lambda p: len(FORMAT_RE.findall((p or "").lower())),
    "F_explain": lambda p: len(EXPLAIN_RE.findall((p or "").lower())),
    "F_reason":  lambda p: len(REASON_RE.findall((p or "").lower())),
    "F_task":    lambda p: len(TASK_RE.findall((p or "").lower()))
}

def extract_h1_combination(prompts, active_groups):
    """Extracts a matrix for selected feature groups."""
    matrix = []
    for p in prompts:
        row = [H1_FUNCS[grp](p) for grp in active_groups]
        matrix.append(row)
    return np.array(matrix, dtype=float)

# ---------------------------------------------------------------------------
# 2. DEFINE COMBINATIONS YOU WANT TO TRY HERE!
# ---------------------------------------------------------------------------
# You can manually specify combinations like this:
CUSTOM_COMBINATIONS = [
    ["F_reason", "F_task"],                           # 2-group pair
    ["F_format", "F_task"],                              # 2-group pair
    ["F_format", "F_explain", "F_reason"],               # 3-group combo
    ["F_format", "F_explain", "F_reason", "F_task"],     # All 4 groups together
]

# Or set RUN_ALL_POSSIBLE = True to automatically sweep all 15 combinations!
RUN_ALL_POSSIBLE = True

# ---------------------------------------------------------------------------
# 3. BENCHMARK RUNNER
# ---------------------------------------------------------------------------
def run_combination_test():
    print("Loading prompt difficulty dataset...")
    df = load_dataset("agentlans/prompt-difficulty", split="train").to_pandas()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Pre-compute TF-IDF + Baseline Formula Features
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_tr_tfidf = vec.fit_transform(train_df['prompt'])
    X_te_tfidf = vec.transform(test_df['prompt'])
    
    sc = StandardScaler()
    X_tr_orig = sc.fit_transform(np.array([orig_feats(p) for p in train_df['prompt']]))
    X_te_orig = sc.transform(np.array([orig_feats(p) for p in test_df['prompt']]))
    
    X_tr_base = sparse.hstack([X_tr_tfidf, X_tr_orig])
    X_te_base = sparse.hstack([X_te_tfidf, X_te_orig])
    
    # Baseline Score
    base_model = Ridge(alpha=1.0)
    base_model.fit(X_tr_base, train_df['difficulty'])
    base_preds = base_model.predict(X_te_base)
    base_rho, _ = spearmanr(test_df['difficulty'], base_preds)
    
    print("\n" + "="*75)
    print(f"DEPLOYED BASELINE ROUTER SPEARMAN (ρ): {base_rho:.4f}")
    print("="*75 + "\n")
    
    # Determine combinations to evaluate
    if RUN_ALL_POSSIBLE:
        all_groups = list(H1_FUNCS.keys())
        combos = []
        for r in range(1, len(all_groups) + 1):
            combos.extend([list(c) for c in itertools.combinations(all_groups, r)])
    else:
        combos = CUSTOM_COMBINATIONS
        
    results = []
    
    for combo in combos:
        combo_name = " + ".join(combo)
        
        # Extract columns for this specific combination
        tr_h1 = extract_h1_combination(train_df['prompt'], combo)
        te_h1 = extract_h1_combination(test_df['prompt'], combo)
        
        h1_sc = StandardScaler()
        tr_h1_scaled = h1_sc.fit_transform(tr_h1)
        te_h1_scaled = h1_sc.transform(te_h1)
        
        # Attach combination columns to baseline
        X_tr_aug = sparse.hstack([X_tr_base, tr_h1_scaled])
        X_te_aug = sparse.hstack([X_te_base, te_h1_scaled])
        
        aug_model = Ridge(alpha=1.0)
        aug_model.fit(X_tr_aug, train_df['difficulty'])
        aug_preds = aug_model.predict(X_te_aug)
        aug_rho, _ = spearmanr(test_df['difficulty'], aug_preds)
        
        results.append({
            "Combination": combo_name,
            "# Features Added": len(combo),
            "Spearman (ρ)": f"{aug_rho:.4f}",
            "Δ vs Baseline": f"{aug_rho - base_rho:+.4f}",
            "Est. Energy Overhead": "< 0.001 mJ"
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.sort_values(by="Spearman (ρ)", ascending=False).to_string(index=False))

if __name__ == "__main__":
    run_combination_test()