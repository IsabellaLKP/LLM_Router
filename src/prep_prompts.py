"""
prep_prompts.py

Builds a difficulty-stratified benchmark prompt set from
agentlans/prompt-difficulty (99,758 prompts, CC-BY-4.0).

Output: prompts.jsonl  ->  {"id", "prompt", "difficulty", "bin"}

Why not NeMo Curator: Curator's text pipeline officially targets Ubuntu +
NVIDIA GPU, and its pretrained classifiers (quality, domain) are GPU-only.
It will not run on the AMD/Windows setup. The filters below reproduce the
three Curator stages we actually need (word-count filter, non-alphanumeric
ratio filter, exact dedup) in ~30 lines of pandas. Document this in methods
as a tooling limitation, same as the CodeCarbon/AMD gap.

Deps: pip install datasets pandas
"""

import hashlib
import json
import re

import pandas as pd
from datasets import load_dataset

# --- config ---------------------------------------------------------------
N_BINS = 5           # difficulty strata
PER_BIN = 20         # prompts sampled per stratum
MIN_WORDS = 5
MAX_WORDS = 300      # keep prompts short enough that context size is not the confound
MAX_NON_ALNUM = 0.25
DROP_TOP_PCT = 0.02  # see note below
SEED = 42
OUT = "prompts.jsonl"

# The dataset card notes that jailbreak / unsafe prompts are scored as
# "difficult to answer", so the extreme upper tail is contaminated with
# refusal-bait rather than genuine task complexity. Trim it.
BLOCK = re.compile(
    r"\b(jailbreak|DAN mode|ignore (all )?(previous|prior) instructions|"
    r"NSFW|explicit sex|as a hacker|how to (make|build) a bomb)\b",
    re.I,
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def non_alnum_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)


def main():
    ds = load_dataset("agentlans/prompt-difficulty", split="train")
    df = ds.to_pandas()[["prompt", "difficulty"]].dropna()
    print(f"loaded {len(df)} prompts")

    # --- Curator-equivalent filtering stages ---
    df["n_words"] = df["prompt"].str.split().str.len()
    df = df[(df.n_words >= MIN_WORDS) & (df.n_words <= MAX_WORDS)]
    df = df[df["prompt"].map(non_alnum_ratio) <= MAX_NON_ALNUM]
    df = df[~df["prompt"].str.contains(BLOCK)]
    df["_h"] = df["prompt"].map(lambda t: hashlib.md5(norm(t).encode()).hexdigest())
    df = df.drop_duplicates("_h").drop(columns="_h")

    cutoff = df["difficulty"].quantile(1 - DROP_TOP_PCT)
    df = df[df["difficulty"] <= cutoff]
    print(f"{len(df)} prompts after filtering")

    # --- stratify by difficulty ---
    df["bin"] = pd.qcut(df["difficulty"], N_BINS, labels=range(N_BINS)).astype(int)
    sample = (
        df.groupby("bin", group_keys=False)
        .apply(lambda g: g.sample(min(PER_BIN, len(g)), random_state=SEED))
        .reset_index(drop=True)
    )

    with open(OUT, "w", encoding="utf-8") as f:
        for i, row in sample.iterrows():
            f.write(
                json.dumps(
                    {
                        "id": f"p{i:04d}",
                        "prompt": row["prompt"],
                        "difficulty": round(float(row["difficulty"]), 4),
                        "bin": int(row["bin"]),
                    }
                )
                + "\n"
            )

    print(f"wrote {len(sample)} prompts to {OUT}")
    print(sample.groupby("bin")["difficulty"].agg(["count", "min", "mean", "max"]))


if __name__ == "__main__":
    main()
