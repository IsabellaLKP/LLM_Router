
from datasets import load_dataset

# Load the prompt difficulty dataset
df = load_dataset(
    "agentlans/prompt-difficulty",
    split="train"
).to_pandas()

# Check the dataset size and difficulty distribution
print(f"Total prompts: {len(df):,}")
print(f"Average difficulty: {df['difficulty'].mean():.3f}")
print(f"Minimum difficulty: {df['difficulty'].min():.3f}")
print(f"Maximum difficulty: {df['difficulty'].max():.3f}")


import re

# Preliminary keywords associated with potentially unsafe requests
safety_terms = [
    "hack", "steal", "meth", "weapon",
    "bomb", "fraud", "phishing", "forgery",
    "bypass", "malware", "poison"
]

# Match whole words, including simple word endings
pattern = r"\b(?:" + "|".join(
    re.escape(term) for term in safety_terms
) + r")\b"

# Identify prompts containing these terms
df["safety_keyword_match"] = (
    df["prompt"]
    .fillna("")
    .astype(str)
    .str.contains(pattern, case=False, regex=True)
)

matched = df[df["safety_keyword_match"]]
unmatched = df[~df["safety_keyword_match"]]

print("\n--- Preliminary Safety Keyword Analysis ---")
print(f"Keyword matches: {len(matched):,}")
print(f"Percentage: {len(matched) / len(df) * 100:.2f}%")
print(f"Average difficulty (matched): {matched['difficulty'].mean():.3f}")
print(f"Average difficulty (unmatched): {unmatched['difficulty'].mean():.3f}")

print("\nSample matching prompts:")
for prompt in matched["prompt"].head(10):
    print("-", str(prompt)[:120])


print("\n--- Matches by Keyword ---")

for term in safety_terms:
    term_pattern = rf"\b{re.escape(term)}\b"

    count = df["prompt"].fillna("").astype(str).str.contains(
        term_pattern,
        case=False,
        regex=True
    ).sum()

    print(f"{term:<12}: {count:,}")
