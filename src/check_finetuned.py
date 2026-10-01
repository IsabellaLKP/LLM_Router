"""
check_finetuned.py -- score the fine-tuned difficulty model as an upper reference.

Loads agentlans/bge-small-en-v1.5-prompt-difficulty (bge-small fine-tuned on the
prompt-difficulty labels), runs it on a sample of the dataset, and reports
Pearson / Spearman against the panel `difficulty` column, plus the 5 easiest
and 5 hardest prompts side by side.

Note: this model was fine-tuned on the same dataset, so scores on prompts it
saw during training are optimistic. Treat the result as an upper reference.

Run:  TOKENIZERS_PARALLELISM=false python check_finetuned.py
"""

import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

MODEL = "agentlans/bge-small-en-v1.5-prompt-difficulty"
N = 500  # how many prompts to check

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
print("device:", device, "| output dim:", model.config.num_labels)

df = load_dataset("agentlans/prompt-difficulty", split="train").to_pandas()
df = df.sample(n=N, random_state=42).reset_index(drop=True)

preds = []
with torch.no_grad():
    for i in range(0, len(df), 32):
        batch = df["prompt"].iloc[i:i+32].tolist()
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        out = model(**enc).logits.squeeze(-1)      # regression head -> one value per prompt
        preds.extend(out.cpu().numpy().tolist())

pred = np.array(preds)
true = df["difficulty"].to_numpy()

def spearman(a, b):
    return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]

print(f"Pearson  {np.corrcoef(pred, true)[0,1]:+.3f}")
print(f"Spearman {spearman(pred, true):+.3f}")
print("\n predicted   panel   prompt")
order = np.argsort(true)
for i in list(order[:5]) + list(order[-5:]):          # 5 easiest, 5 hardest
    print(f"  {pred[i]:8.3f}  {true[i]:6.2f}   {df['prompt'][i][:60]}")
