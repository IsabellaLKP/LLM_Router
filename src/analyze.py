"""
analyze.py

Joins energy.csv and similarity.csv and answers the three questions:

  1. How much energy does each model use for the same task?
  2. Does the small model actually give the same answer?
  3. If you routed each prompt to the smallest model that still agreed with
     the reference, how much energy would that save vs always using the
     largest model?

Question 3 is the headline number for the paper. Everything else feeds it.

Outputs: summary_by_model.csv, summary_by_bin.csv, router_result.csv

Deps: pip install pandas
"""

import pandas as pd

ENERGY = "energy.csv"
SIMILARITY = "similarity.csv"
REFERENCE_MODEL = "llama3.1:8b"
THRESHOLD = 0.90          # cosine at or above this = "close enough"
MODEL_ORDER = [           # smallest to largest; router walks this list
    "qwen2.5:0.5b",
    "llama3.2:1b",
    "llama3.2:3b",
    "qwen2.5:7b",
    "llama3.1:8b",
]


def load():
    e = pd.read_csv(ENERGY)
    s = pd.read_csv(SIMILARITY)
    # average the repeated runs to cut measurement noise
    e = (
        e.groupby(["model", "prompt_id", "bin", "difficulty"], as_index=False)
        .agg(
            joules=("total_joules", "mean"),
            joules_sd=("total_joules", "std"),
            cpu_ram_joules=("cpu_ram_joules", "mean"),
            seconds=("wall_seconds", "mean"),
            output_tokens=("output_tokens", "mean"),
        )
    )
    return e.merge(s, on=["model", "prompt_id"], how="left")


def by_model(df):
    out = df.groupby("model").agg(
        mean_joules=("joules", "mean"),
        median_joules=("joules", "median"),
        mean_sd=("joules_sd", "mean"),
        joules_per_token=("joules", lambda x: x.sum()),
        mean_cosine=("cosine", "mean"),
        pct_agree=("cosine", lambda x: (x >= THRESHOLD).mean() * 100),
        mean_len_ratio=("len_ratio", "mean"),
    )
    tok = df.groupby("model")["output_tokens"].sum()
    out["joules_per_token"] = out["joules_per_token"] / tok
    # the metric that matters: energy spent per answer you would actually accept
    accepted = df[df["cosine"] >= THRESHOLD].groupby("model").size()
    total_j = df.groupby("model")["joules"].sum()
    out["joules_per_accepted_answer"] = (total_j / accepted).round(1)
    return out.round(3).reindex([m for m in MODEL_ORDER if m in out.index])


def by_bin(df):
    out = df.groupby(["bin", "model"]).agg(
        mean_joules=("joules", "mean"),
        mean_cosine=("cosine", "mean"),
        pct_agree=("cosine", lambda x: (x >= THRESHOLD).mean() * 100),
        mean_output_tokens=("output_tokens", "mean"),
    )
    return out.round(3)


def router(df):
    """Oracle router: cheapest model whose output still matches the reference."""
    rows = []
    for pid, g in df.groupby("prompt_id"):
        g = g.set_index("model")
        baseline = g.loc[REFERENCE_MODEL, "joules"] if REFERENCE_MODEL in g.index else None
        chosen, chosen_j = REFERENCE_MODEL, baseline
        for m in MODEL_ORDER:
            if m not in g.index:
                continue
            if g.loc[m, "cosine"] >= THRESHOLD:
                chosen, chosen_j = m, g.loc[m, "joules"]
                break
        rows.append({
            "prompt_id": pid,
            "bin": g["bin"].iloc[0],
            "difficulty": g["difficulty"].iloc[0],
            "chosen_model": chosen,
            "routed_joules": chosen_j,
            "baseline_joules": baseline,
        })
    r = pd.DataFrame(rows)
    r["saved_joules"] = r["baseline_joules"] - r["routed_joules"]
    return r


def main():
    df = load()

    m = by_model(df)
    b = by_bin(df)
    r = router(df)

    m.to_csv("summary_by_model.csv")
    b.to_csv("summary_by_bin.csv")
    r.to_csv("router_result.csv", index=False)

    pd.set_option("display.width", 140)
    print("\n--- per model (runs averaged) ---")
    print(m)

    print("\n--- per difficulty bin (0 = easiest) ---")
    print(b)

    total_base = r["baseline_joules"].sum()
    total_routed = r["routed_joules"].sum()
    print("\n--- oracle router vs always-largest ---")
    print(f"threshold           cosine >= {THRESHOLD}")
    print(f"baseline energy     {total_base:.1f} J")
    print(f"routed energy       {total_routed:.1f} J")
    print(f"saved               {total_base - total_routed:.1f} J "
          f"({(1 - total_routed / total_base) * 100:.1f}%)")
    print("\nmodel selected, by difficulty bin:")
    print(pd.crosstab(r["bin"], r["chosen_model"]))

    print("\nnote: this is an oracle router, it sees the reference answer.")
    print("a real router predicts difficulty first, so treat this as the")
    print("upper bound on what prompt pre-classification could save.")


if __name__ == "__main__":
    main()
