# Difficulty-Aware LLM Routing for Lower AI Energy Use

Summer research project at Siena University, part of **The Ethical Implications of AI**
(Computer Science, Environmental Studies & Sciences, and the Academic Center for Sustainability).

**Question:** Can we cut the energy cost of using large language models by sending easy prompts
to small models and hard prompts to large ones, using a router cheap enough to run on a CPU?

**Short answer:** Yes. On our hardware, a CPU-only router cut total generation energy by about 37%
compared with always using the largest model, and the router itself costs under 10 mJ per prompt.

## Team

- **Student researcher:** Max Vanderbeek
- **Faculty:** Dr. Ting Liu, Dr. Kate Meierdiercks

## How it works

1. **Measure** how much energy each local model uses per generation (Ollama on an NVIDIA RTX 2070, GPU power read through NVML).
2. **Predict** prompt difficulty with a lightweight regressor: TF-IDF + 10 handcrafted text features + Ridge regression, trained on the [`agentlans/prompt-difficulty`](https://huggingface.co/datasets/agentlans/prompt-difficulty) dataset.
3. **Route** each prompt to the smallest model tier its predicted difficulty calls for, and compare total energy against baselines.

## Key results

Generation energy per model (`measure_gen.py`, 100 generations each, output capped at 256 tokens, idle subtracted):

| Model | Net energy per generation |
|---|---|
| qwen2.5:0.5b | 72.9 J |
| llama3.2:1b | 128.0 J |
| llama3.2:3b | 221.8 J |
| llama3.1:8b | 486.6 J |

Routing experiment (`routing_experiment.py`, 19,952 held-out prompts, energy needed to adequately serve 90% of prompts):

| Policy | Total energy (kJ) | Savings vs. always-large |
|---|---|---|
| Always use largest model | 9,664 | baseline |
| Random routing | 8,152 | 16% |
| **CPU router (deployed)** | **6,099** | **37%** |
| Fine-tuned bge-small router (includes its own 6.4 kJ running cost) | 4,544 | 53% |

Difficulty prediction quality (`improved_train.py`, `check_finetuned.py`; Spearman correlation with the label):

| Representation | Spearman |
|---|---|
| Handcrafted formula features (GBM) | 0.499 |
| Frozen bge-small embeddings (Ridge / GBM) | 0.550 / 0.585 |
| TF-IDF | 0.633 |
| **TF-IDF + formula (deployed router)** | **0.650** |
| TF-IDF + bge + formula union | 0.687 |
| Fine-tuned bge-small (reference) | 0.895 |
| Label-noise ceiling | ~0.910 |

Other findings:

- **The router is nearly free.** The full CPU router costs about 8.7 mJ per prompt at roughly 500 prompts/sec (`energy_benchmark.py`), thousands of times less than a single generation.
- **The biggest errors come from bad labels.** All 15 largest router errors were safety-refusal prompts that the dataset labels as very hard because models refused them, not because they are complex (`error_analysis.py`).
- **Frozen embeddings are not enough.** Frozen bge-small did worse than TF-IDF. Fine-tuning is what captures semantic difficulty.
- **Oracle savings are ceilings.** An earlier oracle experiment (`analyze.py`) found ~30% savings, but the oracle sees the reference answer. The predictive router numbers above are the realistic claims.
- **qwen2.5:7b is a dominated option** (near-8B energy with lower agreement), so it was dropped from the routing tiers.

See [`docs/METHODS_NOTES.md`](docs/METHODS_NOTES.md) for measurement lessons, including where CodeCarbon silently gives wrong numbers.

## Repository layout

```
src/            all main scripts (run them from inside src/)

### Scripts in `src/`

**Study 1: model energy and answer agreement (oracle router)**

| Script | What it does | Writes |
|---|---|---|
| `prep_prompts.py` | Filters the dataset and samples 100 difficulty-stratified prompts | `prompts.jsonl` |
| `run_energy_benchmark_nvidia.py` | Runs every prompt through 5 models x 3 runs, CodeCarbon energy per call | `energy.csv`, `outputs.jsonl` |
| `score_similarity.py` | Compares each model's answer to the 8B reference (cosine, Jaccard, length) | `similarity.csv` |
| `analyze.py` | Per-model and per-difficulty tables plus the oracle router | `summary_by_*.csv`, `router_result.csv` |

**Study 2: predictive router and end-to-end savings**

| Script | What it does | Writes |
|---|---|---|
| `improved_train.py` | Label-noise ceiling and comparison of all router representations | `emb_bge_*.npy` cache |
| `check_finetuned.py` | Scores the fine-tuned bge-small model as the upper reference | |
| `router_cli.py` | Trains and saves the deployed CPU router; interactive prompt scorer | `router_model.joblib` |
| `error_analysis.py` | Worst misses and candidate-feature tests (needs `router_model.joblib`) | |
| `energy_benchmark.py` | Root-free router energy: NVML for GPU, CPU utilization x stated TDP | |
| `measure_gen.py` | Net generation energy per model tier via NVML | |
| `routing_experiment.py` | Quality-vs-energy frontier for each routing policy (needs `router_model.joblib`) | `routing_frontier.csv`, `ft_scores_*.npy` |
| `gpu_power_log_nvidia.py` | Standalone GPU power logger (time trace to CSV) | `gpu_power_*.csv` |

## Setup

Requirements: Python 3.11 (3.8 works but caused dependency trouble), an NVIDIA GPU for measured
GPU energy, and [Ollama](https://ollama.com) installed and running.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull qwen2.5:7b          # Study 1 only
ollama pull llama3.1:8b
ollama pull nomic-embed-text    # used by score_similarity.py
```

For measurement validity, have Ollama load one model at a time:
`OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NUM_PARALLEL=1`.

## Running

```bash
cd src

# Study 1 (the benchmark run takes 2-4 hours; smoke test with PER_BIN=3 and RUNS=1 first)
python prep_prompts.py
python run_energy_benchmark_nvidia.py
python score_similarity.py
python analyze.py

# Study 2
python improved_train.py                 # add --no-embed to skip the bge embeddings
python check_finetuned.py
python router_cli.py                     # first run trains and saves router_model.joblib; :quit to exit
python error_analysis.py
python energy_benchmark.py --cpu-tdp 85 --embed
python measure_gen.py --n 100
python routing_experiment.py             # measured tier energies are already in its CONFIG block
```

`--cpu-tdp` is the CPU package power (watts) used to model CPU energy when the RAPL counter is not
readable without root. We used 85 W. Set it to your CPU's TDP and report it as an assumption.

If you re-measure tier energies or router cost, update the `TIERS` and `ROUTER_COST_J` values at the
top of `routing_experiment.py`.

## Hardware used

- NVIDIA RTX 2070 on an Ubuntu workstation with an Intel Xeon CPU (all reported results)

## Key references

- Ding et al. (2024). *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing.*
- Aggarwal et al. (2024). *AutoMix: Automatically Mixing Language Models.*
- Luccioni et al. *AI Energy Score.*
- Lawrence Berkeley National Laboratory. *United States Data Center Energy Usage Report.*

