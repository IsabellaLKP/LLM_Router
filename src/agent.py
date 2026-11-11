import os
import time
import csv
import requests
from router_cli import load_or_build, score

MODEL_PATH = "router_model.joblib"
OLLAMA_URL = "http://localhost:11434/api/generate"
CSV_LOG_PATH = "routing_benchmark_results.csv"

SMALL_MODEL  = "qwen2.5:0.5b"  # Class 0: Easy
MEDIUM_MODEL = "llama3.2:3b"  # Class 1: Medium
LARGE_MODEL  = "llama3.1:8b"    # Class 2: Hard

def route_and_generate(bundle, prompt: str):
    d_score, c5, c3, feats = score(bundle, prompt)
    
    # 3-tier Mapping
    if c3 == 0:
        selected_model = SMALL_MODEL
        label = "Easy"
    elif c3 == 1:
        selected_model = MEDIUM_MODEL
        label = "Medium"
    else:  # c3 == 2
        selected_model = LARGE_MODEL
        label = "Hard"
        
    print(f"Difficulty Score : {d_score:+.3f}")
    print(f"Difficulty Class : {label} (c3 = {c3})")
    print(f"Assigned Model   : {selected_model}")