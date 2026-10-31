# main.py
"""Main Entrypoint for Crowdsourced Data Validation Pipeline."""

import os
import pandas as pd

from src.config import DATA_PATH
from src.pipeline import run_pipeline


if __name__ == "__main__":
    print("=" * 60)
    print("CROWDSOURCED DATA VALIDATION SYSTEM (CATBOOST PIPELINE)")
    print(f"Loading data from: {DATA_PATH}")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Error: dataset file not found at {DATA_PATH}")
        exit(1)

    if DATA_PATH.endswith(".csv"):
        df = pd.read_csv(DATA_PATH, nrows=50)
    else:
        import json
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)

    print(f"Loaded {len(df):,} sample records ({len(df.columns)} columns)")

    results_df = run_pipeline(df)

    print("\nSample Validation Results:")
    cols_to_show = ["company", "jobFamily", "level", "baseSalary", "qualityScore", "classification", "p_anomaly", "user_trust"]
    avail_cols = [c for c in cols_to_show if c in results_df.columns]
    print(results_df[avail_cols].head(10).to_string(index=False))

    print("\nDecision Distribution:")
    print(results_df["decision"].value_counts().to_string())

    print("\nPipeline execution finished successfully.")