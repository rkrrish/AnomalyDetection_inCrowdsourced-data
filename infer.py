"""infer.py - Standalone Inference CLI for Crowdsourced Data Validation.

Supports evaluating a single submission JSON or batch validating a CSV dataset
using the new CatBoost anomaly detection and trust scoring system.

Usage Examples:
    # Single entry JSON string:
    & "python.exe" infer.py --data '{"company": "Google", "jobFamily": "Software Engineer", "level": "L5", "baseSalary": 185000, "baseSalaryCurrency": "USD", "totalCompensation": 280000, "yearsOfExperience": 6, "location": "Mountain View, CA"}'

    # Single entry file:
    & "python.exe" infer.py --file data/sample.json

    # Batch CSV processing:
    & "python.exe" infer.py --batch data/submissions.csv --output data/outputs/batch_results.csv --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import pandas as pd

from src.pipeline import ValidationPipeline


def print_single_report(res: Dict[str, Any]) -> None:
    """Format and print an executive validation report to the console."""
    classification = res.get("classification", "Review")
    color_code = {
        "Approve": "\033[92m",  # Green
        "Review": "\033[93m",   # Yellow
        "Reject": "\033[91m",   # Red
    }.get(classification, "\033[0m")
    reset_code = "\033[0m"

    print("\n" + "=" * 65)
    print(f" CROWDSOURCED DATA VALIDATION REPORT  [{res.get('submissionId', '')[:8]}]")
    print("=" * 65)
    print(f" Company:          {res.get('company', '')}")
    print(f" Role / Level:     {res.get('jobRole', '')} ({res.get('level', '')})")
    print(f" Location:         {res.get('location', '')}")
    print(f" Base Salary:      {res.get('baseSalary', 0):,} {res.get('baseSalaryCurrency', 'USD')}")
    print(f" Total Comp:       {res.get('totalCompensation', 0):,} {res.get('baseSalaryCurrency', 'USD')}")
    print(f" Experience:       {res.get('yearsOfExperience', 0)} years")
    print("-" * 65)
    print(f" Final Decision:   {color_code}{classification.upper()}{reset_code}")
    print(f" Quality Score:    {res.get('qualityScore', 0):.1f} / 100")
    print(f" Confidence Level: {res.get('confidenceScore', 0):.1f} %")
    print(f" CatBoost Anomaly: {res.get('p_anomaly', 0):.1%}")
    print(f" User Trust Score: {res.get('user_trust', 1.0):.2f}")

    if res.get("peer_median_usd"):
        print(f" Peer Median:      ${res.get('peer_median_usd', 0):,.0f} USD (Peer Z-Score: {res.get('peer_z_score', 0):+.2f})")

    flags = res.get("flags", {})
    if flags:
        print("\n Triggered Logical Flags:")
        for k in flags.keys():
            print(f"   • {k}")

    reasons = res.get("reasons", [])
    if reasons:
        print("\n Actionable Explanations:")
        for r in reasons:
            print(f"   • {r}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CatBoost Crowdsourced Data Validation Inference")
    parser.add_argument("--data", type=str, help="Single submission JSON string")
    parser.add_argument("--file", type=str, help="Path to single submission JSON file")
    parser.add_argument("--batch", type=str, help="Path to batch CSV file")
    parser.add_argument("--output", type=str, default="data/outputs/inference_results.csv", help="Output path for batch results")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process in batch mode")
    parser.add_argument("--trust", type=float, default=None, help="Initial user trust score (0.0 - 1.0)")
    parser.add_argument("--save", action="store_true", help="Persist validated submission to database")

    args = parser.parse_args()
    pipeline = ValidationPipeline()

    if args.data:
        try:
            entry = json.loads(args.data)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON data string: {e}")
            sys.exit(1)
        res = pipeline.evaluate_submission(entry, initial_trust=args.trust, persist=args.save)
        print_single_report(res)

    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            entry = json.load(f)
        res = pipeline.evaluate_submission(entry, initial_trust=args.trust, persist=args.save)
        print_single_report(res)

    elif args.batch:
        if not os.path.exists(args.batch):
            print(f"Error: batch file not found: {args.batch}")
            sys.exit(1)
        print(f"Loading batch dataset from {args.batch}...")
        df = pd.read_csv(args.batch, nrows=args.limit)
        print(f"Evaluating {len(df):,} submissions with CatBoost pipeline...")
        results_df = pipeline.evaluate_batch(df)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"Batch evaluation complete! Saved results to {args.output}")

        dec_counts = results_df["decision"].value_counts()
        print("\nDecision Summary:")
        for dec, count in dec_counts.items():
            pct = count / len(results_df) * 100
            print(f"  {dec:<10}: {count:>6} ({pct:.1f}%)")

    else:
        # Default demo if run with no args
        print("No input specified. Running test demonstration with a sample submission...")
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample.json")
        if os.path.exists(sample_path):
            with open(sample_path, "r", encoding="utf-8") as f:
                demo_entry = json.load(f)
        else:
            demo_entry = {
                "company": "Microsoft",
                "title": "Software Engineer",
                "jobFamily": "Software Engineer",
                "level": "60",
                "location": "New York, NY",
                "locationSlug": "new-york-usa",
                "baseSalary": 158600,
                "baseSalaryCurrency": "USD",
                "totalCompensation": 235300.0,
                "firstYearTotalCompensation": 233100,
                "avgAnnualStockGrantValue": 55500.0,
                "firstYearStockGrantValue": 55500.0,
                "totalStockGrantValue": 222000.0,
                "avgAnnualBonusValue": 21200.0,
                "firstYearBonusValue": 19000,
                "bonusCurrency": "USD",
                "stockGrantCurrency": "USD",
                "yearsOfExperience": 3,
                "yearsAtCompany": 0,
                "workArrangement": "hybrid",
                "stockType": "stock",
                "compPerspective": "offer"
            }
        res = pipeline.evaluate_submission(demo_entry, persist=False)
        print_single_report(res)


if __name__ == "__main__":
    main()
