"""
Run the full agent pipeline over a sample of customer messages for the
configured brand and write results to a CSV.

Usage:
    python scripts/run_pipeline.py --n 30
    python scripts/run_pipeline.py --input data/golden_set.csv --text-col message
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

from src import config
from src.data_loader import load_pairs_for_brand
from src.pipeline import run_pipeline
from src.retriever import GroundingRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                     help="CSV with a text column of messages to run the pipeline on. "
                          "If omitted, samples N customer messages from the raw dataset.")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--n", type=int, default=20, help="How many messages to sample if --input not given")
    ap.add_argument("--output", default="results.csv")
    args = ap.parse_args()

    print(f"[1/4] Loading resolved pairs for brand '{config.BRAND}' (grounding corpus)...")
    pairs = load_pairs_for_brand()
    print(f"      -> {len(pairs)} resolved (customer, brand-reply) pairs loaded.")
    retriever = GroundingRetriever(pairs)

    if args.input:
        df = pd.read_csv(args.input)
        messages = df[args.text_col].dropna().astype(str).tolist()
        print(f"[2/4] Loaded {len(messages)} messages from {args.input}")
    else:
        import random
        random.seed(42)
        messages = [p.customer_text for p in pairs]
        random.shuffle(messages)
        messages = messages[: args.n]
        print(f"[2/4] Sampled {len(messages)} customer messages from the grounding corpus")

    print(f"[3/4] Running pipeline (classify -> ground -> draft -> escalation)...")
    rows = []
    for msg in tqdm(messages):
        try:
            result = run_pipeline(msg, retriever)
            rows.append(result.to_dict())
        except Exception as e:  # noqa: BLE001
            rows.append({"message": msg, "error": str(e)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"[4/4] Wrote {len(out_df)} results to {args.output}")

    if "escalate" in out_df.columns:
        rate = out_df["escalate"].mean()
        print(f"      Escalation rate: {rate:.1%}")
    if "intent" in out_df.columns:
        print("      Intent distribution:")
        print(out_df["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
