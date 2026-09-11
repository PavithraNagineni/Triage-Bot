"""
Samples candidate messages for you to hand-label into the golden
evaluation set (deliverable #2).

This script does NOT label anything — it produces a sampling pool with
a template for the columns you fill in by hand. The point is to make
the *sampling* reproducible and documented (stratify by a cheap proxy
for intent) while the *labeling* stays human.

Sampling method (documented here so it can be copy-pasted into the
report's "how you sampled and labelled" note):
  1. Load all customer messages addressed to the brand.
  2. Run the (fast, cheap) keyword-based pre-tagger below to bucket
     messages into a rough proxy-intent — NOT the final label, just a
     stratification key so rare categories aren't drowned out.
  3. Sample N_PER_BUCKET messages per bucket (with replacement skipped,
     capped by bucket size), for a target total of ~200.
  4. Shuffle once with a fixed seed for reproducibility.
You then hand-label each row's `true_intent`, `should_escalate`, and
`human_reply_quality_notes` columns yourself.

Usage:
    python scripts/build_golden_set.py --total 200 --output data/golden_set_template.csv
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import config
from src.data_loader import load_pairs_for_brand

# Cheap keyword proxy tags used ONLY to stratify sampling so rare
# intents aren't drowned out by common ones. This is intentionally
# crude — it is not the classifier and not the golden label.
PROXY_RULES = {
    "account_access": r"\b(log ?in|password|locked|reset|can'?t access|sign ?in)\b",
    "billing_dispute": r"\b(charge|refund|billing|subscription|cancel|money)\b",
    "delivery_order_issue": r"\b(order|delivery|shipment|package|arrived|tracking)\b",
    "product_bug_report": r"\b(crash|bug|error|broken|not working|freez)\b",
    "how_to_question": r"\b(how do i|how to|where is|can i)\b",
    "praise_or_thanks": r"\b(thank|thanks|great|love|awesome)\b",
}


def proxy_tag(text: str) -> str:
    low = text.lower()
    for tag, pattern in PROXY_RULES.items():
        if re.search(pattern, low):
            return tag
    return "other_or_general_complaint"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--output", default="data/golden_set_template.csv")
    args = ap.parse_args()

    pairs = load_pairs_for_brand()
    df = pd.DataFrame({"message": [p.customer_text for p in pairs]}).drop_duplicates()
    df["proxy_tag"] = df["message"].apply(proxy_tag)

    n_buckets = df["proxy_tag"].nunique()
    per_bucket = max(1, args.total // n_buckets)

    # NOTE: not using groupby().apply() here — recent pandas versions drop the
    # grouping column from the result by default, which silently loses proxy_tag.
    # Explicit loop keeps this both correct and easy to read.
    chunks = []
    for tag, group in df.groupby("proxy_tag"):
        chunks.append(group.sample(min(len(group), per_bucket), random_state=42))
    sampled = pd.concat(chunks).sample(frac=1, random_state=42).reset_index(drop=True)

    # Columns for hand-labeling. Left blank on purpose.
    sampled["true_intent"] = ""
    sampled["should_escalate"] = ""       # TRUE/FALSE, your judgment
    sampled["escalate_reason_gold"] = ""  # brief note, for eval harness comparison
    sampled["human_reply_quality_notes"] = ""  # free text, filled in after running pipeline

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sampled.to_csv(args.output, index=False)
    print(f"Wrote {len(sampled)} candidate rows to {args.output}")
    print("Proxy-tag distribution (stratification key, not final labels):")
    print(sampled["proxy_tag"].value_counts().to_string())
    print("\nNext: open the CSV and hand-fill true_intent / should_escalate / "
          "escalate_reason_gold for each row. That becomes your golden set.")


if __name__ == "__main__":
    main()
