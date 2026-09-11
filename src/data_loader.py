"""
Loads the Kaggle "Customer Support on Twitter" dataset (twcs.csv) and
reconstructs (customer_message -> brand_response) pairs for one brand.

Expected columns (as shipped by Kaggle):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

`inbound == True`  -> tweet was sent BY a customer TO a brand
`inbound == False` -> tweet was sent BY the brand (author_id is the brand handle)

We only need first-hop pairs (customer tweet -> the brand's direct reply)
for the grounding corpus; deeper multi-turn threads are out of scope for
a take-home-sized system (documented in the decision log).
"""

import os
import re
from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass
class ResolvedPair:
    customer_tweet_id: str
    customer_text: str
    brand_text: str


URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    """Light cleanup: strip urls/mentions, collapse whitespace.

    Deliberately NOT aggressive (no lowercasing/stopword removal) — the
    LLM works better with natural text, and we want replies to read
    naturally, not like bag-of-words output.
    """
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def load_raw(path: str = None) -> pd.DataFrame:
    """Load the raw CSV, pre-filtered to rows relevant to config.BRAND.

    Reads in chunks and keeps only brand tweets / tweets mentioning the
    brand, so we never hold the full multi-GB, 2.8M-row file in memory
    at once — important on lower-RAM machines.
    """
    path = path or config.RAW_DATA_PATH
    if not os.path.exists(path):
        candidates = [config.SAMPLE_DATA_PATH, os.path.join("hiver-agent", config.SAMPLE_DATA_PATH)]
        path = next((candidate for candidate in candidates if os.path.exists(candidate)), path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {config.RAW_DATA_PATH!r}. Download twcs.csv from Kaggle "
            "or provide RAW_DATA_PATH; no fallback sample is available."
        )

    needed_cols = ["tweet_id", "author_id", "inbound", "text",
                   "response_tweet_id", "in_response_to_tweet_id"]
    brand = config.BRAND

    chunks = []
    for chunk in pd.read_csv(path, dtype=str, usecols=needed_cols, chunksize=100_000):
        chunk["inbound"] = chunk["inbound"].astype(str).str.lower().isin(["true", "1"])
        is_brand_tweet = (~chunk["inbound"]) & (chunk["author_id"] == brand)
        mentions_brand = chunk["text"].fillna("").str.contains(f"@{brand}", case=False, regex=False)
        keep = chunk[is_brand_tweet | mentions_brand]
        if len(keep) > 0:
            chunks.append(keep)

    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=needed_cols)
    return df


def filter_brand(df: pd.DataFrame, brand: str = None) -> pd.DataFrame:
    """Keep only rows involving `brand`: the brand's own tweets, plus any
    customer tweet that is *addressed to* the brand (mentions @brand) or
    is a direct reply to one of the brand's tweets.

    Note: load_raw() already pre-filters to brand-tweets/@mentions for
    memory reasons, so this mainly re-applies the in_response_to_tweet_id
    check for replies-to-brand that didn't include an @mention.
    """
    brand = brand or config.BRAND
    brand_tweets = df[(~df["inbound"]) & (df["author_id"] == brand)]
    brand_tweet_ids = set(brand_tweets["tweet_id"])

    mentions_brand = df["text"].fillna("").str.contains(f"@{brand}", case=False, regex=False)
    replies_to_brand = df["in_response_to_tweet_id"].isin(brand_tweet_ids)

    customer_tweets = df[df["inbound"] & (mentions_brand | replies_to_brand)]
    return pd.concat([brand_tweets, customer_tweets]).drop_duplicates(subset="tweet_id")


def build_resolved_pairs(df: pd.DataFrame, brand: str = None) -> list[ResolvedPair]:
    """Reconstruct (customer message -> brand's reply) pairs.

    This is the grounding corpus: "how has this brand historically
    resolved similar issues" (deliverable #2's core requirement).
    """
    brand = brand or config.BRAND
    by_id = df.set_index("tweet_id", drop=False)

    pairs = []
    brand_replies = df[(~df["inbound"]) & (df["author_id"] == brand)]
    for _, brand_row in brand_replies.iterrows():
        parent_id = brand_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if not parent["inbound"]:
            continue  # brand replying to itself/another brand, skip
        cust_text = clean_text(parent["text"])
        brand_text = clean_text(brand_row["text"])
        if len(cust_text) < 5 or len(brand_text) < 5:
            continue
        pairs.append(ResolvedPair(
            customer_tweet_id=str(parent["tweet_id"]),
            customer_text=cust_text,
            brand_text=brand_text,
        ))
    return pairs


def load_pairs_for_brand(raw_path: str = None, brand: str = None) -> list[ResolvedPair]:
    df = load_raw(raw_path)
    df = filter_brand(df, brand)
    pairs = build_resolved_pairs(df, brand)
    if config.MAX_GROUNDING_PAIRS > 0 and len(pairs) > config.MAX_GROUNDING_PAIRS:
        # A fixed sample keeps local evaluation fast while retaining a broad
        # historical precedent pool; set MAX_GROUNDING_PAIRS=0 to disable.
        pairs = list(pd.Series(pairs).sample(
            n=config.MAX_GROUNDING_PAIRS, random_state=42
        ))
    return pairs