"""
Central config for the support-agent pipeline.

Everything a grader (or future-you) needs to tweak lives here, not
scattered across modules. Change BRAND / INTENTS to re-target the
system at a different brand or taxonomy without touching pipeline code.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Brand & data
# ---------------------------------------------------------------------------
BRAND = os.environ.get("SUPPORT_BRAND", "AppleSupport")

# Path to the raw Kaggle "Customer Support on Twitter" CSV
# (twcs.csv from thoughtvector/customer-support-on-twitter).
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH", "data/twcs.csv")

# Fallback small sample shipped with the repo, used when RAW_DATA_PATH
# isn't present, so `run_pipeline.py` is runnable out of the box.
SAMPLE_DATA_PATH = "data/sample_data.csv"

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
INTENTS = [
    "account_access",       # login, password reset, locked account
    "billing_dispute",      # unexpected charge, refund request, subscription cost
    "delivery_order_issue", # late/missing/wrong order or shipment
    "product_bug_report",   # app crash, feature broken, error message
    "how_to_question",      # how do I do X with the product/service
    "general_complaint",    # venting / dissatisfaction, no clear actionable ask
    "praise_or_thanks",     # positive sentiment, no action needed
    "other",                # doesn't fit cleanly; catch-all
]

INTENT_DESCRIPTIONS = {
    "account_access": "Customer cannot log in, is locked out, or needs a password/account reset.",
    "billing_dispute": "Customer is disputing a charge, asking for a refund, or confused about billing/subscription cost.",
    "delivery_order_issue": "Customer's order/shipment is late, missing, damaged, or wrong.",
    "product_bug_report": "Customer reports the product/app/service is broken, crashing, or erroring.",
    "how_to_question": "Customer is asking how to do something with the product or service (no failure implied).",
    "general_complaint": "Customer expresses frustration or dissatisfaction without a specific actionable request.",
    "praise_or_thanks": "Customer is expressing satisfaction, gratitude, or praise. No action needed.",
    "other": "Doesn't clearly fit any other category.",
}

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
# "offline" uses deterministic, local heuristics and is the reproducible default
# for evaluation; set LLM_PROVIDER=anthropic/openai for the optional LLM path.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "offline")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
LLM_MODEL_OPENAI = os.environ.get("LLM_MODEL_OPENAI", "llama-3.1-8b-instant")
LLM_TEMPERATURE_CLASSIFY = 0.0
LLM_TEMPERATURE_DRAFT = 0.4
LLM_TEMPERATURE_JUDGE = 0.0

# ---------------------------------------------------------------------------
# Retrieval (grounding) settings
# ---------------------------------------------------------------------------
TOP_K_GROUNDING = 3
MIN_GROUNDING_SIMILARITY = 0.08
MAX_GROUNDING_PAIRS = int(os.environ.get("MAX_GROUNDING_PAIRS", "25000"))

# ---------------------------------------------------------------------------
# Escalation policy
# ---------------------------------------------------------------------------
ALWAYS_ESCALATE_INTENTS = {"billing_dispute"}
MIN_CLASSIFY_CONFIDENCE_FOR_AUTO = 0.6

ESCALATION_KEYWORDS = [
    "lawyer", "legal action", "sue", "fraud", "unauthorized charge",
    "self harm", "suicide", "kill myself", "emergency", "discriminat",
    "racist", "lawsuit", "sec filing", "data breach", "hacked",
]