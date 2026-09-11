"""
Intent classification: prompted LLM classifier over the fixed taxonomy
in config.INTENTS.

Why prompted rather than fine-tuned: at take-home scale (a few hundred
to a few thousand labeled examples, 2-day timeline) a well-specified
few-shot prompt on a strong LLM will out-perform a from-scratch
fine-tune and is far cheaper to iterate on. This is called out
explicitly in the report's "what I chose not to build" section — a
fine-tuned DistilBERT classifier (cf. the BERT/LoRA sentiment work) is
the natural next step if latency/cost at scale becomes the constraint.
"""

from dataclasses import dataclass

from . import config
from .llm_client import complete_json

SYSTEM_PROMPT = f"""You are an intent classifier for {config.BRAND}'s customer support \
messages. Classify the customer's message into exactly one of the intents below.

Intents:
{chr(10).join(f"- {name}: {desc}" for name, desc in config.INTENT_DESCRIPTIONS.items())}

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"intent": "<one of the intent names above, exactly as written>",
  "confidence": <float 0.0-1.0, your genuine confidence in this label>,
  "reason": "<one short sentence>"}}
"""


@dataclass
class ClassificationResult:
    intent: str
    confidence: float
    reason: str


def classify(message: str) -> ClassificationResult:
    result = complete_json(
        system=SYSTEM_PROMPT,
        user=f"Customer message:\n{message}",
        temperature=config.LLM_TEMPERATURE_CLASSIFY,
        max_tokens=600,
    )
    intent = result.get("intent", "other")
    if intent not in config.INTENTS:
        intent = "other"
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        reason=str(result.get("reason", "")).strip(),
    )
