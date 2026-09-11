"""
Auto-handle vs. escalate-to-human decision.

Deliberately a HYBRID of cheap deterministic rules and an LLM judgment
call, not a pure LLM decision:

  1. Keyword/intent-based rules run first and can force escalation
     regardless of what the LLM thinks. These are auditable, cheap, and
     don't depend on the LLM being well-calibrated about high-stakes
     situations (legal threats, safety, billing disputes involving money).
  2. Classifier confidence is checked next — low-confidence intents
     escalate rather than let a shaky classification drive an
     auto-reply.
  3. Only if neither rule fires does the LLM get a genuine judgment
     call, informed by whether the drafted reply was actually grounded
     in precedent.

Every path produces a human-readable `reason` string, per the
assignment's "with a stated reason" requirement.
"""

from dataclasses import dataclass

from . import config
from .classifier import ClassificationResult
from .llm_client import complete_json
from .reply_drafter import DraftResult

SYSTEM_PROMPT = f"""You decide whether a drafted customer-support reply for {config.BRAND} is \
safe to auto-send, or should be escalated to a human agent instead. Be conservative: when in \
doubt, escalate. Consider: is the issue emotionally charged, high-stakes, ambiguous, or does \
the drafted reply overcommit to something not clearly grounded in precedent?

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"escalate": <true or false>, "reason": "<one short sentence, specific to this message>"}}
"""


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str
    stage: str  # "keyword_rule" | "confidence_rule" | "intent_rule" | "llm_judgment"


def _keyword_hit(message: str) -> str | None:
    low = message.lower()
    for kw in config.ESCALATION_KEYWORDS:
        if kw in low:
            return kw
    return None


def decide(message: str, classification: ClassificationResult,
           draft: DraftResult) -> EscalationDecision:
    kw = _keyword_hit(message)
    if kw:
        return EscalationDecision(
            escalate=True,
            reason=f"Message contains high-risk keyword ('{kw}') — routed to human regardless of content.",
            stage="keyword_rule",
        )

    if classification.intent in config.ALWAYS_ESCALATE_INTENTS:
        return EscalationDecision(
            escalate=True,
            reason=f"Intent '{classification.intent}' is policy-flagged to always involve a human (money/refund at stake).",
            stage="intent_rule",
        )

    if classification.confidence < config.MIN_CLASSIFY_CONFIDENCE_FOR_AUTO:
        return EscalationDecision(
            escalate=True,
            reason=f"Classifier confidence {classification.confidence:.2f} below auto-handle threshold "
                   f"({config.MIN_CLASSIFY_CONFIDENCE_FOR_AUTO}); routed to human to avoid acting on a shaky read.",
            stage="confidence_rule",
        )

    if not draft.grounded:
        return EscalationDecision(
            escalate=True,
            reason="Drafted reply is not clearly grounded in a past resolved case for this brand; "
                   "escalating rather than risk an unsupported resolution.",
            stage="confidence_rule",
        )

    # Genuine judgment call left to the LLM.
    user_prompt = (
        f"Customer message:\n{message}\n\n"
        f"Classified intent: {classification.intent} (confidence {classification.confidence:.2f})\n\n"
        f"Drafted reply:\n{draft.reply}\n\n"
        f"Grounding note: {draft.grounding_note}"
    )
    result = complete_json(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=config.LLM_TEMPERATURE_JUDGE,
        max_tokens=500,
    )
    return EscalationDecision(
        escalate=bool(result.get("escalate", True)),
        reason=str(result.get("reason", "")).strip() or "LLM judgment call, no reason returned.",
        stage="llm_judgment",
    )
