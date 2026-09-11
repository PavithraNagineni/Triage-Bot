"""
End-to-end pipeline: message -> classify -> ground+draft -> escalation
decision -> structured result.

This is the single entry point both `scripts/run_pipeline.py` and the
eval harness call, so "the pipeline I evaluate" and "the pipeline that
runs" are guaranteed to be the same code path.
"""

from dataclasses import dataclass, asdict

from .classifier import classify, ClassificationResult
from .escalation import decide, EscalationDecision
from .reply_drafter import draft_reply, DraftResult
from .retriever import GroundingRetriever


@dataclass
class PipelineResult:
    message: str
    classification: ClassificationResult
    draft: DraftResult
    escalation: EscalationDecision

    def to_dict(self) -> dict:
        d = {
            "message": self.message,
            "intent": self.classification.intent,
            "intent_confidence": self.classification.confidence,
            "intent_reason": self.classification.reason,
            "drafted_reply": self.draft.reply,
            "grounded": self.draft.grounded,
            "grounding_note": self.draft.grounding_note,
            "grounding_examples_used": len(self.draft.grounding_hits),
            "escalate": self.escalation.escalate,
            "escalation_reason": self.escalation.reason,
            "escalation_stage": self.escalation.stage,
        }
        return d


def run_pipeline(message: str, retriever: GroundingRetriever) -> PipelineResult:
    classification = classify(message)
    draft = draft_reply(message, retriever)
    escalation = decide(message, classification, draft)
    return PipelineResult(message=message, classification=classification,
                           draft=draft, escalation=escalation)
