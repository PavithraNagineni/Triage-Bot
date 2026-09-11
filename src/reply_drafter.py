"""
Drafts a reply grounded in how the brand has historically resolved
similar issues (deliverable #2's core requirement — "grounded", not
just "plausible-sounding").

Approach: retrieve top-k similar past (customer -> brand reply) pairs
via the TF-IDF retriever, show them to the LLM as worked examples of
this brand's tone/resolution pattern, and require the model to draft a
reply consistent with that pattern for the new message. If no
precedent clears the similarity floor, the model is told explicitly so
it doesn't fabricate false confidence in a "here's exactly how we
always fix this" tone.
"""

from dataclasses import dataclass

from . import config
from .llm_client import complete_json
from .retriever import GroundingRetriever, GroundingHit

SYSTEM_PROMPT = f"""You are drafting a customer support reply on behalf of {config.BRAND}, \
in the brand's own voice (as shown by the historical examples below). Ground your reply in \
how this brand has actually resolved similar issues before — don't invent policies or \
promises the examples don't support. If the examples don't clearly cover this case, say so \
in your reply by keeping it general (acknowledge + ask a clarifying question / point to \
official support) rather than guessing at a specific resolution.

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"reply": "<the drafted reply text, ready to send>",
  "grounded": <true if the reply's resolution is directly supported by the examples, else false>,
  "grounding_note": "<one short sentence on what precedent you used, or 'no strong precedent'>"}}
"""


@dataclass
class DraftResult:
    reply: str
    grounded: bool
    grounding_note: str
    grounding_hits: list[GroundingHit]


def _format_examples(hits: list[GroundingHit]) -> str:
    if not hits:
        return "(no similar historical examples found)"
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(
            f"Example {i} (similarity={hit.similarity:.2f}):\n"
            f"  Customer: {hit.pair.customer_text}\n"
            f"  {config.BRAND} reply: {hit.pair.brand_text}"
        )
    return "\n\n".join(blocks)


def draft_reply(message: str, retriever: GroundingRetriever,
                 top_k: int = None) -> DraftResult:
    top_k = top_k or config.TOP_K_GROUNDING
    hits = retriever.query(message, top_k=top_k)
    strong_hits = [h for h in hits if h.similarity >= config.MIN_GROUNDING_SIMILARITY]

    user_prompt = (
        f"Historical resolved examples for this brand:\n\n{_format_examples(strong_hits)}\n\n"
        f"New customer message to reply to:\n{message}"
    )
    result = complete_json(
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=config.LLM_TEMPERATURE_DRAFT,
        max_tokens=900,
    )
    return DraftResult(
        reply=str(result.get("reply", "")).strip(),
        grounded=bool(result.get("grounded", False)) and len(strong_hits) > 0,
        grounding_note=str(result.get("grounding_note", "")).strip(),
        grounding_hits=strong_hits,
    )
