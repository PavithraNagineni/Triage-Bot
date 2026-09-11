"""
Minimal LLM client wrapper.

The assignment allows "any LLM API or open model" — this module isolates
that choice behind one function (`complete`) so swapping providers means
editing this file only, not the classifier/drafter/judge code.

Default: Anthropic (Claude). Set LLM_PROVIDER=openai + OPENAI_API_KEY to
switch without touching anything else. Groq is routed through the same
OpenAI-compatible client, using GROQ_API_KEY and a Groq base_url.
"""

import json
import os
import time

from . import config


class LLMError(RuntimeError):
    pass


def _complete_anthropic(system: str, user: str, temperature: float, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        extra_body={"temperature": temperature},
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _complete_openai(system: str, user: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        extra = {"reasoning_effort": "low"}
    else:
        client = OpenAI()  # reads OPENAI_API_KEY from env
        extra = {}

    resp = client.chat.completions.create(
        model=config.LLM_MODEL_OPENAI,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_body=extra,
    )
    return resp.choices[0].message.content


def _complete_offline(system: str, user: str, temperature: float, max_tokens: int) -> str:
    """Return deterministic JSON for reproducible runs without an API key.

    This is intentionally a conservative baseline, not a claim that heuristics
    replace an LLM. It makes the complete pipeline runnable in CI and lets the
    evaluation separate model quality from provider availability.
    """
    import re

    text = user.lower()
    intents = {
        "billing_dispute": r"\b(charge|charged|refund|billing|subscription|payment|money|price|cost)\b",
        "account_access": r"\b(log ?in|sign ?in|password|locked|access|account|wifi password)\b",
        "delivery_order_issue": r"\b(order|delivery|shipment|package|tracking|pre[- ]?order|paypal)\b",
        "product_bug_report": r"\b(crash|bug|error|broken|freeze|freez|slow|battery|not working|issue|problem)\b",
        "how_to_question": r"\bhow (do|can|to)\b|\bwhere (is|can)\b",
        "praise_or_thanks": r"\b(thank|thanks|great|love|awesome|appreciate)\b",
    }
    if "judge" in system.lower() or "grading" in system.lower():
        reply = user.split("Drafted reply:\n", 1)[-1].split("\n\nGrounding note", 1)[0].strip()
        grounded = "no strong precedent" not in user.lower() and bool(reply)
        score = 4 if grounded and len(reply) >= 20 else 2 if not reply else 3
        return json.dumps({
            "relevance": score, "groundedness": score, "tone": score,
            "actionability": score, "overall": score,
            "justification": "Deterministic offline rubric; API judge not called.",
        })
    if "drafting" in system.lower():
        examples = user.split("Historical resolved examples", 1)[-1]
        has_example = "(no similar historical examples found)" not in examples.lower()
        reply = ("Thanks for reaching out. We have seen similar reports and recommend "
                 "checking the latest software update and restarting the device. "
                 "If the issue continues, please contact Apple Support with your device "
                 "model and diagnostic details.")
        return json.dumps({
            "reply": reply,
            "grounded": has_example,
            "grounding_note": "Used a retrieved historical support example." if has_example else "no strong precedent",
        })
    if "escalat" in system.lower():
        high_risk = bool(re.search(r"\b(lawyer|legal|sue|fraud|emergency|hacked|data breach|suicide)\b", text))
        return json.dumps({
            "escalate": high_risk,
            "reason": "Offline policy rule detected a high-risk term." if high_risk else "No high-risk term detected by offline policy.",
        })
    for intent, pattern in intents.items():
        if re.search(pattern, text):
            return json.dumps({"intent": intent, "confidence": 0.78, "reason": "Matched a deterministic taxonomy cue."})
    return json.dumps({"intent": "other", "confidence": 0.35, "reason": "No deterministic taxonomy cue matched."})


def complete(system: str, user: str, temperature: float = 0.0, max_tokens: int = 800,
             retries: int = 3) -> str:
    """Call the configured LLM provider with basic retry on transient errors."""
    if config.LLM_PROVIDER == "offline":
        fn = _complete_offline
    else:
        fn = _complete_anthropic if config.LLM_PROVIDER == "anthropic" else _complete_openai
    last_err = None
    for attempt in range(retries):
        try:
            return fn(system, user, temperature, max_tokens)
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a retry boundary
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(f"LLM call failed after {retries} attempts: {last_err}")


def complete_json(system: str, user: str, temperature: float = 0.0, max_tokens: int = 800) -> dict:
    """Call the LLM and parse a JSON object from its response.

    Strips markdown code fences defensively since models sometimes wrap
    JSON in ```json ... ``` even when told not to.
    """
    raw = complete(system, user, temperature=temperature, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMError(f"Could not parse JSON from LLM response: {raw[:500]!r}") from e