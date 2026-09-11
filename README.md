# Hiver Take-Home — AI Support Agent for AppleSupport (Twitter)

An AI customer-support agent that: (1) classifies incoming messages into a
brand-specific intent taxonomy, (2) drafts a reply grounded in how the brand
has historically resolved similar issues, and (3) decides auto-handle vs.
escalate-to-human with a stated reason.

Built against the Kaggle **Customer Support on Twitter** dataset
(`thoughtvector/customer-support-on-twitter`), targeting the `AppleSupport`
brand handle by default (edit `SUPPORT_BRAND` in `src/config.py` to retarget).

## What's here (the LLM/agent side — this repo covers deliverables #1–#3)

```
src/
  config.py          intent taxonomy, brand, escalation policy — edit here
  data_loader.py      parses twcs.csv, reconstructs (customer -> brand reply) pairs
  retriever.py         TF-IDF grounding retriever over resolved pairs
  classifier.py        LLM intent classifier (prompted, JSON output)
  reply_drafter.py     grounded reply drafter (RAG-lite over retriever hits)
  escalation.py        rule + LLM hybrid auto-handle/escalate decision
  pipeline.py           orchestrates the above into one PipelineResult
  llm_client.py         provider-agnostic LLM call wrapper (Anthropic default, OpenAI supported)
scripts/
  run_pipeline.py       CLI: run the agent over sampled/given messages -> results.csv
  build_golden_set.py    samples a stratified pool for you to hand-label into a golden set
eval/
  eval_harness.py        classification metrics + baselines, escalation agreement, LLM-as-judge
data/
  sample_data.csv         tiny offline sample (20 pairs) so the repo runs with zero setup
```

The repository also includes `REPORT.md`, `DECISION_LOG.md`, the checked-in
golden labels, and the evaluation rubric. The only intentionally manual step
left is human scoring of a small reply-quality calibration sample, because an
AI-generated score cannot honestly be called a human judgment.

## Setup

```bash
pip install -r requirements.txt
# Reproducible offline mode (default; no key or network required)
$env:LLM_PROVIDER="offline"           # PowerShell
# Optional hosted mode:
$env:ANTHROPIC_API_KEY="sk-..."        # or OPENAI_API_KEY + LLM_PROVIDER=openai
```

Download `twcs.csv` from Kaggle and place it at `data/twcs.csv`. **If you skip
this step, everything below still runs** against the bundled 20-pair
`data/sample_data.csv` — useful for a fast sanity check, not for real results.

## Reproduce the headline results (< 15 minutes)

```bash
# 1. Run the agent over a random sample of real messages (smoke test)
python scripts/run_pipeline.py --n 30 --output results_sample.csv

# 2. Build a stratified pool to hand-label into your golden set
python scripts/build_golden_set.py --total 200 --output data/golden_set_template.csv
# -> open data/golden_set_template.csv, fill in true_intent / should_escalate /
#    escalate_reason_gold by hand for each row, save as data/golden_set.csv

# 3. Run the evaluation harness against your labeled golden set
python eval/eval_harness.py --golden data/golden_set.csv --output eval/eval_results.csv
```

Step 3 prints: classification accuracy/macro-F1 for the pipeline vs. a
**trivial baseline** (majority-class intent) and a **simple baseline**
(hand-written keyword rules), escalation agreement (accuracy/precision/recall)
against your labels, and LLM-as-judge reply-quality scores. If you add a
`human_judge_score` column for a subset of golden-set rows (score the drafted
replies yourself 1-5 after step 3's first pass), re-run step 3 and it will
also report judge-vs-human agreement — required for deliverable #3.

## Design choices worth knowing before you read the code

- **Retrieval is TF-IDF, not embeddings.** Fully offline, no model download,
  fast at this scale. Grounding quality depends on the retrieved precedent
  being lexically similar, which is a real limitation — call this out in
  your failure analysis if paraphrased issues aren't retrieved well.
- **Escalation is a rule+LLM hybrid, not pure LLM.** Keyword triggers and
  policy-flagged intents (billing disputes) escalate deterministically before
  the LLM ever gets a say — this is intentional and auditable, not a
  cost-saving shortcut. See `src/escalation.py`'s docstring.
- **The classifier is prompted, not fine-tuned.** Faster to iterate at
  take-home scale; a fine-tuned classifier is a reasonable "what I'd do with
  one more week" item.
- **Every LLM call returns strict JSON** (`llm_client.complete_json`) so the
  pipeline never depends on parsing free text.

## Reproducibility and honest evaluation notes

The checked-in `data/golden_set.csv` contains 196 hand-labeled examples. The
default offline provider is deterministic and exists so a reviewer can
reproduce the pipeline and metrics without an API key or a rate-limit failure.
It is a heuristic baseline, not the claimed production-quality LLM. Hosted
providers remain supported by setting `LLM_PROVIDER` and the corresponding key.
The grounding corpus is capped at 25,000 fixed-seed pairs by default for a
sub-15-minute local evaluation; set `MAX_GROUNDING_PAIRS=0` for the full corpus.
For the required judge-vs-human calibration, fill `human_judge_score` for the
25 rows in `data/human_judge_template.csv`, then run the command in
[eval/rubric.md](eval/rubric.md). The harness reports exact match, within-one
agreement, and MAE only when those scores exist. It never fabricates human
judgments.

See [REPORT.md](REPORT.md) for the problem framing, baselines, measured
results, failure analysis, misleading-number section, and next steps.
See [DECISION_LOG.md](DECISION_LOG.md) for the non-obvious implementation
decisions.

## Known gaps (be upfront about these in your report)

- `data_loader.py` only reconstructs first-hop (customer -> direct brand
  reply) pairs, not full multi-turn threads — documented, not hidden.
- The intent taxonomy in `config.py` is a reasonable starting point but you
  should re-derive it from ~100 real threads for your actual chosen brand
  before trusting it (the assignment explicitly wants taxonomy from data).
- No cost/latency logging is wired in yet — add a token-usage counter to
  `llm_client.py` if your report wants to discuss cost per ticket.
