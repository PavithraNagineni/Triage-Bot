# AppleSupport AI support agent — evaluation report

## 1. Problem framing

The system handles a narrow, auditable slice of public AppleSupport Twitter
traffic. “Good” means: the intent is correct, a reply is grounded in a
retrieved historical resolution, and high-risk or uncertain cases reach a
human with an explicit reason. It does not access private account/order
records, promise refunds, or attempt full multi-turn case management.

The brand-specific taxonomy is `account_access`, `billing_dispute`,
`delivery_order_issue`, `product_bug_report`, `how_to_question`,
`general_complaint`, `praise_or_thanks`, and `other`.

## 2. System and data

`data/twcs.csv` is streamed in chunks, filtered to AppleSupport, and converted
to direct customer-to-brand reply pairs. A TF-IDF retriever supplies up to
three precedents. Classification, drafting, and escalation are structured JSON
steps. The checked-in golden set has 196 hand-labeled examples, sampled with a
fixed seed after proxy-intent stratification.

The reproducible headline path is **offline mode**, which uses deterministic
local heuristics and the same retriever/pipeline interfaces. Hosted Anthropic,
OpenAI, and Groq-compatible providers are optional and must not be confused
with the offline numbers.

## 3. Results and baselines

Run:

```powershell
$env:LLM_PROVIDER="offline"
python eval/eval_harness.py --golden data/golden_set.csv --skip-judge --output eval/eval_results.csv
```

On the checked-in 196-row set, the offline pipeline scored **0.658 accuracy /
0.604 macro-F1**, versus **0.316 / 0.060** for the majority baseline and
**0.622 / 0.553** for keyword rules. Escalation agreement was **0.679
accuracy, 0.357 precision, and 0.769 recall**. These are measured offline
numbers, not a hosted-LLM claim. The generated CSV is the source of truth; do
not copy numbers from a hosted run after a rate-limit error.

Reply quality uses a four-dimension 1–5 rubric. The offline judge mean was
4.0 on every dimension. On the 25-row human calibration sample, exact
agreement was **32%**, agreement within one point was **84%**, and mean
absolute error was **0.88**. This is useful but limited evidence: the sample
is small and the exact-match rate shows that the automated judge should not be
treated as a replacement for human review.

## 4. Failure analysis

1. **Overlapping intents:** battery, Wi-Fi, and update complaints can be both
   access/configuration and product bugs. Hypothesis: hierarchical labeling or
   multi-label annotations would reduce forced errors.
2. **Very short tweets:** “help”, screenshots, and pronouns have insufficient
   lexical evidence. Hypothesis: retrieve thread context and escalate these.
3. **Paraphrase retrieval misses:** TF-IDF favors shared words and can retrieve
   the wrong precedent for semantically similar wording. Hypothesis: compare
   sentence embeddings against TF-IDF on a held-out set.
4. **Historical replies are often generic:** a retrieved reply may say “DM us”
   without resolving the issue. Hypothesis: score resolution completeness and
   exclude non-resolution precedents.
5. **Public-data limits:** billing and order cases need private identifiers.
   Hypothesis: collect redacted internal resolution outcomes and keep public
   tweets as intake only.

## 5. What is misleading about my headline number?

The headline classification score is not production accuracy. The golden set
is small, stratified, and derived from the same brand corpus used for
retrieval. The offline provider is a transparent heuristic baseline, while a
hosted LLM run has different cost, latency, and failure modes. Labels are
single-intent even when a tweet is genuinely multi-intent. Finally, judge
agreement is based on only 25 human-scored examples and should not be
generalized beyond this calibration sample.

## 6. One more week

I would (1) reconstruct full threads with attribution tests, (2) add a
time-split held-out evaluation to reduce retrieval leakage, (3) compare TF-IDF
with a local sentence encoder, (4) collect 30 independently scored judge
examples and report inter-rater agreement, and (5) add privacy-safe tool
calls for order/account lookup with mandatory human approval.
