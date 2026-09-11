# Triage-Bot: AppleSupport AI Support Agent

Triage-Bot is a small, reproducible customer-support agent built for the
Hiver SDE take-home assignment. It uses the **Customer Support on Twitter**
dataset and targets the `AppleSupport` handle.

Given one incoming customer message, it:

1. assigns one of eight support intents;
2. retrieves similar historical AppleSupport conversations;
3. drafts a reply based on those past resolutions; and
4. decides whether to auto-handle the message or escalate it to a human,
   including an explanation.

The project is designed to be inspectable rather than magical: every decision
is returned as structured data, the baselines are included, and the limitations
are reported honestly.

## Results at a glance

The checked-in golden set contains **196 hand-labeled examples**.

| System | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline | 31.6% | 6.0% |
| Keyword-rule baseline | 62.2% | 55.3% |
| Triage-Bot offline pipeline | **65.8%** | **60.4%** |

Escalation agreement with human labels was:

- Accuracy: **67.9%**
- Precision: **35.7%**
- Recall: **76.9%**

For reply quality, an automated judge scored 196 replies at 4.0/5. On a
separate 25-example human calibration sample, judge-human agreement was **32%
exact**, **84% within one point**, with **0.88 mean absolute error**. The low
exact-match rate is an important limitation, not something hidden.

See [REPORT.md](REPORT.md) for the full report and failure analysis.
The same report is available as a formatted Word document:
[REPORT.docx](REPORT.docx).

## Quick start: reproduce the evaluation

These commands are for **PowerShell on Windows**. The default offline mode
needs no API key and avoids rate limits.

```powershell
cd C:\path\to\Triage-Bot
pip install -r requirements.txt
$env:LLM_PROVIDER = "offline"

python eval\eval_harness.py `
  --golden data\golden_set.csv `
  --skip-judge `
  --output eval\eval_results_reproduced.csv
```

This runs the pipeline and prints the classification and escalation metrics.
With the included golden set, it completes in a few minutes after the raw
dataset has been loaded.

## Run the agent on sample messages

Download `twcs.csv` from Kaggle
(`thoughtvector/customer-support-on-twitter`) and place it at
`data\twcs.csv`. The raw file is intentionally not committed because it is
large. Then run:

```powershell
$env:LLM_PROVIDER = "offline"
python scripts\run_pipeline.py --n 30 --output results_sample.csv
```

The output CSV contains the message, predicted intent, confidence, drafted
reply, grounding details, escalation decision, and escalation reason.

To run your own input file:

```powershell
python scripts\run_pipeline.py `
  --input path\to\messages.csv `
  --text-col message `
  --output results.csv
```

## Optional hosted LLM mode

Offline mode is the reproducible evaluation path. To use a hosted model
instead, copy `.env.example` to `.env` and provide a valid key. Supported
providers are Anthropic and OpenAI-compatible providers:

```powershell
$env:LLM_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY = "your-key"
python scripts\run_pipeline.py --n 10 --output hosted_results.csv
```

Never commit `.env` or API keys.

## How the pipeline works

```text
Incoming message
       |
       v
Intent classifier
       |
       +--> TF-IDF retriever --> historical customer/reply examples
       |                              |
       v                              v
Escalation policy <----------- grounded reply drafter
       |
       v
Structured result: intent, reply, grounded?, escalate?, reason
```

### Intent taxonomy

The taxonomy is intentionally small and brand-specific:

- `account_access`
- `billing_dispute`
- `delivery_order_issue`
- `product_bug_report`
- `how_to_question`
- `general_complaint`
- `praise_or_thanks`
- `other`

### Grounding

`data_loader.py` reconstructs direct customer-to-brand reply pairs from the
Twitter dataset. `retriever.py` uses TF-IDF and cosine similarity to find up
to three similar resolved cases. Weak matches are not treated as strong
evidence. This is fast and reproducible, but it can miss paraphrases.

### Escalation

The escalation policy is deliberately hybrid:

- high-risk keywords always escalate;
- billing disputes always escalate;
- low-confidence classifications escalate;
- ungrounded drafts escalate;
- remaining cases receive a model judgment.

This prevents the agent from confidently inventing a refund, legal response,
or account-specific resolution.

## Evaluation assets

| Path | Purpose |
|---|---|
| `data\golden_set.csv` | 196 hand-labeled evaluation examples |
| `data\human_judge_template.csv` | 25-row human calibration sample |
| `eval\eval_harness.py` | Metrics, baselines, escalation, judge |
| `eval\eval_results.csv` | Main evaluation output |
| `eval\eval_results_calibrated.csv` | Output including human calibration |
| `eval\rubric.md` | Reply-quality scoring rubric |
| `REPORT.md` | Six-page-style report and failure analysis |
| `DECISION_LOG.md` | Non-obvious design decisions |

To reproduce the judge-human calibration after reviewing the 25 replies:

```powershell
python eval\eval_harness.py `
  --golden data\golden_set.csv `
  --human-calibration data\human_judge_template.csv `
  --output eval\eval_results_calibrated.csv
```

## Repository layout

```text
src/
  config.py          brand, taxonomy, thresholds, and policy
  data_loader.py     dataset loading and reply-pair reconstruction
  retriever.py       TF-IDF historical-case retrieval
  classifier.py      intent classification
  reply_drafter.py   grounded reply generation
  escalation.py      auto-handle/escalation policy
  pipeline.py        end-to-end orchestration
  llm_client.py      offline and hosted provider adapter
scripts/
  run_pipeline.py    command-line runner
  build_golden_set.py reproducible sampling helper
eval/
  eval_harness.py    evaluation and baseline comparison
```

## Scope and limitations

This is a support-triage prototype, not a production support system. It does
not access private order/account systems, process payments, promise refunds,
or reconstruct every multi-turn Twitter thread. The evaluation set is small
and brand-specific, and the offline provider is a transparent heuristic path,
not evidence that a hosted LLM will perform identically.

The next improvements would be full-thread reconstruction, time-split
evaluation, embedding retrieval, resolution-quality filtering, and privacy-safe
account/order tools behind mandatory human approval.
