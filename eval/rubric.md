# Reply-quality judge rubric

Score each drafted reply from 1 to 5 on relevance, groundedness, tone, and
actionability, then assign one holistic `human_judge_score`:

* **5** — directly addresses the request, makes no unsupported promise, is
  empathetic, and gives a concrete safe next step.
* **4** — useful and safe, with only a minor omission or wording issue.
* **3** — partially useful but generic, incomplete, or weakly grounded.
* **2** — mostly misses the request or gives an unclear/unsupported next step.
* **1** — wrong, unsafe, fabricated, or unusable.

For calibration, independently score the 25 rows in
`data/human_judge_template.csv` before looking at the LLM score. Put an
integer 1–5 in `human_judge_score`, then run:

```powershell
python eval/eval_harness.py --golden data/golden_set.csv `
  --human-calibration data/human_judge_template.csv `
  --output eval/eval_results_calibrated.csv
```

The reported exact match, within-one agreement, and MAE are the evidence for
judge reliability; no agreement number is reported when no human labels exist.
