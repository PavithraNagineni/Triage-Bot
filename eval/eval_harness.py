"""
Evaluation harness for deliverable #3.

Given a hand-labeled golden set CSV (columns: message, true_intent,
should_escalate, and optionally human_judge_score / escalate_reason_gold),
this script:

  1. Runs the real pipeline (src.pipeline.run_pipeline) on every message.
  2. Computes classification accuracy + macro-F1 against two baselines:
       - trivial: always predict the majority intent
       - simple: crude keyword-rule classifier (see build_golden_set.py's
         PROXY_RULES, reused here as the "simple" baseline)
  3. Computes escalation agreement (accuracy, precision, recall) against
     should_escalate.
  4. Runs an LLM-as-judge rubric over each drafted reply (relevance,
     groundedness, tone, actionability; 1-5 each) and, if a
     `human_judge_score` column is present for a subset of rows, reports
     judge-vs-human agreement (exact match rate + mean absolute
     difference) so the judge's reliability is itself measured, not
     assumed.

Usage:
    python eval/eval_harness.py --golden data/golden_set.csv --output eval/eval_results.csv
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src import config
from src.data_loader import load_pairs_for_brand
from src.llm_client import complete_json
from src.pipeline import run_pipeline
from src.retriever import GroundingRetriever
from scripts.build_golden_set import proxy_tag

JUDGE_SYSTEM_PROMPT = f"""You are grading a customer-support reply drafted by an AI agent for \
{config.BRAND}, on a 1-5 scale (5=excellent) for each of these dimensions:

- relevance: does the reply actually address what the customer asked/reported?
- groundedness: does the reply avoid inventing policies/promises not evidenced by the note provided?
- tone: is the tone appropriate for a support interaction (empathetic, professional, not robotic)?
- actionability: does the customer know what happens next after reading this reply?

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"relevance": <1-5>, "groundedness": <1-5>, "tone": <1-5>, "actionability": <1-5>,
  "overall": <1-5, your holistic judgment, need not be the average>,
  "justification": "<one short sentence>"}}
"""


def llm_judge(message: str, reply: str, grounding_note: str) -> dict:
    user = (
        f"Customer message:\n{message}\n\n"
        f"Drafted reply:\n{reply}\n\n"
        f"Grounding note from the drafting step: {grounding_note}"
    )
    return complete_json(JUDGE_SYSTEM_PROMPT, user, temperature=0.0, max_tokens=200)


def trivial_baseline_predict(train_intents: pd.Series, n: int) -> list[str]:
    """Always predict the majority class. The floor any real system must clear."""
    majority = train_intents.mode().iloc[0]
    return [majority] * n


def simple_baseline_predict(messages: list[str]) -> list[str]:
    """Crude keyword-rule classifier, reusing build_golden_set's proxy tagger.

    This is the "simple, non-trivial" baseline: no LLM, no learning,
    just hand-written regex rules. If the LLM pipeline can't beat this,
    the LLM isn't earning its cost/latency.
    """
    mapping = {
        "other_or_general_complaint": "general_complaint",  # closest available label
    }
    preds = []
    for m in messages:
        tag = proxy_tag(m)
        preds.append(mapping.get(tag, tag))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="Hand-labeled golden set CSV")
    ap.add_argument("--output", default="eval/eval_results.csv")
    ap.add_argument("--human-calibration", default=None,
                    help="CSV with message,human_judge_score,human_notes columns")
    ap.add_argument("--skip-judge", action="store_true", help="Skip the LLM-as-judge pass (faster/cheaper)")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    if args.human_calibration:
        calibration = pd.read_csv(args.human_calibration)
        required = {"message", "human_judge_score"}
        missing = required - set(calibration.columns)
        if missing:
            raise ValueError(
                f"Human calibration file is missing required columns: {sorted(missing)}"
            )
        scores = calibration[["message", "human_judge_score"]].drop_duplicates("message")
        golden = golden.drop(columns=["human_judge_score"], errors="ignore").merge(
            scores, on="message", how="left"
        )
    golden = golden[golden["true_intent"].notna() & (golden["true_intent"] != "")]
    assert len(golden) > 0, "Golden set has no labeled rows (true_intent is empty everywhere)."
    print(f"Loaded {len(golden)} labeled golden examples.")

    print("Building grounding retriever from the full resolved-pairs corpus...")
    pairs = load_pairs_for_brand()
    retriever = GroundingRetriever(pairs)

    # ---- Run the real pipeline over every golden example -------------------
    print("Running pipeline over golden set (classify -> ground -> draft -> escalate)...")
    results = []
    for msg in tqdm(golden["message"].astype(str).tolist()):
        try:
            r = run_pipeline(msg, retriever)
            results.append(r.to_dict())
        except Exception as e:  # noqa: BLE001
            results.append({"message": msg, "intent": "other", "escalate": True,
                             "drafted_reply": "", "grounding_note": "", "error": str(e)})
    pred_df = pd.DataFrame(results)
    merged = pd.concat([golden.reset_index(drop=True), pred_df.reset_index(drop=True).add_prefix("pred_")], axis=1)

    # ---- Classification: pipeline vs. two baselines -------------------------
    y_true = merged["true_intent"]
    y_pred_pipeline = merged["pred_intent"]
    y_pred_trivial = trivial_baseline_predict(y_true, len(merged))
    y_pred_simple = simple_baseline_predict(merged["message"].astype(str).tolist())

    def clf_metrics(y_true, y_pred, name):
        return {
            "system": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    clf_summary = pd.DataFrame([
        clf_metrics(y_true, y_pred_trivial, "trivial_baseline (majority class)"),
        clf_metrics(y_true, y_pred_simple, "simple_baseline (keyword rules)"),
        clf_metrics(y_true, y_pred_pipeline, "llm_pipeline"),
    ])
    print("\n=== Classification: pipeline vs baselines ===")
    print(clf_summary.to_string(index=False))

    # ---- Escalation agreement ------------------------------------------------
    if "should_escalate" in merged.columns and merged["should_escalate"].notna().any():
        esc_gold = merged["should_escalate"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
        esc_pred = merged["pred_escalate"].astype(bool)
        esc_summary = {
            "accuracy": accuracy_score(esc_gold, esc_pred),
            "precision": precision_score(esc_gold, esc_pred, zero_division=0),
            "recall": recall_score(esc_gold, esc_pred, zero_division=0),
        }
        print("\n=== Escalation agreement vs. hand-labeled ground truth ===")
        for k, v in esc_summary.items():
            print(f"  {k}: {v:.3f}")
    else:
        print("\n(No `should_escalate` labels found in golden set — skipping escalation metrics.)")

    # ---- LLM-as-judge reply quality -------------------------------------------
    if not args.skip_judge:
        print("\nRunning LLM-as-judge over drafted replies...")
        judge_rows = []
        for _, row in tqdm(merged.iterrows(), total=len(merged)):
            try:
                j = llm_judge(row["message"], row.get("pred_drafted_reply", ""), row.get("pred_grounding_note", ""))
            except Exception as e:  # noqa: BLE001
                j = {"relevance": None, "groundedness": None, "tone": None,
                     "actionability": None, "overall": None, "justification": f"error: {e}"}
            judge_rows.append(j)
        judge_df = pd.DataFrame(judge_rows).add_prefix("judge_")
        merged = pd.concat([merged.reset_index(drop=True), judge_df.reset_index(drop=True)], axis=1)

        print("\n=== LLM-as-judge reply quality (mean scores, 1-5) ===")
        for col in ["judge_relevance", "judge_groundedness", "judge_tone", "judge_actionability", "judge_overall"]:
            if col in merged.columns:
                print(f"  {col}: {pd.to_numeric(merged[col], errors='coerce').mean():.2f}")

        # ---- Judge-vs-human agreement (only over rows with a human score) ----
        if "human_judge_score" in merged.columns and merged["human_judge_score"].notna().any():
            sub = merged[merged["human_judge_score"].notna()].copy()
            sub["judge_overall_num"] = pd.to_numeric(sub["judge_overall"], errors="coerce")
            sub["human_judge_score_num"] = pd.to_numeric(sub["human_judge_score"], errors="coerce")
            sub = sub.dropna(subset=["judge_overall_num", "human_judge_score_num"])
            if len(sub) > 0:
                exact_match = (sub["judge_overall_num"] == sub["human_judge_score_num"]).mean()
                mae = (sub["judge_overall_num"] - sub["human_judge_score_num"]).abs().mean()
                within_1 = ((sub["judge_overall_num"] - sub["human_judge_score_num"]).abs() <= 1).mean()
                print(f"\n=== Judge-vs-human agreement (n={len(sub)}) ===")
                print(f"  exact match rate: {exact_match:.2%}")
                print(f"  within +-1 point: {within_1:.2%}")
                print(f"  mean absolute difference: {mae:.2f}")
                print("  NOTE: report this in the report's judge-reliability section. "
                      "If agreement is weak, say so — that's exactly what the "
                      "'misleading headline number' section should flag.")
        else:
            print("\n(No `human_judge_score` column with values found — add human scores for a "
                  "subset of rows to measure judge reliability, required for deliverable #3.)")
    else:
        print("\n(--skip-judge set: skipped LLM-as-judge pass.)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nFull per-example results written to {args.output}")


if __name__ == "__main__":
    main()
