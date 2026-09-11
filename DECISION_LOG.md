# Decision log

1. **AppleSupport as the brand:** it has a large, coherent subset and enough
   repeated product/support issues for retrieval.
2. **Eight intents:** the taxonomy is small enough to label consistently while
   separating access, billing, order, bugs, how-to, complaint, praise, and
   residual traffic.
3. **First-hop resolution pairs:** customer tweets are paired with the direct
   brand reply; deeper thread reconstruction is deferred to avoid inferred
   attribution errors.
4. **TF-IDF retrieval:** it is fast, inspectable, and reproducible without
   downloading an embedding model.
5. **Similarity floor:** weak matches are not presented as evidence; this
   trades coverage for safer drafts.
6. **Prompted classifier:** the assignment permits any LLM and a prompted model
   is quicker to iterate than a fine-tune at this dataset size.
7. **Offline provider:** a deterministic local path makes the headline run
   reproducible and prevents API availability from becoming a hidden metric.
8. **Hybrid escalation:** high-risk keywords, billing, low confidence, and
   missing grounding are deterministic gates before any model judgment.
9. **Billing escalation:** monetary disputes require account-specific checks
   unavailable in a public-tweet-only system.
10. **Golden sampling:** proxy-intent stratification prevents common complaint
    traffic from crowding out rare categories; final labels are human.
11. **No synthetic human agreement:** the harness refuses to claim judge
    agreement until a human fills the calibration scores.
12. **Accuracy plus macro-F1:** accuracy shows aggregate behavior while macro-F1
    exposes minority-intent failures.
13. **No full-dataset training:** a full run is unnecessary for a take-home
    proof and makes leakage and runtime harder to audit.
