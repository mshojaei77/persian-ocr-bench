# Human review guide

For each sample in `review_queue.csv`:

1. Open the original JPEG at 100–300% zoom.
2. Compare character by character against `references.jsonl.scorable_text`.
3. Do not modernize spelling, fix grammar or normalize digit scripts.
4. Preserve visible punctuation, ZWNJ and meaningful line breaks.
5. Exclude only non-text decoration or spans explicitly hidden by redaction.
6. Record uncertain characters instead of guessing.
7. A second reviewer must check every P0 sample independently.
8. Record reviewer, date, decision and correction summary in the queue.
9. Recompute all reference and dataset hashes after accepted corrections.
10. Do not set final-leaderboard eligibility on this 20-image screen.
