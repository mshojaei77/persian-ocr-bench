# smoke20-v1 refinement audit

## Result

This package refines the 20-image Phase 1 screen as dataset version `1.1.0` while preserving historical image paths for existing adapters.

It remains **development/model-selection data**, not a statistically reliable or held-out leaderboard test set.

## Critical findings

1. The uploaded ZIP SHA-256 is `69f53febe4af562d021f3b3ecd2f4c4ecd028d31f50531d716047e5c010040f8`, while every legacy manifest row declares `e7e085e636512bb1ec009877a2d616d85865a879fb6a83b3052108fe9118fce1`.
2. All 20 legacy `source_member` paths point to Markdown files that are absent from the uploaded package.
3. `typed-004` had a reference containing text outside the visible crop and a clipped final line. A separate scorable view now keeps only complete visible text.
4. The attribution in `hand-written-004` is uncertain and is excluded from its scorable text pending human review.
5. `typed-006`, `typed-007`, and `typed-009` are excluded from plain-text CER/WER ranking because table/form ordering or redaction masking is not yet sufficiently specified.
6. No reference has been upgraded to human-reviewed status.

## What changed

- Dataset version advanced from `1.0.0` to `1.1.0`.
- Added a v2 manifest with document type, capture type, layout complexity, script profile, text density, reference risk, metric eligibility, privacy flags, image statistics, and review priority.
- Added `references.jsonl` containing legacy recovered text, refined scorable text, Persian-normalized text, hashes, boundary policy, and excluded spans.
- Regenerated compatibility `typed.json` and `hand-written.json` from refined scorable references.
- Preserved the original JSON and manifest under `legacy_original/`.
- Added `review_queue.csv`, `dataset_identity.json`, and a standalone validator.
- Removed the invalid implication that the missing Markdown members are recoverable from this package.

## Human review order

### P0

- Dense or difficult handwriting: `hand-written-003`, `hand-written-007`, `hand-written-008`, `hand-written-010`
- Uncertain attribution: `hand-written-004`
- Crop-boundary correction confirmation: `typed-004`
- Structured form/table annotations: `typed-006`, `typed-007`
- Redaction masks, privacy and text review: `typed-009`

### P1

Review all low-resolution or mixed-script samples.

### P2

Review clear linear printed pages and simple graphics last.

## Scoring policy

- Use `references.jsonl.scorable_text` for raw smoke CER/WER.
- Use `fa_ir_normalized_text` for the normalized Persian diagnostic.
- Honor `metric_eligibility`; do not silently score ineligible pages as linear text.
- Keep failures in the denominator.
- Do not publish model rankings based on this 20-image set.
