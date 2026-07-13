# Dataset schema

## `manifest.jsonl`

One row per image. Important fields:

- `sample_id`: stable identity independent of the historical folder name
- `content_type`: printed or handwritten
- `document_type`, `capture_type`, `layout_complexity`
- `condition`: observed challenge tags
- `image_sha256`, `image_properties`
- `reference_scorable_sha256`, `reference_normalized_sha256`
- `reference_risk`, `review_priority`, `review_status`
- `metric_eligibility`: which metrics are valid for the current annotation
- `privacy_sensitive`, `redaction_policy`
- `dataset_sha256`: shared content identity

## `references.jsonl`

- `legacy_recovered_text`: unmodified previous recovered reference
- `scorable_text`: corrected smoke-scoring view
- `fa_ir_normalized_text`: conservative normalized view
- `boundary_policy`: how crop/uncertainty was handled
- `excluded_spans`: text intentionally omitted from the scoring view

## Review status

Only a human review with recorded reviewer, date, scope and decision may change `review_status` from `ai_audited_not_human_reviewed`.
