# Changelog

## 1.1.0

- Rebuilt the manifest as schema `persian_ocr_smoke_manifest_v2`.
- Added a separate reference registry with legacy, scorable and normalized text.
- Corrected the scoring boundary for `typed-004`.
- Excluded the uncertain attribution from `hand-written-004` scoring.
- Disabled linear CER/WER for table/form/redacted samples lacking suitable annotations.
- Added image-quality diagnostics and near-duplicate checks.
- Added a prioritized human-review queue.
- Preserved historical image paths and compatibility JSON files.
- Preserved all original metadata/reference files under `legacy_original/`.
- Documented the archive-hash mismatch and missing source Markdown files.
