# smoke20-v1 dataset card

## Purpose

`smoke20-v1` is a 20-image Phase 1 viability screen for Persian OCR systems. It is development/model-selection data, not a held-out test set. Results may answer “is this system worth a larger benchmark?” They must not be presented as a statistically reliable model ranking.

The set contains ten images under `small_bench/typed/` and ten under `small_bench/hand-written/`. Those directory names are historical storage paths, not authoritative content labels. `small_bench/manifest.jsonl` supplies the reviewed machine-readable `content_type`; notably, `hand-written/5.jpg` is printed and `hand-written/7.jpg` is handwritten.

## Version and identity

- Dataset ID: `smoke20-v1`
- Dataset version: `1.0.0`
- Role: `phase1_screening`
- Canonical source archive: `small_bench.zip`
- Source archive SHA-256: `e7e085e636512bb1ec009877a2d616d85865a879fb6a83b3052108fe9118fce1`
- Canonical references: `small_bench/typed.json` and `small_bench/hand-written.json`
- Canonical sample registry: `small_bench/manifest.jsonl`

The manifest records a SHA-256 for each image, recovered reference string, archive member, and archive itself. `dataset_sha256` is computed over the dataset ID/version and the ordered `sample_id|image_sha256|reference_sha256` rows. It is a content identity, not a claim about provenance or licensing.

## Recovery and transcription policy

The original Markdown references are recovered from `small_bench.zip`. The live JPEG for every sample must be byte-identical to its archived JPEG before references can be regenerated.

Recovery follows these rules:

- Unicode text is normalized to NFC and stored as UTF-8.
- Visible/scorable text surrounding bracketed annotations is preserved. This prevents redacted spans in `typed/9.jpg` from deleting the rest of an entire line.
- Only an explicit allowlist of image-only descriptions, such as `[لوگوی شرکت]`, is removed.
- Textual watermarks and footer text are preserved.
- Markdown heading/emphasis markers, table pipes/alignment rows, and HTML `<br>` presentation tags are not reference text.
- Underscores inside visible hashtags or text are preserved.
- Source redactions remain untranscribed; visible text surrounding them remains in reading order.

The references were recovered from the archived Markdown and checked with AI-assisted visual inspection. They were **not human-reviewed**. Every manifest row therefore uses `review_status=ai_assisted_recovered_not_human_reviewed`. This status must not be upgraded without a recorded human review.

## Manifest contract

Each JSONL row contains:

- identity: `schema`, `dataset_id`, `dataset_version`, `dataset_sha256`, `sample_id`;
- evaluation labels: `split`, `content_type`, `condition`, and compatibility field `track`;
- image identity: relative `image` path and `image_sha256`;
- reference identity: `reference_corpus`, `reference_key`, and per-sample `reference_sha256`;
- recoverability: `source_archive`, `source_archive_sha256`, `source_member`, and `source_member_sha256`;
- review/provenance: `annotation_method`, `review_status`, and `provenance_status`.

`content_type` and `condition` are dataset properties. Model capability claims such as handwriting support belong in the model registry, not in this manifest.

## Provenance, rights, and privacy

The original acquisition URLs, creators, licenses, and redistribution permissions were not recorded when these files entered the repository. The manifest marks this honestly as `source_and_license_unknown`. The set should remain a local engineering screen until provenance and rights are established.

Some pages contain names, addresses, case identifiers, correspondence, or source redactions. In particular, `typed/9.jpg` is a legal document with partially redacted personal information. Do not publish or expand this dataset without a privacy and licensing review.

These gaps are documented in line with current dataset-card practice, which calls for language, license, creation context, limitations, and responsible-use information. See the [Hugging Face dataset-card guide](https://huggingface.co/docs/hub/datasets-cards) and the checksum-bearing [MLCommons Croissant format](https://github.com/mlcommons/croissant).

## Known limitations

- Twenty pages cannot establish a robust leaderboard or population-level conclusion.
- The sample mix is opportunistic and not source-group balanced.
- References are AI-assisted recoveries rather than human double-checked annotations.
- Plain-text CER/WER does not independently measure layout, table structure, reading order, or redaction handling.
- The 20 samples must not be reused in the later held-out large benchmark.

## Deterministic maintenance commands

Check without writing:

```powershell
uv run python scripts/merge_small_bench.py --check
uv run python scripts/validate_smoke20.py
```

Regenerate atomically from the frozen archive, then verify:

```powershell
uv run python scripts/merge_small_bench.py --write
uv run python scripts/validate_smoke20.py
```

No OCR model is loaded by these commands, and no pytest test is written or run.
