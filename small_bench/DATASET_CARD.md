# smoke20-v1 dataset card — refined v1.1.0

## Purpose

`smoke20-v1` is a 20-image Phase 1 viability screen for Persian OCR and document VLM systems. It answers only whether a system is worth advancing to a larger benchmark. It is not a held-out test set and must not be presented as a statistically reliable final ranking.

## Contents

- 20 JPEG images
- 10 images in the historical `typed/` storage directory
- 10 images in the historical `hand-written/` storage directory
- Content labels come from `manifest.jsonl`, not directory names
- `hand-written/5.jpg` is printed content despite its historical path

## Identity

- Dataset ID: `smoke20-v1`
- Dataset version: `1.1.0`
- Dataset SHA-256: `9543aff102da6cb4b3df1c4ac0f3c3a7dcf73c302233e8281fe68ad7411c28b7`
- Role: `phase1_screening`
- Uploaded input ZIP SHA-256: `69f53febe4af562d021f3b3ecd2f4c4ecd028d31f50531d716047e5c010040f8`

The dataset hash covers ordered sample identifiers, image hashes, refined scorable-reference hashes and evaluation metadata hashes. The output ZIP checksum is distributed as a sidecar file because an archive cannot safely contain its own checksum.

## Reference layers

Each sample has three text views in `references.jsonl`:

1. `legacy_recovered_text`: the previous recovered transcription, preserved for auditability.
2. `scorable_text`: the text to use for smoke scoring after explicit exclusions.
3. `fa_ir_normalized_text`: a conservative Iranian-Persian normalization of the scorable text.

All references remain `ai_audited_not_human_reviewed`.

## Important exclusions

- `typed-004`: clipped opening/closing material excluded from the scorable view.
- `hand-written-004`: uncertain attribution excluded from the scorable view.
- `typed-006`: plain CER/WER disabled until table structure is annotated.
- `typed-007`: plain CER/WER disabled until form reading order is defined.
- `typed-009`: plain CER/WER disabled until redaction masks and privacy review are complete.

## Provenance and privacy

Original acquisition URLs, creators, licenses and redistribution permissions are not available in this package. Legacy manifest pointers refer to source Markdown members that are absent. `typed-009` contains a redacted legal document and must remain private until an explicit privacy and redistribution review is completed.

## Validation

```powershell
python small_bench/scripts/validate_dataset.py small_bench
```

The validator checks manifest/reference cardinality, hashes, paths, identity, image readability and compatibility JSON files. It intentionally does not claim human transcription accuracy.
