# Persian OCR roadmap

This roadmap reflects the repository state on 2026-07-13. The three P0
classical adapters have completed the 20-image `smoke20-v1` viability screen;
that screen is not a final ranking.

## 0. Keep the current baseline reproducible

- [x] Keep the canonical `src/persian_ocr` package and optional runtime extras.
- [x] Validate the smoke20 manifest, content hashes, and dataset identity.
- [x] Run Tesseract, PP-OCRv5 Arabic, and EasyOCR through the shared v2 artifact contract.
- [x] Generate the three-model Phase 1 screening report with portable paths and a manifest.
- [x] Add raw Unicode versus Persian-normalized scores, faithfulness, reading-order,
  exactness, and operational diagnostics to the Phase 1 metric contract.
- [x] Add an archive-first `--reset` path before rerunning smoke20-v1.
- [ ] Resolve the three catalog status warnings: change the first-wave entries from
  `adapter_ready` to `benchmarked` in `models.yaml`.
- [ ] Keep `bench_runs/smoke20-v1/` artifacts archived with the exact command,
  dependency lock, model identity, and hardware/runtime metadata.

## 1. Finish Phase 1 screening coverage

Implement adapters and produce complete smoke20 artifacts in this order:

1. `hezar_crnn_base_fa_v2` (P0, Persian-specific CRNN baseline)
2. `surya_ocr_2` (P0, document OCR control)
3. `paddleocr_vl_1_6` (P0, document/VLM control)
4. `qwen3_vl_2b_persian_arabic_ocr` (P0, Persian-targeted VLM)
5. `qwen3_vl_2b_instruct` (P0, general VLM control)

For every adapter:

- [ ] Pin the exact checkpoint/revision and license metadata.
- [ ] Add a small initialization smoke path before a 20-image run.
- [ ] Preserve failure accounting: every selected image must have a result.
- [ ] Validate the artifact against the expected model id and comparison class.
- [ ] Regenerate the screening report after each completed wave.

Then evaluate P1 models only where a capability gap remains:
`weightedai_persian_ocr`, `falcon_ocr`, `got_ocr2`, `hunyuanocr_1_5`,
`lightonocr_2_1b`, `mineru_2_5_pro_2605`, `khanandeh_0_1_persian_ocr_2b`,
`firered_ocr`, `dots_ocr`, `dots_mocr`, `deepseek_ocr_2`, `chandra_ocr_2`,
`qianfan_ocr`, and `qwen3_vl_8b_instruct`. Keep P2 entries as diagnostic or
negative controls until the P0/P1 results justify their cost.

## 2. Human-review and dataset readiness

- [ ] Review all 20 references against the source images and correct transcripts.
- [ ] Record reviewer, date, review scope, and corrections per sample.
- [ ] Replace `ai_assisted_recovered_not_human_reviewed` only when review is
  actually complete; do not bypass the strict validator.
- [ ] Add the sealed large-benchmark manifest only after the review gate passes.
- [ ] Expand coverage across clean print, degraded scans, phone photos,
  multi-column RTL, mixed Persian/English, tables/forms, handwriting, historical
  fonts, and formulas.
- [ ] Freeze the large-benchmark dataset identity before collecting model runs.

## 3. Final benchmark and comparable leaderboards

- [ ] Define the large-benchmark protocol and hardware profile in versioned config.
- [ ] Run only complete, failure-free artifacts with identical scorer, protocol,
  dataset, and capability-class identities.
- [ ] Publish separate raw-recognition, normalized-recognition, Persian failure
  slice, document-structure, and operations reports.
- [ ] Report CER, WER, exact-line accuracy, reading order, layout/table metrics,
  throughput, cold start, peak VRAM, and failure rate where applicable.
- [ ] Keep detector/recognizer pipelines, component recognizers, and document
  VLMs in separate comparison classes; never combine them into one rank.
- [ ] Require an explicit decision record for exclusions, ties, and incompatible
  runs.

## 4. Release and maintenance

- [ ] Add a reproducible report command to the release checklist.
- [ ] Add a non-pytest smoke benchmark covering one real page per implemented
  runtime and one initialization-failure case.
- [ ] Verify wheel installation and CLI execution from outside the repository.
- [ ] Keep model downloads, secrets, and heavyweight runtimes out of the core
  install; update `uv.lock` whenever dependencies change.
- [ ] Record benchmark provenance, checksums, generated-file manifests, and known
  limitations before sharing results.
- [ ] Update this roadmap and `models.yaml` after each completed adapter wave.

## Definition of done

The project is ready to claim a final result only when the dataset passes strict
human review, the large benchmark is frozen, every published row is complete and
comparable, and the generated report can be reproduced from a clean environment.
