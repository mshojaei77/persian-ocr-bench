# Persian OCR Benchmark Roadmap

Audit date: 2026-07-13  
Scope: the live working tree, `models.yaml`, `small_bench`, committed benchmark artifacts, packaging, and the path from model screening to a large Persian OCR evaluation.

## Roadmap charter

> `small_bench` is an intentionally small Phase 1 screening set. Its 20 images exist to identify models worth the time, hardware, and annotation cost of a large benchmark. It is not representative enough to establish a winner, publish a credible leaderboard, or support fine-grained ranking.

The intended decision funnel is:

```text
26-model research catalog
    -> eligibility and two-image preflight
    -> fixed 20-image Phase 1 screen
    -> Advance / Hold / Reject / Blocked
    -> diverse shortlist of about 5-8 systems
    -> new sealed, large, stratified benchmark
    -> robustness and operational evaluation
    -> deployment decision by use case
```

`small_bench` becomes development and model-selection data as soon as it is used for screening. None of its 20 images should appear in the final held-out test set.

## Executive judgment

The project has a strong start: Persian-aware normalization and diagnostics, raw per-page outputs, model/component identity for the newer adapters, operational timing, uncertainty estimates, and an unusually candid 26-model catalog. The live execution surface is much smaller: two committed benchmark artifacts, one additional adapter ready to run, and 23 catalog-only candidates.

The main risk is not that 20 images are too few for Phase 1. Twenty is reasonable for a cheap viability screen. The risk is that the current references are not yet trustworthy enough even for screening: the Markdown-to-JSON migration removed visible transcription text, review status is inconsistent, and both committed artifacts still point to deleted Markdown files. Those historical scores must not be used to advance or reject models.

The project should therefore repair and freeze `smoke20-v1`, screen a diverse set of candidates without producing ranks, and spend large-benchmark resources only on the systems that survive that gate.

## Ten critical opinions

| # | Severity | Critical opinion | Evidence and impact | Required response |
|---:|---|---|---|---|
| 1 | Critical | The current ground truth was damaged during migration. | `scripts/merge_small_bench.py` drops an entire line whenever it contains `[ ... ]`. Comparison with the tracked `small_bench.zip` shows visible transcription losses in at least `typed/4`, `typed/9`, and `hand-written/10`; the loss in `typed/9` removes substantial legal-document text around redacted names. Current CER/WER can therefore be wrong for the reference, not only the prediction. | Recover from the archived Markdown, compare every one of the 20 references to its image, and freeze a reviewed `smoke20-v1`. Do not rerun the destructive converter first. |
| 2 | Critical | “Reviewed” does not currently mean reviewed. | The manifest has no reviewer or `reference_quality` fields. The Tesseract loader unconditionally returns `reviewed` for migrated JSON, PP-OCR inherits that behavior, while EasyOCR correctly requires an explicit manifest flag. A gate whose meaning changes by adapter is not a gate. | Put review state, annotator/reviewer, revision, provenance, license, and reference hash in the manifest; make every adapter enforce the same policy. |
| 3 | Critical | The committed benchmark artifacts are historical and non-reproducible from the live dataset. | Both artifacts contain 20/20 stale paths to deleted `.md` references; Tesseract also lacks the expected model ID. The status validator reports all three problems. The JSON reference corpora are a different benchmark version from the one those artifacts scored. | Mark existing results `legacy_invalid_for_selection`, define run identity, then regenerate fixed controls only after `smoke20-v1` is frozen. |
| 4 | Critical | The project currently presents a screening set as a leaderboard. | The artifacts call the run `persian_ocr_smoke20`, yet `leaderboard.py` assigns numeric ranks. Twenty selected pages can reveal catastrophic incompatibility, but small point differences cannot establish a general Persian OCR winner. | Rename the Phase 1 output a screening report, remove ranks, and record `Advance`, `Hold`, `Reject`, or `Blocked` with reasons. |
| 5 | High | The ranking logic is unsafe even apart from sample size. | Current Tesseract and PP-OCR page-bootstrap intervals overlap substantially. The leaderboard ignores run compatibility and uncertainty, ranks incomplete/mismatched JSON, and `row["cer"] or infinity` places a valid `0.0` CER last. | Do not rank Phase 1. For the large benchmark, require matching identities and use paired document-level differences with practical-tie rules. |
| 6 | High | Dataset categories are not reliable enough for slice claims. | `collect_track_rows` maps every track without the substring `hand` to `typed`, producing 12 typed and 8 handwritten rows despite the 10/10 directory split. Visual inspection also contradicts at least one manifest label. One-page categories cannot support a category conclusion anyway. | Add explicit, reviewed `split`, `content_type`, and condition labels; aggregate exact fields and publish counts. Treat one-page slices as case studies only. |
| 7 | High | Unlike systems are being pushed toward one aggregate comparison. | The catalog mixes line recognizers, detector-plus-recognizer pipelines, classical OCR, and full-page document VLMs. A line recognizer cannot fairly be judged on layout or RTL page order without a fixed detector and ordering policy. | Define separate eligibility classes and score recognition, document structure, and operations separately. Never create one universal quality rank across incompatible scopes. |
| 8 | High | Catalog ambition is ahead of execution capacity. | Of 26 active entries, 23 are catalog-only, one is adapter-ready, and two are benchmarked; 18 are P0. Implementing every nominal P0 before screening would spend heavily before learning. | Use waves, cap the large-benchmark shortlist at roughly 5-8 diverse systems, and advance no more than about two candidates per comparison class unless evidence justifies an exception. |
| 9 | Critical | Installation and default paths are broken. | `pyproject.toml` packages the deleted `src/persian_ocr` tree while live modules are flat under `src/`. `src/leaderboard.py` resolves the repository root one parent too high. The project also has no installed console scripts, README metadata, or license metadata. | Repair the package root and output-path safety before reusable orchestration is declared stable; keep this engineering lane small enough that it does not delay the learning loop. |
| 10 | High | The shared benchmark contract is coupled to one adapter and incompletely fingerprinted. | EasyOCR and PP-OCR import dataset, preprocessing, normalization, metrics, and artifact helpers from `tesseract_fas.py`; the Tesseract file is 1,243 lines. EasyOCR hashes its own runner but not the imported scorer implementation. | Extract a small shared contract for data, normalization, scoring, artifacts, and adapter IO. Fingerprint all code that affects a score and preserve raw model output. |

## Rules that apply to every phase

- Keep `smoke20` as a screening/regression asset, never as the final test.
- Compare systems only within a declared capability class.
- Preserve raw model output before normalization, ordering, or postprocessing.
- Record failures as failures; never convert them to empty successful predictions.
- Keep raw and Persian-normalized metrics side by side.
- Prefer advancing an ambiguous model to falsely rejecting it when the next-stage cost is acceptable.
- Keep prompts, detectors, preprocessing, decoding, and hardware visible in every result.
- Use reusable scripts in `scripts/` for repeated commands and `uv` for dependencies.
- Use executable smoke/eval scripts for validation; do not add or run pytest tests.

## Phase 0 — Recover and freeze a safe screening protocol

This phase is blocking. Do not use the current scores for model decisions and do not run `scripts/merge_small_bench.py` until it is made safe.

### Data repair

- Recover the 20 original Markdown references from `small_bench.zip` into a review workspace without overwriting the current JSON.
- Produce a per-sample diff between archived Markdown, current JSON, and the visible image.
- Have a human review all 20 transcriptions, including redactions, digits, ZWNJ, Persian/Arabic Yeh and Kaf, line breaks, and text that belongs to the image rather than editorial description.
- Correct manifest labels after visual review; do not infer typed/handwritten from a folder name or substring.
- Declare the canonical reference corpus and whether the ZIP is source material or an archival backup.

### Safety and identity

- Make the migration utility fail closed on zero inputs, reject image/reference count mismatches, write atomically, and provide a non-mutating `--check` mode.
- Create `smoke20-v1` identity with SHA-256 for every image/reference plus a digest of the complete manifest.
- Add per-sample source, rights/license, collection date when known, transcription method, annotator, reviewer, review status, revision, split, content type, and condition labels.
- Define the minimum comparable run identity: dataset digest, model and component revisions, prompt/configuration, preprocessing, scorer/runner hashes, package lock, device, precision, and hardware.
- Fix repository-root resolution and prevent outputs from escaping the repository by default.
- Mark the two existing artifacts as legacy and regenerate Tesseract/PP-OCR controls only after the freeze.

Exit criteria:

- Exactly 20 images resolve to 20 reviewed, non-empty references.
- Archived, reviewed, and normalized text differences are explicit rather than silently discarded.
- All manifest paths and hashes validate.
- Every adapter agrees on review status and run identity.
- Maintenance checks cannot alter ground truth without an explicit write action.
- Regenerated control artifacts point to live references and pass strict status validation.

## Phase 1 — Screen model viability on 20 images

This is the intended role of `small_bench`.

### Phase 1A — Eligibility and two-image preflight

Assign each candidate to one comparison class before running it:

1. Full-page detector plus recognizer.
2. Line recognizer or OCR component, evaluated with a fixed detector/crop protocol.
3. Full-page document OCR/VLM parser.

For each candidate:

- Confirm exact Persian support evidence, license, model revision, trust-remote-code policy, hardware fit, and adapter ownership.
- Run one reviewed printed page and one reviewed handwriting page where the model claims both capabilities.
- Require structured output, non-empty literal transcription, bounded runtime, and preserved errors/raw output.
- Assign `Blocked`, not `Reject`, when a valid run is prevented by missing hardware, dependencies, licensing clarity, or a comparable page pipeline.

### Phase 1B — Complete `smoke20-v1` screen

- Run the identical frozen 20 pages for every eligible candidate.
- Prefer 20/20 completion. Permit one controlled retry for a transient failure; unresolved reliability becomes `Hold` or `Reject` depending on severity.
- Report coverage, failure rate, macro and micro CER, WER, median page CER, paired page win/tie/loss versus the appropriate control, warm latency, and observed failure modes.
- Detect blank output, gross omission, unsupported-script output, truncation, invented text, duplication, and repetition loops.
- Show results by exact reviewed slice, but do not infer population performance from a one-page slice.
- Generate a screening matrix, not a numeric leaderboard.

### Promotion decisions

| Decision | Meaning |
|---|---|
| `Advance` | The run is complete and reproducible, and the system is Pareto-competitive within its class or supplies a predeclared unique capability/operational advantage. |
| `Hold` | The model is promising but evidence is ambiguous, one reliability issue remains, or operational/comparability data is incomplete. |
| `Reject` | Repeated empty/corrupt output, severe hallucination/repetition, unusable Persian, or domination on quality, reliability, and cost with no unique role. |
| `Blocked` | A valid evaluation could not be completed because of adapter, detector, dependency, hardware, or licensing constraints. |

Shortlist policy:

- Retain Tesseract as the deterministic CPU floor and at least one compact detector-plus-recognizer control.
- Advance at most about two systems per comparison class, plus a justified specialist or operational control.
- Target roughly 5-8 systems for the large benchmark; this is a cost guardrail, not a forced quota.
- Freeze the shortlist and its reasons before building or exposing the final held-out test.

Exit criteria:

- Every attempted model has a decision, class, evidence, and reason.
- No Phase 1 output contains a `rank`, `winner`, or general Persian-performance claim.
- The shortlist covers materially different architectures/capabilities rather than near-duplicates only.
- `smoke20-v1` is formally classified as selection data and excluded from Phase 2 test data.

## Phase 2 — Build a sealed, large Persian benchmark

There is no universal correct page count. Size the benchmark for both coverage and decision precision. A practical initial planning target is 400-1,000 unique pages from many independent documents, with at least 50 pages in each primary slice; adjust upward after a pilot until the 95% confidence interval for the leading paired difference is narrow enough for the project's predeclared practical threshold.

### Dataset design

- Split by source document, template, book, writer, and collection origin—not by independent page or crop—so related samples cannot cross selection and test boundaries.
- Deduplicate by source identity, perceptual image similarity, and text similarity.
- Keep a sealed local test set inaccessible during adapter, prompt, threshold, normalization, and preprocessing development.
- Record source URL/ID, rights and redistribution status, retrieval date, document/page ID, split, all slice labels, annotation history, reviewer, and hashes.
- Publish a dataset card covering motivation, composition, collection, annotation, normalization, intended use, limitations, bias, language, size, and licensing.

### Separate evaluation tracks

- Printed full-page recognition: clean digital, scans, photos, degradation, modern/historical fonts, and text-density ranges.
- Handwriting: group by writer and writing condition.
- Persian-specific fidelity: Arabic/Persian Yeh and Kaf, ZWNJ, digits, punctuation, diacritics, tatweel, mixed Persian-English, and exact entities/numbers.
- Document structure: multi-column RTL order, blocks, forms, tables, charts, and formulas only on pages with the required annotations.
- Component recognition: line/crop evaluation for recognizer-only models, separate from full-page pipeline results.

### Metrics

- Raw Unicode and versioned normalized CER/WER.
- Macro page CER, micro corpus CER, median and p90/p95 page CER.
- Insertions, deletions, and substitutions; empty, crash, timeout, truncation, and repetition rates.
- Paired page/document win/tie/loss and paired 95% intervals.
- Reading order, layout mAP/mAR, table TEDS, or formula CDM only where eligible.
- Downstream retrieval or critical-field extraction when that reflects the deployment goal.

Exit criteria:

- The manifest and dataset card are immutable and source-grounded.
- Every metric names its eligible annotated slice.
- The final set is disjoint from `smoke20` and all tuning/development data at the source-group level.
- Pilot variance supports the declared sample size and minimum meaningful difference.

## Phase 3 — Harden the reusable harness in parallel with Phase 2

Do the minimum Phase 0 engineering first; perform this broader cleanup while the large dataset is being built so it does not delay Phase 1 learning.

- Restore one canonical `src/persian_ocr/` package and add installed console entry points.
- Extract focused modules for dataset loading, Persian normalization, scoring, artifacts/provenance, preprocessing, and adapter protocol.
- Keep adapters responsible for pull/load/infer and model-specific structured output only.
- Add resumable `scripts/run_smoke20.py`, `scripts/build_screening_report.py`, and `scripts/run_large_bench.py` orchestration instead of long repeated CLI sequences.
- Keep heavyweight runtimes in model-specific `uv` optional groups and preserve the existing retry, logging, model storage, and raw-output conventions.
- Add executable smoke scripts for schema compatibility, path safety, zero-CER ordering, one-image inference, resume behavior, and failure propagation.
- Add README, license metadata, supported Python range, and package-data rules.

Exit criteria:

- Three comparison classes can emit one versioned artifact contract without importing one adapter from another.
- All code that affects a score is fingerprinted.
- Commands work from the repository root and an unrelated working directory.
- Interrupted runs resume without silently changing configuration.

## Phase 4 — Run the large benchmark on promoted systems only

- Freeze prompts, detectors, preprocessing, postprocessing, decoding, precision, quantization, package lock, and hardware before opening the test set.
- Run every model on the same eligible pages and preserve immutable raw outputs and error logs.
- Use paired document-level resampling; report practical ties when the interval crosses the predeclared meaningful-difference boundary.
- Use warm-up runs and repeated operational measurements. Report cold start, warm p50/p95 latency, throughput, peak RAM/VRAM, failure rate, and cost per page/success where relevant.
- Keep recognition quality, document-structure fidelity, reliability, and operations as separate views.
- Do not tune a model or prompt after inspecting sealed test results; a change requires a new benchmark version or untouched test set.

Exit criteria:

- Every comparison is compatible, paired, reproducible, and accompanied by uncertainty.
- No recognizer-only component is ranked against a full-page parser on tasks it cannot produce.
- No winner claim depends only on a point estimate or one aggregate score.

## Phase 5 — Real-world robustness mini-evals

Run controlled mini-evals on promoted systems for:

- blur, compression, rotation, skew, glare, shadows, curved pages, and screen photography;
- very long, dense, high-resolution, and near-blank pages;
- mixed scripts, Persian/Arabic/Latin numeral systems, rare Unicode, and redactions;
- headers/footers, multi-column order, tables/forms/formulas, and handwriting where supported;
- omission, duplication, invented correction, truncation, refusal, timeout, and repetition loops;
- intended production batch/concurrency and resource ceilings.

Community reports are useful for choosing these adversarial cases, but anecdotal reports must never determine model rank.

Exit criteria:

- Each finalist has a documented failure envelope, not only an average score.
- Operational thresholds match the intended deployment hardware and workload.
- Any downstream RAG or structured-extraction use case has its own outcome-based mini-eval.

## Phase 6 — Select, release, and monitor

- Select by use case and Pareto frontier; the best handwriting system, low-cost CPU system, and structured-document parser may be different models.
- Publish dataset/evaluator versions, configurations, uncertainty, slice results, hardware envelope, licenses, raw-output access policy, and known failure cases.
- Retain `smoke20-v1` permanently as a cheap adapter/regression sentinel.
- For drifting cloud APIs or updated model aliases, record date/region/API configuration and periodically rerun the sentinel without rewriting historical results.

## Initial screening waves

1. Repair data, then regenerate `tesseract_fas` and `ppocrv5_arabic_mobile_rec` as fixed controls.
2. Finish `easyocr_fa` as the next traditional neural baseline.
3. Add a Persian specialist such as `hezar_crnn_base_fa_v2` with an explicit detector/crop track.
4. Screen compact full-page candidates with explicit Persian evidence, including `surya_ocr_2` and `paddleocr_vl_1_6`.
5. Compare `qwen3_vl_2b_persian_arabic_ocr` with the exact `qwen3_vl_2b_instruct` base to measure the fine-tune rather than the family name.
6. Add a higher-cost ceiling such as `chandra_ocr_2` only when available hardware and expected information value justify it.

Stop implementing candidates when the shortlist has sufficient architectural and capability diversity. The catalog remains useful even when not every entry receives an adapter.

## Current-practice basis

- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) evaluates diverse document sources with separate text, table, formula, layout, reading-order, and attribute-specific views; its current repository describes 1,651 pages rather than a tiny universal ranking set.
- Arabic-focused [KITAB-Bench](https://github.com/mbzuai-oryx/KITAB-Bench) spans multiple domains and uses CER/WER plus structure-specific metrics, reinforcing the need for Persian domain and task slices.
- [OCRBench v2](https://arxiv.org/abs/2501.00321) includes a private evaluation set, supporting a sealed local holdout when public-model training data is opaque.
- [Real5-OmniDocBench](https://arxiv.org/abs/2603.04205) uses matched scan, warp, screen-photo, illumination, and skew conditions, a useful pattern for Phase 5 robustness tests.
- [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) cover dataset context, language, size, licensing, bias, and intended use; [Hub download guidance](https://huggingface.co/docs/huggingface_hub/en/guides/download) supports full commit revisions for model identity.
- [NVIDIA inference guidance](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/performance/best-practices.html) recommends warm-up, repeated measurements, and latency percentiles rather than one timing.
- Community discussions report literal-fidelity failures, invented corrections, omissions, and repetition loops in document VLMs. Use [Hacker News](https://news.ycombinator.com/item?id=43118514) and [Arabic OCR community testing](https://www.reddit.com/r/LocalLLaMA/comments/1qiyxl4/we_tested_every_vlm_for_arabic_document/) as anecdotal stress-test inputs only.

## Safe commands at the current checkpoint

These commands inspect the current state without running a benchmark against the damaged references:

```powershell
uv run python scripts/models_status.py --all
uv run python scripts/models_status.py --strict  # Expected to fail until stale artifacts are repaired.
uv run --extra easyocr python src/easyocr_fa.py --help
```

Do not run `scripts/merge_small_bench.py` or regenerate selection artifacts until Phase 0 data recovery and review are complete.

Target commands to implement during Phase 3:

```powershell
uv run python scripts/validate_smoke20.py --strict
uv run python scripts/run_smoke20.py --model easyocr_fa --device cpu
uv run python scripts/build_screening_report.py --input bench_runs/smoke20-v1
uv run python scripts/run_large_bench.py --shortlist shortlist.yaml --resume
```
