# Locked benchmark design

Build **669 human-verified base images**:

* **20 smoke images:** determine whether each model can perform Persian OCR at all.
* **49 public development images:** debug adapters, prompts, output extraction, and model configuration.
* **600 private test images:** calculate the official leaderboard.
* **630 derived robustness images:** generated from 30 private test samples; these reuse the same labels and do **not** count toward the 669 base samples.

This gives you **1,299 image files**, but only **669 independently labelled samples**.

For the common leaderboard, every image should contain **one Persian line or tightly cropped text region**. Full-page layout, reading order, tables, and formulas must remain a separate future leaderboard because several candidates are line recognizers. This matches your metric specification and the modular evaluation approach used by OmniDocBench.  ([GitHub][1])

---

# Phase 1 — Create the first 20 smoke images

These should be deliberately small but diagnostic.

| ID          | Content                                          |
| ----------- | ------------------------------------------------ |
| `smoke_001` | Simple clean Persian sentence                    |
| `smoke_002` | Clean serif Persian font                         |
| `smoke_003` | Clean sans-serif font                            |
| `smoke_004` | Small font                                       |
| `smoke_005` | Bold text with punctuation                       |
| `smoke_006` | Real scanned book line                           |
| `smoke_007` | Low-resolution JPEG line                         |
| `smoke_008` | Slightly skewed phone photograph                 |
| `smoke_009` | Neat Persian handwriting                         |
| `smoke_010` | Cursive handwriting                              |
| `smoke_011` | ZWNJ-heavy text: `می‌روم`, `خانه‌ها`, `گفته‌شده` |
| `smoke_012` | Joining and spacing challenge                    |
| `smoke_013` | Persian digits: `۱۲۳۴۵۶۷۸۹۰`                     |
| `smoke_014` | Latin digits inside Persian text                 |
| `smoke_015` | Date, time, price, percentage                    |
| `smoke_016` | Mixed Persian and English                        |
| `smoke_017` | Email, URL, abbreviation, parentheses            |
| `smoke_018` | Long dense Persian line                          |
| `smoke_019` | Historical font or bleed-through                 |
| `smoke_020` | Blank or non-text image                          |

ZWNJ and ordinary-space errors deserve explicit coverage because Persian joining, splitting, and نیم‌فاصله recognition are distinct segmentation problems. ([arXiv][2])

## Smoke-gate rules

Run all 82 candidates on these images.

A model qualifies for the 649-sample stage when it satisfies:

* Successful execution on at least **18 of 20** images.
* Non-empty output on at least **15 of 19** text images.
* Canonical CER Score of at least **10/100**.
* Output is meaningfully Persian rather than transliteration or unrelated text.
* No more than two catastrophic repetition outputs.

Define catastrophic repetition as either:

* Output longer than four times the reference, or
* The same character or n-gram repeated at least five times abnormally.

Assign one status:

| Status                   | Meaning                                           |
| ------------------------ | ------------------------------------------------- |
| `QUALIFIED`              | Run the complete benchmark                        |
| `BORDERLINE`             | Inspect configuration and rerun smoke once        |
| `FAILED_PERSIAN_SUPPORT` | Stop after smoke                                  |
| `NOT_APPLICABLE`         | Not an image-to-text recognition model            |
| `UNAVAILABLE`            | No usable weights, code, endpoint, or credentials |

The smoke score never contributes to the final leaderboard.

---

# Phase 2 — Create the remaining 649 samples

Split them into:

* **49 development samples:** labels visible.
* **600 private test samples:** labels hidden during inference.

## Exact composition

| Primary stratum                    | Development | Private test |   Total |
| ---------------------------------- | ----------: | -----------: | ------: |
| Clean modern printed Persian       |          12 |          148 |     160 |
| Real scanned documents             |           7 |           83 |      90 |
| Mobile-camera photographs          |           5 |           55 |      60 |
| Persian handwriting                |           9 |          111 |     120 |
| Mixed Persian/English              |           5 |           55 |      60 |
| Numeric-heavy text                 |           6 |           74 |      80 |
| Historical or naturally degraded   |           4 |           55 |      59 |
| Blank/non-text hallucination cases |           1 |           19 |      20 |
| **Total**                          |      **49** |      **600** | **649** |

These are exclusive **primary strata**. Attributes may overlap. For example, a handwritten line may also be ZWNJ-heavy and contain a date.

METATR likewise argues that realistic OCR evaluation needs variation in language, script, document condition, handwriting, and layout rather than a large homogeneous clean-print set. ([arXiv][3])

## Secondary attribute targets

Across the 649 samples, require at least:

| Attribute                            | Target |
| ------------------------------------ | -----: |
| Independent source documents         |    200 |
| Independent handwriting writers      |     30 |
| ZWNJ-positive lines                  |    180 |
| Annotated ZWNJ boundaries            |    300 |
| Numeric spans                        |    300 |
| Dates and times                      |     60 |
| Prices and currencies                |     60 |
| Percentages                          |     40 |
| Phone, ID, postal, or serial numbers |     80 |
| Persian-digit spans                  |    150 |
| Latin-digit spans                    |     60 |
| Arabic-Indic-digit spans             |     20 |
| Mixed Persian/Latin lines            |     60 |
| Blank/non-text crops                 |     20 |
| Distinct printed font families       |  15–20 |

Persian handwriting collections illustrate why writer identity matters: PHTD contains thousands of lines from dozens of writers, and later work uses page- or writer-aware splits to prevent leakage. ([arXiv][4])

## Source rules

* Use at least **500 real samples**.
* Use no more than **149 synthetic samples**.
* Limit each source document to approximately four line crops.
* Never place lines from one source document in both development and test.
* Never place one handwriting writer in both development and test.
* Do not take adjacent near-identical lines from the same page merely to increase the count.
* Store the uncropped source page privately for audit purposes.

Synthetic examples are useful for controlled fonts, rare ZWNJ constructions, and numeric formats, but should not dominate. IDPL-PFOD results demonstrate that OCR performance can differ substantially across fonts, blur, distortion, and textured backgrounds. ([pcdp.qut.ac.ir][5])

---

# Phase 3 — Prepare every image correctly

Use lossless PNG for the base dataset.

Each crop must:

* Contain one line or one tightly bounded text region.
* Include a small margin around the text.
* Avoid cutting dots, descenders, ascenders, or punctuation.
* Preserve the original color or grayscale appearance.
* Avoid resizing the stored master image.
* Have enough resolution for a human to transcribe confidently.
* Contain no model-generated overlays or bounding boxes.

Assign identifiers such as:

```text
fa_smoke_0001
fa_dev_0001
fa_test_0001
```

Calculate and store a SHA-256 checksum for every image.

---

# Phase 4 — Create the manifest

Use JSONL as the canonical format. CSV can be generated from it for inspection.

```json
{
  "sample_id": "fa_test_000001",
  "split": "test",
  "image_path": "images/test/fa_test_000001.png",
  "sha256": "IMAGE_SHA256",
  "source_document_id": "doc_00291",
  "writer_id": null,
  "reference_raw": "امروز ساعت ۱۲:۳۰ به خانه می‌روم.",
  "primary_stratum": "numeric_heavy",
  "attributes": [
    "zwnj",
    "persian_digits",
    "time",
    "punctuation"
  ],
  "origin": "real",
  "numeric_spans": [
    {
      "start": 11,
      "end": 16,
      "type": "time",
      "raw": "۱۲:۳۰",
      "canonical_value": "12:30",
      "preserve_leading_zero": false
    }
  ],
  "is_blank": false,
  "annotator_status": "approved",
  "adjudicated": true
}
```

Do not manually maintain `reference_canonical`. Generate it from `reference_raw` during scoring using the frozen `canonical_v1` function. Otherwise the raw and canonical references will eventually drift apart.

---

# Phase 5 — Establish the transcription contract

Annotators must transcribe only what is visible.

Preserve:

* Ordinary spaces
* ZWNJ, U+200C
* Punctuation
* Repeated spaces when visibly intentional
* Line-internal quotation marks
* Persian, Arabic-Indic, and Latin digit forms
* Arabic/Persian character variants according to your reference policy
* Spelling errors present in the image

Never:

* Correct spelling
* Improve punctuation
* Insert missing ZWNJ
* Rewrite dates
* Convert digits manually
* Expand abbreviations
* Guess unreadable characters
* Add `[illegible]` to official references

If a sample cannot be transcribed confidently, remove it from the benchmark and replace it.

## Important Unicode limitation

For raster-only images, visually identical forms such as Arabic `ي` versus Persian `ی` may not be inferable from pixels.

Use this policy:

* When a trustworthy original digital text exists, preserve its exact codepoints.
* For raster-only material, use modern Iranian Persian codepoints consistently.
* Interpret Strict CER partly as an output-convention metric, not necessarily as recoverable visual information.

## Annotation interface

Your review interface should render invisible characters visibly:

```text
Ordinary space → [SP]
ZWNJ           → [ZWNJ]
Tab            → [TAB]
```

Example:

```text
می[ZWNJ]روم[SP]به[SP]خانه
```

This will prevent silent annotation mistakes.

---

# Phase 6 — Annotation workflow

OCRBench manually verified and corrected all of its benchmark answers; benchmark labels must be treated as a separate engineering product, not as disposable model output. ([GitHub][6])

Use this pipeline:

1. A strong teacher model produces the first transcription.
2. A human reviews every character against the image.
3. Automated validation checks Unicode, numeric spans, spaces, and ZWNJ.
4. Every handwriting, ZWNJ-heavy, numeric-heavy, and difficult sample receives a second review.
5. Randomly second-review at least 20% of the remaining clean samples.
6. Resolve disagreements manually.
7. Freeze the approved reference.
8. Compute its image and annotation checksums.

If only one human is available, perform the second review in a separate delayed session without displaying the first-pass decisions.

---

# Phase 7 — Generate the robustness set

Choose **30 clean anchor samples from the private test set**:

| Anchor type                  |  Count |
| ---------------------------- | -----: |
| Modern print                 |     14 |
| Handwriting                  |      6 |
| Numeric-heavy                |      4 |
| Mixed Persian/English        |      3 |
| Form-like or structured line |      3 |
| **Total**                    | **30** |

Generate seven corruption families at three severity levels:

1. Gaussian blur
2. Motion blur
3. JPEG compression
4. Downsampling and upscaling
5. Gaussian or sensor noise
6. Rotation/skew or mild perspective
7. Shadow, illumination gradient, or low contrast

This produces:

[
30 \times 7 \times 3 = 630
]

derived images.

Each derived record should contain:

```json
{
  "sample_id": "fa_test_000041__gaussian_blur_s2",
  "parent_sample_id": "fa_test_000041",
  "corruption": "gaussian_blur",
  "severity": 2,
  "seed": 12345,
  "parameters": {
    "sigma": 1.4
  }
}
```

All corruption parameters and random seeds must be frozen. Current robustness projects similarly use paired clean and perturbed images so performance loss can be attributed to controlled visual degradation. ([GitHub][7])

---

# Phase 8 — Build the repository

```text
persian-ocr-benchmark/
├── configs/
│   ├── models.yaml
│   ├── prompts.yaml
│   └── benchmark.yaml
├── data/
│   ├── smoke/
│   ├── dev/
│   ├── test/
│   ├── robustness/
│   ├── manifest.jsonl
│   └── robustness_manifest.jsonl
├── benchmark/
│   ├── adapters/
│   │   ├── base.py
│   │   ├── tesseract_adapter.py
│   │   ├── paddle_adapter.py
│   │   ├── easyocr_adapter.py
│   │   ├── hf_line_adapter.py
│   │   ├── hf_vlm_adapter.py
│   │   ├── pipeline_adapter.py
│   │   ├── commercial_api_adapter.py
│   │   └── custom_adapter.py
│   ├── normalization/
│   │   └── canonical_v1.py
│   ├── metrics/
│   │   ├── alignment.py
│   │   ├── cer_wer.py
│   │   ├── chrfpp.py
│   │   ├── spacing_f1.py
│   │   ├── numeric_spans.py
│   │   ├── omission_hallucination.py
│   │   └── robustness.py
│   ├── validate_dataset.py
│   ├── run_model.py
│   ├── score_model.py
│   ├── bootstrap.py
│   └── build_leaderboard.py
├── runs/
├── leaderboard/
├── docs/
└── pyproject.toml
```

Recommended dependencies:

```bash
uv add pillow opencv-python numpy pandas pyarrow pydantic typer tqdm pyyaml
uv add rapidfuzz jiwer sacrebleu scipy
```

Pin everything in `uv.lock`.

---

# Phase 9 — Convert the 82-model CSV into a registry

Do not write 82 completely independent scripts. Your CSV contains several model families, so build reusable adapters.

Add these fields to every model:

```yaml
id: qwen3_vl_2b_instruct
display_name: Qwen3-VL-2B-Instruct
track: general_line_recognition
runner: hf_vlm
repository: Qwen/Qwen3-VL-2B-Instruct
revision: EXACT_COMMIT_HASH
availability: ready
requires_api_key: false
input_mode: line_image
prompt_id: exact_transcription_v1
max_new_tokens: 512
temperature: 0
dtype: bfloat16
device: cuda
adapter_version: 1.0.0
```

## Tracks

Every one of the 82 candidates receives a leaderboard row, but not necessarily a general rank.

| Track                      | Examples of system type                                |
| -------------------------- | ------------------------------------------------------ |
| `general_line_recognition` | Models that transcribe arbitrary Persian text lines    |
| `handwriting_specialist`   | Models intended only for handwriting                   |
| `digit_specialist`         | Models intended only for digits                        |
| `pipeline`                 | Framework plus a fully specified OCR backend           |
| `not_ocr`                  | Models that consume OCR text rather than generating it |
| `unavailable`              | No reproducible checkpoint or endpoint                 |

A framework is not a reproducible model configuration. Record pipelines explicitly, for example:

```text
Docling + Tesseract fas
Marker + Surya OCR 2
MMOCR + exact checkpoint name
```

Your existing metric design already correctly treats non-recognition document models as ineligible rather than assigning them artificial zero scores. 

---

# Phase 10 — Freeze the VLM prompt

For general VLMs, use one semantic instruction:

```text
Transcribe the visible text exactly.

Preserve:
- ordinary spaces,
- U+200C zero-width non-joiners,
- punctuation,
- line order,
- Persian, Arabic-Indic, and Latin digit forms.

Do not explain, translate, correct, summarize, or wrap the answer in Markdown.
Return only the transcription.
```

Model-specific task tokens required by the official implementation are allowed, but the semantic instruction must remain unchanged.

Use the 49 development images to detect:

* Markdown wrappers
* Prompt echoing
* Truncation
* Incorrect task tokens
* Wrong image resizing
* Unwanted explanations
* Batch-order bugs
* Encoding errors

Once the adapter passes development, freeze:

* Prompt
* Processor version
* Model revision
* Decoding settings
* Maximum output length
* Image preprocessing
* Dependency lockfile

Do not tune prompts after viewing private-test labels.

---

# Phase 11 — Standardize prediction output

Every prediction should be one JSONL record:

```json
{
  "run_id": "qwen3_vl_2b_instruct__2026_07_14",
  "model_id": "qwen3_vl_2b_instruct",
  "model_revision": "COMMIT_HASH",
  "adapter_version": "1.0.0",
  "sample_id": "fa_test_000001",
  "raw_prediction": "امروز ساعت ۱۲:۳۰ به خانه می‌روم.",
  "runtime_ms": 844,
  "peak_vram_mb": 6210,
  "input_width": 1280,
  "input_height": 128,
  "status": "success",
  "retry_count": 0,
  "prompt_hash": "HASH",
  "config_hash": "HASH"
}
```

Rules:

* Save output before normalization.
* Never spell-correct predictions.
* Never remove explanations or Markdown during scoring.
* Retry only infrastructure failures.
* Do not retry a bad OCR answer.
* After retry exhaustion, treat the prediction as empty.
* Cache completed predictions so interrupted runs resume safely.
* Preserve API raw responses in a private audit directory.

JiWER’s defined handling for empty references is particularly useful for the blank hallucination subset. ([GitHub][8])

---

# Phase 12 — Run all 82 models

## Stage A: Registry audit

For every candidate, determine:

* Is it actually image-to-text?
* Is there a public checkpoint?
* Is its tokenizer capable of Persian output?
* Does it need a detector?
* Does it require line crops?
* Does it require an API key?
* Is there a concrete reproducible configuration?
* What is the official model revision?

Output:

```text
leaderboard/model_registry_audit.csv
```

## Stage B: Smoke benchmark

Run:

[
82 \times 20 = 1,640
]

inferences.

Output:

```text
runs/{model_id}/smoke_predictions.jsonl
runs/{model_id}/smoke_score.json
```

After one adapter-correction rerun, freeze the qualification result.

## Stage C: Development run

Run qualified models on 49 development samples.

Purpose:

* Confirm output extraction.
* Find token truncation.
* Check memory requirements.
* Lock model configuration.
* Validate commercial endpoints.
* Measure approximate runtime.

Development results are not included in the leaderboard.

## Stage D: Private test

Run every qualified model once on all 600 private images.

For a qualified-model count (Q):

[
600Q
]

test inferences are required.

## Stage E: Robustness

Run every qualified model on 630 derived images:

[
630Q
]

robustness inferences.

Therefore, after smoke:

[
Q(49+600+630)=1,279Q
]

inferences are required per qualified-model collection.

For nondeterministic commercial APIs, run three complete test passes or label single-run scores as provisional. Your metric specification already requires reporting mean and variance for nondeterministic systems. 

---

# Phase 13 — Implement the metrics

Use the exact weights in `benchmark_metrics.md`:

| Metric                      | Weight |
| --------------------------- | -----: |
| Canonical CER Score         |    25% |
| Strict CER Score            |    10% |
| Canonical WER Score         |    10% |
| chrF++                      |     8% |
| Exact Line Accuracy         |     7% |
| ZWNJ and Spacing F1         |    10% |
| Numeric Span Exact Accuracy |    10% |
| Omission Score              |     7% |
| Hallucination Score         |     7% |
| Robustness Retention        |     6% |



## Implementation ownership

| Metric        | Implementation                                    |
| ------------- | ------------------------------------------------- |
| CER/WER       | Shared deterministic edit alignment               |
| Omission      | Deletions from the same character alignment       |
| Hallucination | Insertions from the same character alignment      |
| chrF++        | SacreBLEU with character order 6 and word order 2 |
| Spacing/ZWNJ  | Custom boundary alignment                         |
| Numeric spans | Custom annotated-span matcher                     |
| Robustness    | Clean/degraded pair scoring                       |

OCR-D defines CER and WER from substitution, insertion, and deletion counts, while JiWER exposes these alignment components programmatically. ([ocr-d.de][9])

SacreBLEU provides reproducible chrF scoring and bootstrap significance tooling. ([GitHub][10])

## Blank-image treatment

Blank samples should:

* Contribute to Hallucination Score.
* Contribute to blank exact-output diagnostics.
* Be excluded from normal CER, WER, chrF++, ZWNJ, numeric, and omission denominators.

---

# Phase 14 — Calculate domain-balanced results

For the first nine metrics, calculate per-domain results.

Recommended fixed domain weights:

| Domain                | Weight |
| --------------------- | -----: |
| Clean printed         |    20% |
| Real scans            |    15% |
| Mobile photographs    |    10% |
| Handwriting           |    20% |
| Mixed Persian/English |    10% |
| Numeric-heavy         |    15% |
| Historical/degraded   |    10% |

Blank images only affect hallucination reporting.

For sparse event metrics:

* Compute ZWNJ/spacing F1 globally over boundary events.
* Compute numeric accuracy globally over annotated spans.
* Publish per-domain and per-type breakdowns as diagnostics.
* Compute robustness globally over the clean/degraded pairs.

Freeze these weights before running the first private test.

---

# Phase 15 — Bootstrap confidence intervals

Use at least **2,000 paired bootstrap replicates**.

Resample clusters rather than independent lines:

* Printed lines: `source_document_id`
* Handwriting: `writer_id`
* Robustness: `parent_sample_id`
* Synthetic examples: generated source sentence or template group

For each replicate:

1. Sample clusters with replacement.
2. Recalculate all ten metrics.
3. Recalculate the composite score.
4. Store pairwise score differences between models.

Report:

* Composite score
* 95% confidence interval
* Pairwise difference interval
* Statistical tier

SacreBLEU implements paired bootstrap resampling for system comparisons; your implementation must adapt this to source-document and writer clusters. ([GitHub][10])

Do not claim one model outranks another merely because it scores `88.42` instead of `88.31` when the paired difference interval includes zero.

---

# Phase 16 — Build the leaderboard

## Main leaderboard columns

```text
rank
statistical_tier
model_id
model_name
status
persian_ocr_score
ci_low
ci_high
canonical_cer_score
strict_cer_score
canonical_wer_score
chrfpp
exact_line_accuracy
spacing_zwnj_f1
numeric_span_accuracy
omission_score
hallucination_score
robustness_score
failure_rate
mean_latency_ms
p95_latency_ms
peak_vram_mb
api_cost
model_revision
adapter_version
prompt_hash
```

## Publish separate views

1. **General Persian OCR leaderboard**
2. **Handwriting-specialist leaderboard**
3. **Digit-specialist leaderboard**
4. **Smoke support matrix for all 82 candidates**
5. **Per-domain breakdown**
6. **Robustness breakdown by corruption**
7. **Efficiency leaderboard**
8. **Failure and availability table**
9. **Reproducibility metadata**

All 82 candidates appear somewhere, but only general arbitrary-text recognizers receive a rank in the main table.

## Ranking eligibility

A model receives a general rank only when:

* It passed the Persian smoke gate.
* It completed at least 99% of private-test inputs after retries.
* Its configuration is reproducible.
* It outputs arbitrary Persian text rather than only digits or one narrow field type.
* Its raw predictions are available for audit.

A failed inference becomes an empty prediction rather than being silently excluded. Otherwise models could improve their score by failing on difficult samples.

---

# Phase 17 — Dataset validation gates

Do not run the expensive models until all these pass:

### Image validation

* Exactly 669 base images.
* Exactly 20 smoke, 49 development, and 600 private test.
* Every file has a matching checksum.
* No duplicate SHA-256.
* No high-similarity perceptual duplicates across splits.
* No clipped text.
* No unreadable references.

### Annotation validation

* Every nonblank sample has non-empty `reference_raw`.
* Every blank sample has an empty reference.
* All numeric offsets point to the correct substring.
* No forbidden control characters.
* Every ZWNJ is explicitly reviewed.
* Development and test sources do not overlap.
* Handwriting writers do not overlap.
* Every hard sample has a second review.

### Metric validation

Create fixtures for:

```text
يادگیری ↔ یادگیری
كتاب ↔ کتاب
می روم ↔ می‌روم
میروم ↔ می‌روم
۱۲۳ ↔ 123
۱۲۳۴ ↔ ۱۲۳
سلام ↔ سلام سلام سلام
"" ↔ متن ساختگی
```

For identical reference and prediction, every applicable score must equal exactly `100.0`.

### End-to-end validation

Before running all 82:

1. Run Tesseract Persian.
2. Run EasyOCR Persian.
3. Run one Persian CRNN.
4. Run one general VLM.
5. Run one commercial API.
6. Regenerate the complete leaderboard.
7. Repeat one deterministic model and confirm byte-identical predictions.

Published Persian testing found notably different Tesseract and EasyOCR performance across a diverse synthetic benchmark, so they are useful pipeline sanity baselines—but your scores should not be expected to reproduce that study’s exact numbers because your dataset composition is different. ([pcdp.qut.ac.ir][5])

---

# Final execution order

1. Freeze the 669-sample design.
2. Build and annotate the 20 smoke images.
3. Audit all 82 candidate configurations.
4. Run the 1,640 smoke inferences.
5. Label every candidate `QUALIFIED`, `FAILED`, `N/A`, or `UNAVAILABLE`.
6. Collect and annotate the 49 development samples.
7. Collect and annotate the 600 private-test samples.
8. Validate source, writer, and duplicate separation.
9. Generate the 630 paired robustness variants.
10. Implement and verify all ten metrics.
11. Run five representative systems end-to-end.
12. Freeze `dataset_v1.0.0`, prompts, adapters, and dependencies.
13. Run qualified models on development.
14. Freeze their final configurations.
15. Run the 600-sample private test.
16. Run the 630-image robustness test.
17. Calculate metrics and cluster-bootstrap intervals.
18. Generate the main, specialist, robustness, and efficiency leaderboards.
19. Publish smoke and development data.
20. Keep the 600 private labels hidden until a future benchmark refresh.

[1]: https://github.com/opendatalab/OmniDocBench?utm_source=chatgpt.com "opendatalab/OmniDocBench: [CVPR 2025] A ..."
[2]: https://arxiv.org/abs/2010.00287?utm_source=chatgpt.com "Joint Persian Word Segmentation Correction and Zero-Width Non-Joiner Recognition Using BERT"
[3]: https://arxiv.org/html/2605.26712v1?utm_source=chatgpt.com "METATR: A Multilingual, Evolving Benchmark for Automatic ..."
[4]: https://arxiv.org/html/2605.02089v2?utm_source=chatgpt.com "Cross-Lingual Learning within Arabic Script for Low- ..."
[5]: https://pcdp.qut.ac.ir/article_722202_daedbb20a5d8c7aa91eae2bf3715cb4b.pdf?utm_source=chatgpt.com "Empirical Evaluation of Well-known Farsi OCR Engines on ..."
[6]: https://github.com/yuliang-liu/multimodalocr?utm_source=chatgpt.com "Yuliang-Liu/MultimodalOCR: On the Hidden Mystery ..."
[7]: https://github.com/pasterinjlu/OCR-Reasoning-Robust?utm_source=chatgpt.com "pasterinjlu/OCR-Reasoning-Robust"
[8]: https://github.com/jitsi/jiwer?utm_source=chatgpt.com "jitsi/jiwer: Evaluate your speech-to-text system with ..."
[9]: https://ocr-d.de/en/spec/ocrd_eval.html?utm_source=chatgpt.com "Quality Assurance in OCR-D"
[10]: https://github.com/mjpost/sacrebleu?utm_source=chatgpt.com "mjpost/sacrebleu: Reference BLEU implementation that ..."
