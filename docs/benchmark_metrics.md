# Recommended benchmark design

Your ten metrics are a strong basis for a **Persian text-recognition leaderboard**. CER/WER remain standard because they are reproducible and derived from one Levenshtein alignment, but they conceal whether errors came from deletion, insertion, substitution, spacing, or catastrophic repetition. That is exactly why your Persian-specific and failure-oriented metrics add value. ([OCR-D][1])

I would make four important refinements:

1. Calculate Canonical CER, Omission Score, and Hallucination Score from **one shared deterministic character alignment**.
2. Use a deliberately minimal, versioned Persian canonicalizer—not Hazm spell correction or an LLM.
3. Measure robustness relative to each model’s clean accuracy, incorporating both average and worst-case retention.
4. Make the common benchmark a **line/text-region recognition track**. Keep full-page layout, reading order, tables, and formulas separate.

OmniDocBench follows the same general principle by evaluating text recognition, layout detection, reading order, tables, and formulas as distinct components rather than collapsing them into one ambiguous number. ([GitHub][2])

---

## 1. Define the common task precisely

For the leaderboard shared by all 82 candidates, the official input unit should be:

> **One image containing one Persian text line or tightly cropped text region, with one exact reference transcription.**

This lets you compare:

* Classical OCR engines
* CRNN and transformer line recognizers
* General VLMs
* Document VLMs
* Commercial OCR APIs

A page-level model can process the same crop. A line recognizer cannot fairly process a full page containing layout, tables, headers, and multiple columns.

Maintain three tracks:

| Track                    | Eligible systems              | Evaluated output                        |
| ------------------------ | ----------------------------- | --------------------------------------- |
| Persian Line Recognition | All real image-to-text models | Exact line transcription                |
| Persian Page Text        | Full-page OCR/VLM systems     | Text content and line segmentation      |
| Persian Document Parsing | Document parsers only         | Layout, reading order, tables, formulas |

Models such as LayoutLMv3, LayoutXLM, and DocFormer should be marked **ineligible for recognition**, not assigned zero accuracy, because they consume OCR information rather than directly transcribing page pixels. Configurable frameworks such as MMOCR, Docling, or Marker must be entered as concrete pipelines, for example `docling+tesseract_fas`, rather than scored as abstract frameworks.

---

## 2. Build the evaluation dataset

Use both real and controlled synthetic material. A Persian OCR study using IDPL-PFOD found substantial differences between clean, textured, blurred, and distorted subsets, while a recent Devanagari benchmark found that very strong synthetic-image results could collapse on real scans. ([pcdp.qut.ac.ir][3])

Each sample should contain:

```json
{
  "sample_id": "line_000001",
  "source_document_id": "doc_0017",
  "image_path": "images/line_000001.png",
  "reference_raw": "می‌روم ساعت ۱۲:۳۰",
  "reference_canonical": "می‌روم ساعت 12:30",
  "numeric_spans": [
    {
      "start": 11,
      "end": 16,
      "type": "time",
      "value": "12:30",
      "preserve_leading_zero": false
    }
  ],
  "domain": "handwritten_form",
  "script_mix": ["Persian", "digits"],
  "quality": "clean",
  "pair_id": "content_00821",
  "corruption": null,
  "severity": 0
}
```

### Required dataset strata

Include enough examples from:

* Modern printed Persian
* Scanned books and documents
* Mobile photographs
* Historical or degraded print
* Handwriting
* Mixed Persian and Latin text
* Numeric-heavy forms, receipts, tables, and IDs
* ZWNJ-heavy morphology
* Informal Persian
* Multiple fonts and font sizes
* Bold, italic, underlined, and low-contrast text
* Blank or non-text crops for hallucination auditing

METATR similarly emphasizes varied handwriting, print, institutional documents, layouts, and real-world conditions rather than treating OCR as one homogeneous dataset. ([arXiv][4])

Split and sample by `source_document_id`, not by individual line. Lines from one page or document are correlated and can otherwise leak near-identical style, font, and scanning conditions between development and private test sets.

Keep:

* A public development set
* A small public smoke set
* A private leaderboard set
* A private robustness set
* A hidden refresh set for future leaderboard versions

---

## 3. Freeze Persian normalization as `canonical_v1`

Persian electronic text contains visually similar but Unicode-distinct Arabic and Persian characters, along with inconsistent ZWNJ and spacing. Persian NLP projects such as Hazm and ParsiNorm explicitly address these normalization problems, while the Persian ZWNJ paper treats space insertion, deletion, joining, and ZWNJ recognition as a joint segmentation problem. ([GitHub][5])

### Strict representation

For Strict CER:

* Preserve all Unicode characters.
* Preserve Persian versus Arabic digits.
* Preserve ordinary space and ZWNJ.
* Preserve punctuation and diacritics.
* Normalize only transport artifacts such as `CRLF → LF`.
* Remove only a terminal newline inserted by the inference runner.

Do **not** strip:

* Leading model commentary
* Markdown code fences
* Quotation marks
* “Here is the transcription:”
* Duplicate lines

Those are real hallucination/insertion errors.

### Canonical representation

For Canonical CER, WER, chrF++, exact-line accuracy, and edit-based failure metrics:

1. Apply Unicode NFC.
2. Fold Arabic presentation forms into normal characters.
3. Apply explicit Persian mappings:

```text
ي → ی
ى → ی
ك → ک
```

4. Select one representation for Persian heh-with-hamza forms:

```text
ۀ → هٔ
```

5. Remove tatweel:

```text
ـ → ""
```

6. Remove bidi formatting controls, but **never remove ZWNJ**.
7. Convert unusual ordinary whitespace characters to U+0020.
8. Preserve repeated spaces; do not collapse them.
9. Convert Persian, Arabic-Indic, and Latin digits to one canonical digit family for semantic text scoring.
10. Preserve punctuation, ZWNJ, spelling errors, and word segmentation exactly.

Do not run:

* Spell correction
* Contextual normalization
* LLM cleanup
* Word joining or splitting correction
* Automatic insertion of ZWNJ
* Date or number rewriting that changes the represented value

Hazm and ParsiNorm are useful as references and test-case generators, but the leaderboard normalizer should be a small transparent mapping with unit fixtures. Otherwise the normalizer may silently repair model errors. ([GitHub][5])

Version the rules:

```text
persian_normalization_version = "canonical_v1.0.0"
```

Any rule change requires recalculating every historical result.

---

# 4. Exact implementation of the ten metrics

Let the canonical character alignment produce:

* (N_c): number of reference characters
* (S_c): substitutions
* (D_c): deletions
* (I_c): insertions

Use one Wagner–Fischer/Levenshtein alignment with a frozen tie-breaking rule. JiWER uses minimum-edit-distance scoring through RapidFuzz and supports both CER and WER; OCR-D specifies the same insertion, deletion, and substitution decomposition. ([GitHub][6])

## 1. Canonical CER — 25%

[
CER_c=\frac{S_c+D_c+I_c}{N_c}
]

[
CanonicalCERScore=100\max(0,1-CER_c)
]

Use corpus-level totals:

[
CER_c=
\frac{\sum S_c+\sum D_c+\sum I_c}
{\sum N_c}
]

Do not average per-line CER for the official result. A one-character line should not have the same influence as a 100-character line.

Also publish median line CER and the 95th percentile as diagnostics.

---

## 2. Strict CER — 10%

Run another alignment over the strict raw strings:

[
CER_s=\frac{S_s+D_s+I_s}{N_s}
]

[
StrictCERScore=100\max(0,1-CER_s)
]

The difference between Canonical and Strict CER becomes a useful diagnostic:

[
UnicodePenalty=CanonicalCERScore-StrictCERScore
]

A large gap means the model reads the text visually but outputs incorrect Arabic/Persian Unicode forms.

---

## 3. Canonical WER — 10%

Tokenize the canonical text deterministically:

* Keep ZWNJ inside words.
* Split on ordinary whitespace.
* Treat punctuation as separate tokens.
* Keep numeric spans as tokens.
* Do not use contextual Persian token correction.

Then calculate:

[
WER=\frac{S_w+D_w+I_w}{N_w}
]

[
CanonicalWERScore=100\max(0,1-WER)
]

Freeze the tokenizer version alongside the normalizer.

---

## 4. chrF++ — 8%

Use SacreBLEU’s chrF implementation with:

```text
character_order = 6
word_order      = 2
beta            = 2
whitespace      = false
```

Adding word unigrams and bigrams to character n-grams is what distinguishes chrF++ from ordinary chrF. The original work found that the combination improved agreement with human assessment, and SacreBLEU exposes it using `word_order=2`. 

Use canonical text and record the complete SacreBLEU signature and version.

Using `whitespace=false` is appropriate here because spacing already receives a dedicated 10% metric. It prevents ordinary spaces from being rewarded twice inside chrF character n-grams, while word n-grams still capture word-level similarity.

---

## 5. Exact Line Accuracy — 7%

[
ELA=100\times
\frac{#{i:canonical(ref_i)=canonical(pred_i)}}
{#lines}
]

This is intentionally unforgiving. One incorrect punctuation mark, digit, space, or ZWNJ makes the line incorrect.

Publish strict exact-line accuracy separately as an unweighted diagnostic.

---

## 6. ZWNJ and Spacing F1 — 10%

Do not calculate this by simply counting space characters.

### Boundary-based evaluation

For each reference and prediction:

1. Remove ordinary spaces and ZWNJs to create base-character sequences.
2. Align the two base-character sequences.
3. For every gap between adjacent aligned base characters, assign one label:

```text
NONE
SPACE
ZWNJ
INVALID_OR_REPEATED
```

4. Compare reference and prediction labels.
5. Calculate corpus-level F1 separately for `SPACE` and `ZWNJ`.

[
SpacingScore=50(F1_{SPACE}+F1_{ZWNJ})
]

where each F1 is between 0 and 1.

Example:

```text
Reference:  می‌روم
Prediction: می روم
```

This produces:

* One false negative for `ZWNJ`
* One false positive for `SPACE`

For:

```text
Reference:  دانشگاه تهران
Prediction: دانشگاهتهران
```

it produces one false negative for `SPACE`.

Calculate F1 over the whole corpus, not individually per line. Many lines contain no ZWNJ, and per-line averaging would produce unstable or undefined results. Persian segmentation research confirms that ordinary spaces, joining, splitting, and ZWNJ recognition are closely related but distinct error categories. ([arXiv][7])

Also publish:

* `ZWNJ_precision`
* `ZWNJ_recall`
* `space_precision`
* `space_recall`
* ZWNJ-to-space substitution count
* ZWNJ deletion count

---

## 7. Numeric Span Exact Accuracy — 10%

Numeric spans must be annotated in the reference instead of discovered only through regex after inference.

Recommended types:

```text
integer
decimal
currency
percentage
date
time
phone
national_id
postal_code
serial_number
measurement
```

Normalize values type-by-type:

* Fold Persian, Arabic, and Latin digit families.
* Normalize decimal and thousands separators.
* Normalize percent signs and surrounding spaces.
* Preserve leading zeros for phones, IDs, postal codes, and serial numbers.
* Do not treat `1403/02/04` and `04/02/1403` as equivalent.
* Do not rewrite Persian-calendar dates into Gregorian dates.

Use one-to-one span matching based on text alignment and surrounding anchors. A span is correct only when:

* Its type matches.
* Its normalized value matches exactly.
* All required digits are present and ordered correctly.

To penalize invented numeric spans:

[
NumericScore=
100\times
\frac{ExactMatches}
{ReferenceSpans+UnmatchedExtraPredictionSpans}
]

A wrong predicted value matched to a reference is already included as an incorrect reference span; it should not be counted twice.

ParsiNorm’s coverage of numbers, dates, times, telephone numbers, and currencies is useful for constructing normalization fixtures and adversarial examples. ([GitHub][8])

---

## 8. Omission Score — 7%

Use the deletion count from the same canonical character alignment used for CER:

[
DeletionRate=\frac{D_c}{N_c}
]

[
OmissionScore=100\max(0,1-DeletionRate)
]

Do not run a second fuzzy matcher. Separate alignments can assign the same ambiguous error differently and make CER, omission, and hallucination internally inconsistent.

Also report:

* Entirely omitted line rate
* Word deletion rate
* Longest consecutive omitted span
* Percentage of lines losing more than 25% of reference characters

Character deletions are a direct approximation of missed reference content, while insertions approximate extra or hallucinated content. ([arXiv][9])

---

## 9. Hallucination Score — 7%

Use insertion count from that same alignment:

[
InsertionRate=\frac{I_c}{N_c}
]

[
HallucinationScore=100\max(0,1-InsertionRate)
]

This captures:

* Introduced words
* Preambles
* Duplicated lines
* Repetition loops
* Invented numeric values
* Wrong-script additions
* Markdown wrappers

For an empty reference:

```text
reference = ""
```

define:

```text
prediction == ""  → Hallucination Score = 100
prediction != ""  → Hallucination Score = 0
```

JiWER explicitly defines meaningful empty-reference behavior, which is useful for silence and hallucination testing. Modern multilingual OCR studies also report wrong-script generation and repetition as distinct real-world VLM failure modes. ([GitHub][6])

Publish these additional diagnostics:

* Output/reference length ratio
* Repeated 3-gram ratio
* Duplicate-line rate
* Blank-image hallucination rate
* Catastrophic insertion rate
* 95th-percentile insertion rate

The recent Devanagari benchmark reported extreme repetition loops that made mean error statistics misleading, which supports publishing median and catastrophic-rate diagnostics alongside the weighted score. ([GitHub][10])

---

## 10. Robustness Accuracy Retention — 6%

Do not use degraded accuracy alone. A model with 95% clean accuracy dropping to 75% is not equivalent to a model starting at 76% and dropping to 75%.

For every original clean image, create paired degradations of exactly the same content.

Recommended corruption families:

1. Gaussian and motion blur
2. Sensor, Gaussian, and salt-and-pepper noise
3. JPEG compression
4. Low resolution and downsampling
5. Rotation, skew, and perspective distortion
6. Shadows, illumination gradients, and low contrast
7. Background texture or show-through

Use three calibrated severity levels per family. Severity 3 must remain human-readable; otherwise the benchmark measures destroyed information rather than OCR robustness.

Let:

[
A_0=\max(0,1-CER_{clean})
]

For each corruption condition (c):

[
r_c=\min\left(1,\frac{A_c}{\max(A_0,\epsilon)}\right)
]

Then:

[
RCR=\operatorname{mean}_c(r_c)
]

[
WCR=\min_c(r_c)
]

I recommend the official robustness score:

[
RobustnessScore=100\sqrt{RCR\times WCR}
]

This rewards average retention while penalizing a model with one catastrophic weakness.

OCR-Robust similarly separates clean accuracy, relative corrupted performance, and worst-case retention, then combines capability and robustness geometrically. Because your first nine metrics already represent clean capability, including clean accuracy again inside this 6% component would double-count it; therefore the adapted formula above uses only average and worst-case retention. ([arXiv][11])

Keep the corruption generation code and exact parameters in the repository. OCR-Robust publishes its perturbation and evaluation implementation for this purpose. ([GitHub][12])

---

# 5. Final composite calculation

All component values must be scores from 0 to 100:

[
\begin{aligned}
PersianOCRScore={}&
0.25C_{CER}+
0.10S_{CER}+
0.10W_{canonical}+\
&0.08chrF+++
0.07ELA+
0.10SpacingF1+\
&0.10Numeric+
0.07Omission+
0.07Hallucination+
0.06Robustness
\end{aligned}
]

Use full floating-point precision internally and round only the displayed result.

### Avoid dataset-composition bias

Calculate each metric inside fixed strata first:

```text
clean_print
real_scan
mobile_photo
handwriting
numeric_heavy
mixed_script
historical
```

Then combine them using frozen stratum weights:

[
MetricScore=\sum_h w_h MetricScore_h
]

Otherwise adding many easy synthetic lines could move the leaderboard without any model improving.

---

# 6. Standard inference protocol

Every model adapter should save:

```json
{
  "model_id": "qwen3_vl_8b_instruct",
  "model_revision": "commit-or-api-version",
  "adapter_version": "1.2.0",
  "sample_id": "line_000001",
  "prompt": "Transcribe the image exactly...",
  "raw_prediction": "می‌روم ساعت ۱۲:۳۰",
  "runtime_ms": 842,
  "input_width": 1280,
  "input_height": 196,
  "decoding": {
    "temperature": 0,
    "do_sample": false,
    "max_new_tokens": 512
  },
  "status": "success"
}
```

Rules:

* Use the official image processor required by the model.
* Do not add external deskewing, denoising, OCR, or spell correction unless it is part of the declared pipeline.
* Use greedy or deterministic decoding where available.
* Give general VLMs a minimal frozen prompt requesting exact text only.
* Store raw output before normalization.
* Do not retry a low-quality answer.
* Retry only transport, timeout, or infrastructure failures.
* Record truncation separately from ordinary OCR errors.
* Use a sufficiently high output-token limit so the runner does not create artificial omissions.
* Freeze dependency versions, model revisions, prompts, preprocessing, and hardware settings.

METATR likewise applies identical normalization to references and predictions and freezes model settings except where output limits must accommodate long documents. ([arXiv][4])

For nondeterministic closed APIs, run the full set three times and publish mean, standard deviation, and worst run. A deterministic local checkpoint needs one official pass after the adapter has passed validation.

---

# 7. Statistical reporting

The leaderboard should show:

| Result               | Publish                             |
| -------------------- | ----------------------------------- |
| Composite score      | Point estimate and 95% CI           |
| Ten component scores | All visible                         |
| Per-domain results   | All visible                         |
| CER distribution     | Mean, median, p95                   |
| Failure rates        | Timeout, crash, truncation          |
| Efficiency           | Runtime, VRAM, API cost             |
| Versioning           | Model revision and adapter revision |

Use paired bootstrap resampling by `source_document_id`, not individual lines. This preserves the correlation between lines from the same page. At least 1,000 bootstrap replicates is consistent with implementations used in SacreBLEU significance testing and OCR robustness evaluation. ([GitHub][13])

Do not pretend tiny score differences are meaningful. When two models’ paired bootstrap difference interval includes zero, show them in the same statistical tier even if the displayed scores differ slightly.

---

# 8. Validation tests before benchmarking 82 models

Before running the full leaderboard, create a metric fixture containing manually verified cases such as:

```text
يادگیری  vs یادگیری
كتاب     vs کتاب
می روم   vs می‌روم
میروم    vs می‌روم
۱۲۳      vs 123
۱۲۳۴     vs ۱۲۳
۱۴۰۳/۰۲  vs ۱۴۰۳/۰۲/۰۱
سلام     vs سلام سلام سلام
""       vs متن ساختگی
```

For every fixture, assert:

* Strict edit counts
* Canonical edit counts
* Word tokens
* Space and ZWNJ labels
* Numeric span match
* Deletions
* Insertions
* Exact-line status
* Final weighted score

Also verify the identity case:

```text
prediction == reference
```

must produce exactly `100.0` for all applicable metrics.

The final repository should therefore have this structure:

```text
benchmark/
├── schemas/
│   ├── sample.schema.json
│   └── prediction.schema.json
├── normalization/
│   ├── canonical_v1.py
│   └── fixtures.json
├── metrics/
│   ├── edit_alignment.py
│   ├── cer_wer.py
│   ├── chrf.py
│   ├── spacing.py
│   ├── numeric.py
│   ├── failure_scores.py
│   └── robustness.py
├── corruptions/
├── adapters/
├── tests/
├── dataset/
└── leaderboard/
```

This produces a leaderboard that answers not merely **“which model has the lowest CER?”**, but also which model preserves Persian Unicode, handles نیم‌فاصله correctly, reads numbers exactly, avoids dropping content, resists repetition loops, and survives realistic degradation.

[1]: https://ocr-d.de/en/spec/ocrd_eval.html "Quality Assurance in OCR-D - OCR-D"
[2]: https://github.com/opendatalab/OmniDocBench?utm_source=chatgpt.com "opendatalab/OmniDocBench: [CVPR 2025] A ..."
[3]: https://pcdp.qut.ac.ir/article_722202.html "Empirical Evaluation of Well-known Farsi OCR Engines on the IDPL-PFOD Dataset"
[4]: https://arxiv.org/html/2605.26712v1 "METATR: A Multilingual, Evolving Benchmark for Automatic Text Recognition"
[5]: https://github.com/roshan-research/hazm "GitHub - roshan-research/hazm: Persian NLP Toolkit · GitHub"
[6]: https://github.com/jitsi/jiwer "GitHub - jitsi/jiwer: Evaluate your speech-to-text system with similarity measures such as word error rate (WER) · GitHub"
[7]: https://arxiv.org/abs/2010.00287 "[2010.00287] Joint Persian Word Segmentation Correction and Zero-Width Non-Joiner Recognition Using BERT"
[8]: https://github.com/haraai/ParsiNorm/blob/main/README.md "ParsiNorm/README.md at main · haraai/ParsiNorm · GitHub"
[9]: https://arxiv.org/html/2603.02803v1 "Structure-Aware Text Recognition for Ancient Greek Critical Editions"
[10]: https://github.com/Aditya-PS-05/devanagari-ocr-benchmark/blob/main/README.md "devanagari-ocr-benchmark/README.md at main · Aditya-PS-05/devanagari-ocr-benchmark · GitHub"
[11]: https://arxiv.org/html/2606.26041v1 "How Robust is OCR-Reasoning? Evaluating OCR-Reasoning Robustness of Vision-Language Models under Visual Perturbations"
[12]: https://github.com/pasterinjlu/OCR-Reasoning-Robust "GitHub - pasterinjlu/OCR-Reasoning-Robust · GitHub"
[13]: https://github.com/mjpost/sacrebleu?utm_source=chatgpt.com "mjpost/sacrebleu: Reference BLEU implementation that ..."
