## Recommendation

Do **not** rank all 82 systems with one undifferentiated metric. Your candidates include cropped-line recognizers, classical OCR engines, full-page parsers, general VLMs, commercial APIs, handwriting systems, and digit-only models. A single score would either punish specialized models for capabilities they were never designed to provide or reward VLMs for producing plausible-looking but inaccurate text.

Use three linked leaderboards:

1. **Persian Text Recognition** — common to nearly every model.
2. **Persian Document Parsing** — only full-page-capable systems.
3. **Specialized Persian OCR** — handwriting, scene text, and digits.

CER and WER remain the standard transcription metrics, but they cannot detect scrambled columns, broken tables, or semantically plausible hallucinations. Modern document benchmarks therefore combine text accuracy with reading order, layout, tables, faithfulness, and robustness. ([arXiv][1])

---

# 1. Persian Text Recognition leaderboard

This should be the **primary common leaderboard** for all models. Give every model the same cropped lines and single text blocks, so layout capability does not influence recognition quality.

## Suggested score

| Metric                                   |   Weight | Purpose                               |
| ---------------------------------------- | -------: | ------------------------------------- |
| Persian Canonical CER                    |  **30%** | Main Persian transcription accuracy   |
| Strict CER                               |  **10%** | Exact Unicode and typography fidelity |
| Canonical WER                            |  **10%** | Whole-word usability                  |
| chrF++                                   |  **10%** | Partial character and word similarity |
| Exact Line Accuracy                      |   **5%** | Perfect transcription frequency       |
| ZWNJ and Spacing F1                      |  **10%** | Persian word-boundary correctness     |
| Numeral and Critical-Field Accuracy      |  **10%** | Numbers, dates, prices, identifiers   |
| Omission/Hallucination/Repetition Safety |  **10%** | Generative-model reliability          |
| Robustness Retention                     |   **5%** | Resistance to degradation             |
| **Total**                                | **100%** |                                       |

## 1.1 Persian Canonical CER — 30%

Use character edit distance after a documented Persian normalization:

[
CER=\frac{S+D+I}{N}
]

where (S), (D), and (I) are substitutions, deletions, and insertions, and (N) is the number of reference characters.

CER should be the largest component because Persian word segmentation is affected by ordinary spaces and ZWNJ, making WER unusually sensitive to orthographic conventions. Character-level metrics also behave more consistently across different writing systems than word-only metrics. ([arXiv][1])

Report both:

* **Micro CER:** sum all edits and divide by all reference characters.
* **Macro CER:** calculate CER per sample and average samples equally.
* **Median CER:** resistant to a few catastrophic VLM outputs.
* **95% paired-bootstrap confidence interval:** resample the same lines or pages for every model.

Length-weighted and per-instance reporting reveal different behavior, and paired bootstrap intervals prevent tiny, statistically meaningless differences from deciding the ranking. ([arXiv][2])

Convert CER to a leaderboard-friendly score using:

[
CanonicalCERScore=100\times\max(0,1-CER)
]

Do not call this simply “accuracy” without defining it.

## 1.2 Strict CER — 10%

Strict CER should apply only Unicode NFC normalization and newline standardization. It should preserve:

* Persian versus Arabic `ک/ك`
* Persian versus Arabic `ی/ي/ى`
* ZWNJ
* Persian, Arabic-Indic, and Latin digits
* Punctuation
* Diacritics
* Line breaks where relevant

Persian documents frequently contain visually similar but computationally distinct Arabic and Persian code points. Persian also uses ZWNJ, bidirectional control behavior, a distinct digit set, and specific punctuation conventions. 

Strict CER answers: **“Did the model reproduce the encoded text faithfully?”**

Canonical CER answers: **“Did the model recover the intended Persian content?”**

You need both.

## 1.3 Canonical WER — 10%

Compute WER after Persian canonical normalization and tokenization:

[
WER=\frac{S_w+D_w+I_w}{N_w}
]

WER is valuable because one wrong character can make an entire word unusable for search, indexing, language modeling, and RAG. However, it must not be the primary metric because ZWNJ and inconsistent Persian spacing can turn a small boundary error into multiple word errors. ([arXiv][3])

Report:

* WER with ZWNJ treated as an internal word boundary marker.
* WER with ZWNJ collapsed to an ordinary space.
* Word exact-match rate.

The first evaluates correct Persian typography; the second measures recoverable linguistic content.

## 1.4 chrF++ — 10%

chrF++ compares character n-grams together with word n-grams. It gives partial credit when a word is nearly correct without being as forgiving as embedding-based semantic metrics. Character n-gram evaluation has shown useful behavior for morphologically rich languages and is increasingly used in low-resource script stress tests. ([ACL Anthology][4])

Use standard chrF++ settings and publish them unchanged:

* Character n-grams: 1–6
* Word n-grams: 0–2
* (\beta=2), emphasizing recall

BLEU may be included for comparison with PsOCR, SARD, and OmniDocBench, but I would **not** give it leaderboard weight. Exact transcription is not machine translation, and BLEU can obscure important character-level mistakes. PsOCR itself reports a large gap between character accuracy, word accuracy, and BLEU, showing why they should not be treated as interchangeable. ([arXiv][3])

## 1.5 Exact Line Accuracy — 5%

Calculate:

[
ELA=\frac{\text{perfectly matched lines}}{\text{all lines}}
]

Publish both:

* `exact_line_accuracy_strict`
* `exact_line_accuracy_canonical`

IDPL-PFOD2 reports sequence-level accuracy alongside normalized edit distance; its results illustrate how a model can have very high edit similarity while substantially fewer samples are completely correct. 

This metric matters because 98% character similarity can still mean almost every line needs manual editing.

---

# 2. Persian-specific orthographic metrics

## 2.1 ZWNJ and spacing F1 — 10%

Treat every potential boundary between adjacent characters as one of:

* no boundary
* ZWNJ
* ordinary space
* newline

Then report macro precision, recall, and F1 for each class.

At minimum publish:

* `zwnj_precision`
* `zwnj_recall`
* `zwnj_f1`
* `space_f1`
* `boundary_macro_f1`

ZWNJ is not decorative metadata. It is an essential part of Persian orthography, and Persian word segmentation errors commonly involve missing spaces, extra spaces, and ZWNJ confusion. 

Do not hide ZWNJ errors by normalizing them away before all evaluation. Normalize them only for the lenient content score.

## 2.2 Persian character-confusion report

Publish a confusion matrix and dedicated rates for:

* `ی ↔ ي ↔ ى`
* `ک ↔ ك`
* `ه ↔ ۀ ↔ ة`
* `ا ↔ آ`
* Hamza forms
* `ب/پ/ت/ث`
* `ج/چ/ح/خ`
* `د/ذ`
* `ر/ز/ژ`
* `س/ش`
* `ص/ض`
* `ط/ظ`
* `ع/غ`
* `ف/ق`
* `ک/گ`
* Persian versus Arabic punctuation
* Persian versus Arabic-Indic digits

Persian OCR is particularly affected by cursive joining, visually similar letters, dots, and code-point variants. Persian OCR datasets and Arabic-script OCR research repeatedly identify these as core recognition difficulties. 

This should be diagnostic rather than directly folded into the score; otherwise, you risk double-counting CER errors.

## 2.3 Numeral and critical-field accuracy — 10%

Extract and evaluate these spans independently:

* Integers and decimals
* Persian, Latin, and Arabic-Indic digits
* Percentages
* Prices and currencies
* Jalali and Gregorian dates
* Times
* Telephone numbers
* Postal codes
* National identifiers
* Mathematical operators
* Mixed RTL/LTR strings

Report:

* Numeric character accuracy
* Exact numeric span accuracy
* Numeric insertion/deletion/substitution rate
* Date exact-match rate
* Currency-value exact-match rate
* Mixed-direction span accuracy

Persian has a distinct Unicode digit set and frequently mixes RTL text with Latin strings and numbers. Mapping every numeral to a single form in the main metric would conceal whether a system actually reproduced the document correctly. 

For numeric values, add a **semantic numeric score** where `۱٬۲۵۰`, `١٢٥٠`, and `1250` can map to the same numeric value. Keep this separate from exact transcription.

---

# 3. Faithfulness metrics for VLMs

Generative OCR models can produce fluent but visually unsupported text, repeat sections, or silently omit blocks. Aggregate CER alone can hide this behavior, especially when only a small percentage of pages fail catastrophically. ([arXiv][5])

## Required metrics

### Character-level decomposition

Report separately:

[
SubstitutionRate=S/N
]

[
DeletionRate=D/N
]

[
InsertionRate=I/N
]

Interpretation:

* High deletion rate: omitted content.
* High insertion rate: hallucinated or repeated content.
* High substitution rate: recognition confusion.

### Word-level faithfulness

* **Missing-word rate:** unmatched reference words ÷ reference words.
* **Extra-word rate:** unmatched predicted words ÷ predicted words.
* **Content recall:** matched reference words ÷ reference words.
* **Content precision:** matched predicted words ÷ predicted words.

### Catastrophic failure rate

Use:

[
CatastrophicRate=\frac{#{samples:CER>0.5}}{#samples}
]

The recent Devanagari OCR-VLM stress test explicitly recommends median CER and the percentage of samples above 50% CER because rare repetition loops badly distort means. ([arXiv][6])

Also report:

* `output_reference_length_ratio`
* `p95_length_ratio`
* percentage with ratio above 2
* repeated 4-gram ratio
* percentage ending because of token limit
* empty-output rate
* invalid-format rate
* timeout/error rate

A model with 1% catastrophic pages should not beat a stable model merely because its remaining pages are marginally cleaner.

---

# 4. Persian Document Parsing leaderboard

Only page-capable models should enter this track. Line recognizers such as CRNN and PARSeq should be marked `not_applicable`, not assigned zero.

## Suggested document score

| Component                         |   Weight |
| --------------------------------- | -------: |
| Persian text transcription        |  **35%** |
| Reading order                     |  **15%** |
| Layout and element detection      |  **10%** |
| Table extraction                  |  **10%** |
| Formula and code extraction       |   **5%** |
| Hallucination and omission safety |  **10%** |
| Degradation robustness            |  **10%** |
| Efficiency and reliability        |   **5%** |
| **Total**                         | **100%** |

## 4.1 Reading-order score — 15%

Use two metrics:

1. **Reading-order normalized edit similarity**, compatible with OmniDocBench.
2. **Successor-edge F1**, comparing whether each text block or line is followed by the correct next block.

OmniDocBench evaluates reading order using normalized edit distance, while newer work treats order as explicit relations between elements and reports edge-level accuracy. The edge metric is easier to interpret and less entangled with recognition errors. ([arXiv][7])

Persian test pages must include:

* RTL single-column pages
* RTL two- and three-column pages
* Persian body text with LTR formulas
* Persian text containing English paragraphs
* sidebars and pull quotes
* footnotes
* newspapers
* captions
* forms
* tables embedded between paragraphs

Score text recognition and reading order independently. A model should not receive a bad reading-order score merely because it misrecognized a few letters.

## 4.2 Layout F1 or mAP — 10%

For systems returning bounding boxes and confidence scores, report:

* mAP@[.50:.95]
* AP50
* macro class F1
* mean IoU
* detection recall

Recommended Persian classes:

* title
* paragraph
* list
* table
* image
* caption
* footnote
* header
* footer
* page number
* equation
* code
* form field
* handwriting
* stamp/seal

For autoregressive VLMs that do not return confidence values, use class-aware box matching and macro F1 at IoU 0.5 rather than inventing confidence scores. dots.ocr similarly argues that conventional mAP is awkward for autoregressive parsers that do not natively emit confidence values. ([arXiv][7])

## 4.3 Table extraction — 10%

Use:

* **GriTS-Topology:** rows, columns, spans and cell relationships.
* **GriTS-Content:** recognized cell contents.
* **GriTS-Location:** spatial cell alignment when boxes exist.
* **TEDS-Structure:** compatibility with existing document benchmarks.
* **Cell exact-match rate:** especially for numbers.
* **Row/column count accuracy.**

GriTS compares tables as two-dimensional matrices and separates topology, content, and location, avoiding cases where one mistaken cell offset disproportionately destroys the whole score. TEDS remains useful for comparison with PubTabNet and OmniDocBench. 

For Persian tables, explicitly test:

* RTL column order
* Persian and Latin digits in the same table
* merged cells
* multi-line headers
* borderless tables
* financial reports
* timetables
* forms
* tables containing Persian abbreviations
* tables containing formulas

An LLM judge can be added for semantic table quality, but it should be secondary. A recent study reported strong correlation with human table judgments, yet broader judge evaluations show instability across scenarios. ([arXiv][8])

## 4.4 Formula and code extraction — 5%

For pages containing formulas or code, report:

* Exact LaTeX match after safe canonicalization
* Rendered formula similarity or CDM
* Formula detection recall
* Code exact-match rate
* Indentation accuracy
* Persian-text/formula boundary accuracy

OmniDocBench evaluates formulas using CDM, normalized edit distance, and BLEU because character-level string equality can penalize visually equivalent LaTeX forms. ([arXiv][7])

---

# 5. Robustness metrics

A clean synthetic set is useful for debugging but should not dominate the ranking. Recent low-resource-script studies found that systems clustered closely on clean rendered text but separated dramatically on real scans and degraded inputs. Real-world physical reconstruction benchmarks likewise show a substantial gap between digital PDFs and captured documents. ([arXiv][9])

Create matched clean/degraded pairs and report:

## Relative robustness retention

[
RCR=100\times\frac{Score_{degraded}}{Score_{clean}}
]

## Absolute degradation

[
Drop=Score_{clean}-Score_{degraded}
]

## Worst-case retention

Take the minimum retention across degradation families.

Suggested degradation families:

* Gaussian and motion blur
* JPEG compression
* downscaling
* scanner noise
* bleed-through
* faded ink
* uneven illumination
* shadows
* perspective distortion
* page curvature
* rotation and skew
* watermark
* colored or patterned background
* partial cropping
* stains and folds
* screen photography
* photocopy generations

Publish per-severity curves rather than one average. OCR-Robust similarly evaluates clean performance, relative retention, worst-case retention, and a composite robustness index across multiple perturbation levels. ([arXiv][10])

---

# 6. Optional downstream usefulness score

For a real document-intelligence leaderboard, add a **separate**, non-core downstream track:

* Retrieval Recall@5
* nDCG@10
* evidence-span recall
* question-answer exact match
* ANLS for short extracted answers
* key-value extraction F1
* named-entity recall
* numeric answer accuracy

Character accuracy does not always predict RAG quality because structural errors, lost table relationships, and scrambled reading order can break retrieval despite low CER. OHRBench and InduOCRBench therefore evaluate retrieval and generation effects in addition to text edit distance. ([arXiv][11])

ANLS or ANLS* is appropriate for short information-extraction answers with minor OCR differences. It is **not** a replacement for CER on full transcription. ([arXiv][12])

---

# 7. LLM-as-judge: keep it small

I would **not use 70% LLMJudgeScore** for this leaderboard.

For exact OCR, the ground truth already exists. A judge can forgive a wrong digit, silently overlook missing lines, prefer fluent Persian over faithful Persian, or vary when evaluating long documents. Current work finds useful human correlation for specialized table and formula judging, but broader long-form judge studies still report instability, and multi-model agreement has outperformed a single VLM judge for OCR quality verification. ([arXiv][8])

Recommended judge use:

* **0%** for line-level OCR.
* **Maximum 5%** for full document parsing.
* Use it only for:

  * semantic table equivalence
  * visually equivalent formulas
  * formatting usability
  * obvious structural corruption
  * qualitative error taxonomy
* Never let it override:

  * wrong numbers
  * missing text
  * hallucinated text
  * exact character errors
  * ZWNJ and Unicode errors

Calibrate the judge against at least 300 manually scored Persian outputs, publish its agreement with humans, freeze the prompt/model/version, use deterministic inference, and rerun the calibration whenever the judge model changes.

---

# 8. Three normalization tracks

Every prediction should be evaluated through these three pipelines.

## Track A — Strict/diplomatic

Normalize only:

* Unicode NFC
* CRLF to LF
* model wrapper tokens and accidental Markdown fences

Preserve everything else.

## Track B — Persian canonical

Normalize:

* Arabic Kaf `ك` → Persian Kaf `ک`
* Arabic Yeh and Alef Maksura `ي/ى` → Persian Yeh `ی`
* Arabic presentation forms → base Unicode characters
* deprecated Heh/Hamza representations → canonical sequence
* bidi control characters that have no textual content
* repeated ordinary spaces
* line-ending conventions

Preserve:

* ZWNJ
* digit script
* punctuation
* text content

## Track C — Content-lenient

Additionally:

* remove Tatweel
* remove optional diacritics
* collapse ZWNJ and spaces to a common boundary
* map all numeral scripts to numeric values
* normalize punctuation variants

Track C must never replace strict scoring; it only shows how much of the content is recoverable after normalization. Persian computing research documents why Yeh/Kaf variants, ZWNJ, Tatweel, digits, Hamza sequences, and bidi behavior require explicit handling. 

---

# 9. Reporting format

For every model, publish at least:

```yaml
text:
  cer_strict_micro:
  cer_strict_macro:
  cer_canonical_micro:
  cer_canonical_macro:
  cer_median:
  cer_ci95:
  wer_canonical:
  chrf_pp:
  exact_line_strict:
  exact_line_canonical:

persian:
  zwnj_precision:
  zwnj_recall:
  zwnj_f1:
  boundary_macro_f1:
  persian_kaf_yeh_accuracy:
  digit_exact_accuracy:
  numeric_span_exact_accuracy:
  punctuation_accuracy:

faithfulness:
  substitution_rate:
  deletion_rate:
  insertion_rate:
  missing_word_rate:
  extra_word_rate:
  catastrophic_rate:
  repetition_failure_rate:
  empty_output_rate:

document:
  reading_order_ned:
  reading_order_edge_f1:
  layout_macro_f1:
  map_50_95:
  grits_topology:
  grits_content:
  grits_location:
  teds_structure:
  formula_cdm:

robustness:
  clean_score:
  degraded_score:
  relative_retention:
  worst_case_retention:

efficiency:
  median_seconds_per_page:
  p95_seconds_per_page:
  peak_vram_gb:
  peak_ram_gb:
  cost_per_1000_pages:
  timeout_rate:
```

Always publish both **mean and variance/distributional information**, per-domain results, and confidence intervals. Attribute-level evaluation is more informative than one global score, and recent benchmark audits show that annotation errors and score saturation can otherwise produce unreliable rankings. ([arXiv][7])

## Final metric priority

The highest-value metrics for your Persian benchmark are:

1. **Persian Canonical CER**
2. **Strict CER**
3. **Canonical WER**
4. **ZWNJ and spacing F1**
5. **Numeric-span exact accuracy**
6. **chrF++**
7. **Omission, insertion and catastrophic-failure rates**
8. **Reading-order edge F1**
9. **GriTS for tables**
10. **Robustness retention**
11. **Exact-line accuracy**
12. **Latency, memory, cost and runtime-failure rate**

That stack will tell you not merely which model produces the most plausible Persian, but which one most faithfully recovers the document.

[1]: https://arxiv.org/html/2603.25761v1 "A Survey of OCR Evaluation Methods and Metrics and the Invisibility of Historical Documents"
[2]: https://arxiv.org/html/2602.14524v1?utm_source=chatgpt.com "Error Patterns in Historical OCR: A Comparative Analysis of ..."
[3]: https://arxiv.org/html/2505.10055v1 "PsOCR: Benchmarking Large Multimodal Models for Optical Character Recognition in Low-resource Pashto Language"
[4]: https://aclanthology.org/anthology-files/pdf/W/W15/W15-3049.pdf?utm_source=chatgpt.com "chrF: character n-gram F-score for automatic MT evaluation"
[5]: https://arxiv.org/html/2603.02803v1?utm_source=chatgpt.com "Structure-Aware Text Recognition for Ancient Greek Critical ..."
[6]: https://arxiv.org/html/2606.29213v1?utm_source=chatgpt.com "Can OCR-VLMs Read Devanagari? A Stress-Test ..."
[7]: https://arxiv.org/html/2412.07626v2 "OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations"
[8]: https://arxiv.org/html/2603.18652v1?utm_source=chatgpt.com "Benchmarking PDF Parsers on Table Extraction with LLM ..."
[9]: https://arxiv.org/abs/2606.29213?utm_source=chatgpt.com "Can OCR-VLMs Read Devanagari? A Stress-Test Benchmark and Post-Correction Study"
[10]: https://arxiv.org/abs/2606.26041?utm_source=chatgpt.com "How Robust is OCR-Reasoning? Evaluating OCR-Reasoning Robustness of Vision-Language Models under Visual Perturbations"
[11]: https://arxiv.org/html/2412.02592v2?utm_source=chatgpt.com "Evaluating the Cascading Impact of OCR on Retrieval ..."
[12]: https://arxiv.org/html/2402.03848v9 "ANLS* - A Universal Document Processing Metric for Generative Large Language Models"
