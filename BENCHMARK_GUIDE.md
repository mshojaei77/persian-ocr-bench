# Persian OCR/VLM Benchmark Guide

This repo already has a small benchmark set in `small_bench/`:

```text
small_bench/
  typed/
    1.jpg ... 10.jpg
    1.md  ... 10.md
  hand-written/
    1.jpg ... 10.jpg
    1.md  ... 10.md
```

Each `.jpg` is the input image. The `.md` file with the same stem is the
reference transcription. That is enough to benchmark OCR engines and VLMs
without adding a manifest yet.

## What This Benchmark Is Good For

Use it as a smoke benchmark for Persian text extraction:

- `typed/`: printed Persian pages, captions, mixed layout, and some Latin text.
- `hand-written/`: handwritten or informal Persian samples, shorter and harder.
- Paired `.md` references: score raw text, Markdown-preserving output, or both.

Do not treat this as a final leaderboard. It has only 20 images, so it is best
for quick model selection, regression checks, prompt comparison, preprocessing
checks, and finding obvious Persian OCR failures.

## Benchmark Targets

Run every model on the exact same images and save one prediction file per input:

```text
bench_runs/
  MODEL_NAME/
    typed/
      1.md
      ...
    hand-written/
      1.md
      ...
```

For OCR engines, the prediction should be the extracted text.

For VLMs, use a fixed prompt:

```text
Extract all visible text from this image.
Return only the transcription.
Preserve Persian text, numbers, punctuation, line breaks, and any Latin text.
Do not translate, summarize, explain, or add missing text.
```

Keep temperature at `0` when the API supports it. Record model name, version,
date, prompt, image preprocessing, and decoding settings beside the outputs.

## Minimum Metrics

Use three scores:

- `CER`: character error rate. This is the main Persian OCR metric because
  Persian spacing, joins, diacritics, and similar letters can create small but
  important character-level errors.
- `WER`: word error rate. Useful for readability, but it can be harsh when
  Persian half-spaces or punctuation differ.
- `Exact line match`: percentage of reference lines that appear exactly after
  normalization. This catches layout and line-break damage that CER/WER hide.

Score separately for:

- `typed`
- `hand-written`
- `all`

Report median and mean. Median matters because a single bad page can dominate a
20-sample benchmark.

## Persian Normalization

Score two versions:

1. `strict`: only normalize newlines and Unicode form.
2. `normalized`: normalize common Persian/Arabic variants.

Suggested normalization before `normalized` scoring:

- Unicode normalize with `NFC`.
- Convert Arabic Yeh/Kaf to Persian Yeh/Kaf: `ي -> ی`, `ك -> ک`.
- Normalize Arabic/Persian digits only if digit identity is not being tested.
- Normalize zero-width non-joiner variants to `\u200c`.
- Collapse repeated spaces and trim line ends.
- Keep line breaks for line-level scoring.

Do not delete punctuation by default. Persian punctuation errors matter in real
documents.

## Evaluation Script Shape

The simplest script should:

1. Discover all `small_bench/**/*.jpg`.
2. Read the reference from the sibling `.md`.
3. Read the prediction from `bench_runs/MODEL_NAME/.../*.md`.
4. Compute CER and WER with Levenshtein distance.
5. Write `bench_runs/MODEL_NAME/scores.csv`.

Recommended CSV columns:

```csv
model,split,item,cer_strict,wer_strict,cer_norm,wer_norm,line_exact_norm,ref_chars,pred_chars
```

That is enough for repeatable comparisons. Add a richer manifest only when the
benchmark grows beyond simple paired files.

## OCR Model Workflow

For traditional OCR models:

1. Run the model once with no preprocessing.
2. Run a second pass with the smallest useful preprocessing only: grayscale,
   deskew, denoise, or contrast.
3. Save both as separate model names, for example `tesseract_raw` and
   `tesseract_deskew`.
4. Compare typed and handwritten splits separately.

This matters because OCR engines often improve more from preprocessing than
from model changes, especially on scans and low-contrast images.

## VLM Workflow

For VLMs:

1. Use one fixed extraction prompt.
2. Disable explanations and Markdown fences.
3. Keep image resolution consistent across models.
4. Save raw model output before cleanup.
5. Score both raw and cleaned output if cleanup is part of the product.

VLMs should also be checked manually for hallucination:

- Added text not visible in the image.
- Translation instead of transcription.
- Summary instead of full extraction.
- Reordered lines or merged columns.
- Dropped Persian half-spaces or mixed Persian/Latin sections.

## Recommended Leaderboard Format

```markdown
| model | split | CER norm | WER norm | line exact | notes |
| --- | --- | ---: | ---: | ---: | --- |
| model-a | typed | 0.00 | 0.00 | 100% | raw |
| model-a | hand-written | 0.00 | 0.00 | 100% | raw |
```

Keep notes short: `raw`, `deskew`, `prompt-v1`, `api-2026-07-10`, etc.

## When To Expand The Benchmark

Add samples only when a model decision depends on them. Useful next additions:

- Low-quality scans.
- Receipts, forms, IDs, book pages, and tables.
- Mixed Persian/English pages.
- Cropped lines and full-page layouts.
- More handwritten writers.
- Images with known hard Persian characters and half-spaces.

Once the set grows, add a `small_bench/manifest.csv`:

```csv
split,item,image,reference,source,layout,quality,notes
typed,1,typed/1.jpg,typed/1.md,book,page,clean,
hand-written,1,hand-written/1.jpg,hand-written/1.md,note,paragraph,medium,
```

## Current Practice Notes

Recent OCR/VLM benchmark discussions still use CER and WER as core metrics, but
document extraction quality also depends on layout, tables, line order, and
hallucination checks. For Persian specifically, public research keeps pointing
out that right-to-left scripts need language-specific evaluation data rather
than assuming Latin or generic multilingual benchmarks transfer cleanly.

Useful references:

- https://arxiv.org/html/2502.06445v1
- https://arxiv.org/html/2603.25761v1
- https://arxiv.org/pdf/2312.01177
- https://aclanthology.org/2025.evalmg-1.5/
- https://github.com/video-db/ocr-benchmark

