# Run Persian OCR Benchmarks on Google Colab

This is the complete workflow for comparing OCR engines and vision-language
models on the included 20-image Persian benchmark.

## 1. Create the notebook

Open [Google Colab](https://colab.research.google.com/), create a notebook,
and select **Runtime → Change runtime type → T4 GPU** when the model supports
CUDA. CPU is enough for lightweight engines such as Tesseract and EasyOCR.

## 2. Download the benchmark

Run this cell, replacing the URL with the GitHub repository URL:

```python
!git clone https://github.com/OWNER/REPOSITORY.git
%cd REPOSITORY
```

The input layout is:

```text
small_bench/
  typed/1.jpg ... 10.jpg
  typed/1.md  ... 10.md
  hand-written/1.jpg ... 10.jpg
  hand-written/1.md  ... 10.md
```

## 3. Install one model at a time

Keep each experiment isolated. Install only the dependencies for the model
being tested, then restart the runtime if a package asks for it.

Examples:

```python
# Tesseract
!apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-fas
!pip -q install pytesseract pillow
```

```python
# EasyOCR
!pip -q install easyocr pillow
```

For Hugging Face OCR/VLM models, install the model's current requirements in
the model card and load it with the documented Transformers example. Record
the exact model ID, revision, GPU type, and install commands in the notebook.
Do not assume that two models use the same processor or prompt format.

## 4. Use one fixed extraction prompt

For VLMs, use this prompt unless the model card requires a different format:

```text
Extract all visible text from this image.
Return only the transcription.
Preserve Persian text, numbers, punctuation, line breaks, and Latin text.
Do not translate, summarize, explain, or add missing text.
```

Use deterministic decoding where available (`temperature=0`). Keep image
resolution and preprocessing consistent across models. Save the raw output;
do not silently correct it before scoring.

## 5. Save predictions

Create one folder per model and one Markdown file per image:

```text
bench_runs/
  MODEL_NAME/
    typed/1.md ... 10.md
    hand-written/1.md ... 10.md
```

The prediction file must contain only the extracted text. Keep a short record
of model ID, revision, prompt, preprocessing, decoding settings, runtime, and
date in the notebook or a `notes.md` file.

## 6. Score the results

Run this minimal evaluation cell after installing `rapidfuzz`:

```python
!pip -q install rapidfuzz pandas

from pathlib import Path
import pandas as pd
from rapidfuzz.distance import Levenshtein

def norm(s):
    return " ".join(s.replace("ي", "ی").replace("ك", "ک").split())

rows = []
for split in ("typed", "hand-written"):
    for ref_path in sorted(Path("small_bench", split).glob("*.md")):
        pred_path = Path("bench_runs", "MODEL_NAME", split, ref_path.name)
        ref = ref_path.read_text(encoding="utf-8")
        pred = pred_path.read_text(encoding="utf-8") if pred_path.exists() else ""
        ref_n, pred_n = norm(ref), norm(pred)
        rows.append({
            "split": split,
            "item": ref_path.stem,
            "cer": Levenshtein.distance(ref_n, pred_n) / max(len(ref_n), 1),
            "wer": Levenshtein.distance(ref_n.split(), pred_n.split()) / max(len(ref_n.split()), 1),
            "exact": ref_n == pred_n,
        })

scores = pd.DataFrame(rows)
display(scores.groupby("split")[['cer', 'wer', 'exact']].mean())
display(scores[['cer', 'wer', 'exact']].mean())
scores.to_csv("bench_runs/MODEL_NAME/scores.csv", index=False)
```

Replace `MODEL_NAME` in the cell with the folder you created. For a serious
comparison, also calculate strict scores that preserve whitespace and
zero-width non-joiners; normalized scores are useful for ranking but can hide
Persian spacing errors.

## 7. Report and inspect

Report typed, handwritten, and combined results separately. Use:

- CER: primary character-level accuracy metric.
- WER: word-level readability signal; Persian spacing can make it harsh.
- Exact match: strict whole-transcription agreement.

Inspect failed pages manually for hallucinated text, translation, summaries,
reordered lines, dropped half-spaces, and missing mixed Persian/Latin text.
Twenty images are a smoke benchmark, not a final leaderboard. Add receipts,
tables, low-quality scans, more writers, and mixed-language pages before making
a production decision.

## Reproducibility checklist

- Same images and prompt for every model.
- Exact model ID and revision recorded.
- Same preprocessing and output format.
- Raw predictions retained.
- Typed and handwritten scores reported separately.
- GPU type, runtime date, and dependency versions recorded.
