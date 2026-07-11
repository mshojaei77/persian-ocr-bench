# Benchmark the Persian OCR Catalog on Google Colab

`small_bench` contains 20 image/ground-truth pairs: 10 typed and 10
handwritten.  Each `*.jpg` has a matching `*.md` reference transcription.
This is a smoke benchmark: use it to choose candidates, not to publish a final
leaderboard.

## 1. Start a Colab runtime

Create a notebook at [Google Colab](https://colab.research.google.com/) and
select **Runtime -> Change runtime type -> T4 GPU** for neural models.  Use a
CPU runtime for Tesseract.  Colab GPU type and session limits vary, so record
the actual GPU shown by `nvidia-smi` in every run.

Put this repository (or just `small_bench.rar`) in Google Drive.  This keeps
inputs and results after Colab resets.

```python
from google.colab import drive
drive.mount("/content/drive")

!unrar x -o+ /content/drive/MyDrive/persian-ocr/small_bench.rar /content/
!find /content/small_bench -type f | head
!nvidia-smi
```

If the repository is public, cloning it is equivalent:

```python
!git clone https://github.com/mshojaei77/persian-ocr-bench.git /content/persian-ocr
%cd /content/persian-ocr
```

Copy the images to local runtime storage before inference; mounted Drive is
best for persistence, not repeated small-file reads.

```python
from pathlib import Path
from shutil import copytree

DRIVE_ROOT = Path("/content/drive/MyDrive/persian-ocr")
DATA_ROOT = Path("/content/small_bench")
RUN_ROOT = DRIVE_ROOT / "bench_runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
assert sum(1 for _ in DATA_ROOT.rglob("*.jpg")) == 20
assert sum(1 for _ in DATA_ROOT.rglob("*.md")) == 20
```

## 2. Keep every model on the same contract

For each model, create:

```text
bench_runs/<model_id>/
  typed/1.md ... 10.md
  hand-written/1.md ... 10.md
  run.json
  scores.csv
```

Prediction files contain only raw extracted text.  Do not correct spelling,
half-spaces, or Arabic/Persian characters before saving.  `run.json` records
the model ID, revision/commit, install command, GPU, package versions, prompt,
image preprocessing, decoding settings, and date.

For generic VLMs, use deterministic decoding and this fixed prompt unless the
official model card requires a task token or its own prompt format:

```text
Extract all visible text from this image.
Return only the transcription.
Preserve Persian text, numbers, punctuation, line breaks, and Latin text.
Do not translate, summarize, explain, or add missing text.
```

Use the model-card prompt verbatim when required, but record the deviation in
`run.json`.  Do not compare line recognizers with page parsers on layout scores.

## 3. Benchmark in waves

Do not install all 26 models in one Colab environment.  Run one model per
runtime (or restart after its official installation) because the catalog mixes
Paddle, PyTorch, custom `trust_remote_code`, vLLM, and model-specific stacks.

| Wave | Catalog IDs | Purpose |
| --- | --- | --- |
| 1: cheap baselines | `tesseract_fas`, `ppocrv5_arabic_mobile_rec`, `easyocr_fa`, `hezar_crnn_base_fa_v2` | Establish OCR quality, speed, and Persian-script failure cases. |
| 2: primary page models | `falcon_ocr`, `surya_ocr_2`, `paddleocr_vl_1_6`, `hunyuanocr_1_5`, `mineru_2_5_pro_2605`, `nanonets_ocr2_1_5b_exp` | Main document-OCR bake-off. |
| 3: Persian/control models | `qwen3_vl_2b_persian_arabic_ocr`, `qwen3_vl_2b_instruct`, `dots_ocr`, `dots_mocr`, `deepseek_ocr`, `deepseek_ocr_2`, `chandra_ocr_2`, `qianfan_ocr` | Test Persian adaptation and stronger parsers. |
| 4: optional / costly | `weightedai_persian_ocr`, `got_ocr2`, `glm_ocr`, `lightonocr_2_1b`, `khanandeh_0_1_persian_ocr_2b`, `infinity_parser2_flash`, `firered_ocr`, `qwen3_vl_8b_instruct` | Only run if an earlier wave leaves a real gap. |

The line recognizers (`ppocrv5_arabic_mobile_rec`, `hezar_crnn_base_fa_v2`,
and `weightedai_persian_ocr`) need text-line crops or a detector.  Score their
recognition output separately from full-page VLMs; using their output as a
single page transcription without a shared detector makes reading-order scores
meaningless.

For each VLM/pipeline, open its `links.huggingface` or `links.github` entry in
`models.yaml`, install the exact official requirements, and adapt only the
inference cell below.  Pin a model revision; do not use a floating `main`.

```python
MODEL_ID = "falcon_ocr"  # must match an id in models.yaml
RUN_DIR = RUN_ROOT / MODEL_ID

from pathlib import Path
import json, platform, subprocess, sys

for split in ("typed", "hand-written"):
    (RUN_DIR / split).mkdir(parents=True, exist_ok=True)

metadata = {
    "model_id": MODEL_ID,
    "model_revision": "REPLACE_WITH_PINNED_REVISION",
    "prompt": "REPLACE_WITH_OFFICIAL_PROMPT_OR_FIXED_PROMPT",
    "preprocessing": "original JPEG; no rotation/cropping unless recorded",
    "decoding": "deterministic",
    "python": sys.version,
    "platform": platform.platform(),
}
(RUN_DIR / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
```

Implement the smallest adapter for the model's official API.  It must return
only a string; the loop and scorer stay unchanged for every model.

```python
from pathlib import Path

def extract_text(image_path: Path) -> str:
    # Replace this body with the official model-card inference call.
    # Return raw text only; do not normalize or post-edit it here.
    raise NotImplementedError

for split in ("typed", "hand-written"):
    for image_path in sorted(DATA_ROOT.joinpath(split).glob("*.jpg"), key=lambda p: int(p.stem)):
        prediction = extract_text(image_path)
        (RUN_DIR / split / f"{image_path.stem}.md").write_text(prediction, encoding="utf-8")
```

### Lightweight baseline cells

Tesseract is a useful CPU baseline.  Run both `fas` and `fas+eng` as separate
model IDs if mixed Persian/English pages matter, and record `--oem` and `--psm`.

```python
!apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-fas
!pip -q install pytesseract pillow

import pytesseract
from PIL import Image

def extract_text(image_path):
    return pytesseract.image_to_string(Image.open(image_path), lang="fas+eng", config="--oem 1 --psm 6")
```

EasyOCR is the quickest GPU pipeline baseline.  Keep `paragraph=False` to
avoid concealing its own reading-order decisions.

```python
!pip -q install easyocr

import easyocr
reader = easyocr.Reader(["fa", "en"], gpu=True)

def extract_text(image_path):
    return "\n".join(reader.readtext(str(image_path), detail=0, paragraph=False))
```

For PaddleOCR, follow the version-specific command in its official docs and
record the detector and Arabic-script recognition checkpoint separately.  For
all other models, use their repository/model-card example rather than copying
an adapter from a different architecture.

## 4. Score raw and normalized recognition

This cell has no extra dependency.  It writes per-image scores and prints
typed, handwritten, and overall averages.  The raw columns preserve every
character; normalized columns apply the catalog's Persian/Arabic and whitespace
normalization.  Read normalized scores as ranking aids, not proof that ZWNJ or
digit errors do not matter.

```python
from pathlib import Path
from difflib import SequenceMatcher
import csv
import re
import unicodedata

def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", text).strip()

def distance(a, b):
    # Levenshtein distance with O(min(len(a), len(b))) memory.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]

def cer(reference, prediction):
    return distance(reference, prediction) / max(len(reference), 1)

def wer(reference, prediction):
    return distance(reference.split(), prediction.split()) / max(len(reference.split()), 1)

rows = []
for split in ("typed", "hand-written"):
    for reference_path in sorted(DATA_ROOT.joinpath(split).glob("*.md"), key=lambda p: int(p.stem)):
        prediction_path = RUN_DIR / split / reference_path.name
        reference = reference_path.read_text(encoding="utf-8")
        prediction = prediction_path.read_text(encoding="utf-8") if prediction_path.exists() else ""
        ref_normalized, pred_normalized = normalize(reference), normalize(prediction)
        rows.append({
            "split": split, "item": reference_path.stem,
            "CER_raw": cer(reference, prediction), "WER_raw": wer(reference, prediction),
            "exact_line_accuracy": reference == prediction,
            "CER_normalized": cer(ref_normalized, pred_normalized),
            "WER_normalized": wer(ref_normalized, pred_normalized),
            "exact_normalized": ref_normalized == pred_normalized,
        })

with (RUN_DIR / "scores.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0])
    writer.writeheader(); writer.writerows(rows)

for split in ("typed", "hand-written", "overall"):
    group = rows if split == "overall" else [row for row in rows if row["split"] == split]
    print(split, {key: round(sum(row[key] for row in group) / len(group), 4)
                  for key in ("CER_raw", "WER_raw", "exact_line_accuracy", "CER_normalized", "WER_normalized", "exact_normalized")})
```

## 5. Compare and decide

Rank page models by normalized CER on typed and handwritten samples separately,
then manually inspect the worst three pages per model.  Record failures in RTL
order, Persian `ی`/`ک`, ZWNJ, Persian versus Arabic digits, mixed Persian/Latin
order, hallucinated text, and missing lines.  Keep document structure (tables,
layout, Markdown) as a separate qualitative/result column; do not average it
into CER.

Before selecting a production model, expand this 20-image smoke set with the
document types listed in `models.yaml`: low-resolution scans, phone photos,
multi-column RTL, mixed-language pages, tables/forms, historical print, and
math.

Google Drive mounting requires explicit authorization and Colab GPU/runtime
availability changes over time; see the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
