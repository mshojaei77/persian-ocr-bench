# Persian OCR Benchmark

Small benchmark project for comparing Persian OCR and vision-language OCR models
on a fixed 20-image dataset.

The repo contains:

- `small_bench/typed/`: 10 printed Persian samples.
- `small_bench/hand-written/`: 10 handwritten Persian samples.
- Matching `.md` reference transcriptions for every `.jpg`.
- `scripts/pull_*_model.py`: download model weights into `models/`.
- `scripts/benchmark_*.py`: run a model, save predictions, and score them.
- `BENCHMARK_GUIDE.md`: metric and benchmark design notes.

Generated model weights and benchmark outputs are intentionally local artifacts:
`models/` is ignored by Git, and benchmark outputs are written under
`bench_runs/`.

## What The Benchmark Measures

Each benchmark script reads images from `small_bench/`, writes model predictions
to `bench_runs/<model>/`, and writes:

- `scores.csv`: per-image CER, WER, line exact score, and text lengths.
- `summary.csv`: mean metrics for `typed`, `hand-written`, and `all`.
- `run_info.json`: model path, prompt, runtime options, and source notes when
  the script records them.
- `_raw/`: raw model outputs where supported.

Core metrics:

- `CER`: character error rate. This is the main Persian OCR metric.
- `WER`: word error rate.
- `line_exact_norm`: percentage of normalized reference lines found exactly in
  the prediction.

## Local Setup

This project uses `uv`.

```powershell
uv sync
```

Run a benchmark from the repo root:

```powershell
uv run python scripts/benchmark_easyocr_fa.py --gpu
```

If a benchmark needs local Hugging Face weights, download them first:

```powershell
uv run python scripts/pull_qwen3_vl_persian_arabic_ocr_model.py
uv run python scripts/benchmark_qwen3_vl_persian_arabic_ocr.py
```

Score an existing run without rerunning inference:

```powershell
uv run python scripts/benchmark_qwen3_vl_persian_arabic_ocr.py --score-only
```

## Useful Starting Points

Fastest first run:

```powershell
uv add easyocr
uv run python scripts/benchmark_easyocr_fa.py --gpu
```

Local VLM OCR run:

```powershell
uv run python scripts/pull_qwen3_vl_persian_arabic_ocr_model.py
uv run python scripts/benchmark_qwen3_vl_persian_arabic_ocr.py
```

Surya OCR 2 run:

```powershell
uv add surya-ocr vllm
uv run python scripts/pull_surya_model.py
uv run python scripts/benchmark_surya.py
```

Some scripts need dependencies that are not currently listed in
`pyproject.toml`. Add them with `uv add <package>` when you choose that model.
The base project currently includes the common Hugging Face and Transformers
stack.

## Google Colab Guide

Use a GPU runtime:

1. Open Colab.
2. Go to `Runtime -> Change runtime type`.
3. Select a GPU.
4. Run the cells below.

Colab normally runs its own Python environment. Because this repo currently
requires Python `>=3.13` in `pyproject.toml`, do not start with `uv sync` in
Colab unless the runtime Python satisfies that requirement. The simpler Colab
path is to install the needed packages into the active notebook environment.

### 1. Clone The Repo

Replace the URL with your GitHub repo URL after you push this project.

```python
!git clone https://github.com/mshojaei77/persian-ocr-bench-bench.git
%cd persian-ocr-bench
```

If you uploaded the project to Google Drive instead:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/persian-ocr-bench
```

### 2. Install Runtime Dependencies

Minimal Hugging Face/VLM stack:

```python
!pip install -U accelerate bitsandbytes hf-xet huggingface-hub peft pillow qwen-vl-utils safetensors transformers
```

For EasyOCR:

```python
!pip install -U easyocr
```

For Surya OCR 2:

```python
!pip install -U surya-ocr vllm
```

### 3. Run A Small Benchmark

EasyOCR is the simplest first smoke run:

```python
!python scripts/benchmark_easyocr_fa.py --gpu
```

Show the summary:

```python
import pandas as pd
pd.read_csv("bench_runs/easyocr-fa/summary.csv")
```

### 4. Run A Hugging Face VLM Benchmark

Download the model:

```python
!python scripts/pull_qwen3_vl_persian_arabic_ocr_model.py --max-workers 4
```

Run inference and scoring:

```python
!python scripts/benchmark_qwen3_vl_persian_arabic_ocr.py
```

View results:

```python
import pandas as pd
pd.read_csv("bench_runs/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0/summary.csv")
```

### 5. Save Results To Drive

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/persian-ocr-bench-results
!cp -r bench_runs /content/drive/MyDrive/persian-ocr-bench-results/
```

## Output Layout

After running a model:

```text
bench_runs/
  MODEL_NAME/
    typed/
      1.md
      ...
    hand-written/
      1.md
      ...
    _raw/
    scores.csv
    summary.csv
    run_info.json
```

## Notes

- Keep every model on the same `small_bench/` images for fair comparisons.
- Use `--score-only` when you edit prediction files manually or restore a saved
  `bench_runs/` folder.
- GPU memory needs vary by model. Start with EasyOCR or a smaller OCR/VLM model
  before trying the larger checkpoints.
- See `BENCHMARK_GUIDE.md` for benchmark expansion rules and recommended
  leaderboard format.
