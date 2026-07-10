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

Some model scripts need extra packages that are not in the base environment.
Add them only when you pick that model, for example:

```powershell
uv add easyocr
uv add surya-ocr vllm
```

## Pull And Benchmark One Model

Run from the repo root. Pull one model, benchmark it, then inspect its output
before moving to the next model:

```powershell
uv run python scripts/pull_easyocr_fa_model.py
uv run python scripts/benchmark_easyocr_fa.py --gpu
Import-Csv bench_runs/easyocr-fa/summary.csv | Format-Table
```

If a benchmark already has predictions and you only want to recompute scores:

```powershell
uv run python scripts/benchmark_easyocr_fa.py --score-only
```

Each benchmark writes:

```text
bench_runs/<model>/
  typed/*.md
  hand-written/*.md
  _raw/
  scores.csv
  summary.csv
  run_info.json
```

Inspect the run in this order:

```powershell
Import-Csv bench_runs/<model>/summary.csv | Format-Table
Import-Csv bench_runs/<model>/scores.csv | Sort-Object cer_norm -Descending | Select-Object -First 5 | Format-Table
Get-Content bench_runs/<model>/run_info.json
```

Then open the worst prediction beside its reference:

```powershell
Get-Content small_bench/typed/1.md
Get-Content bench_runs/<model>/typed/1.md
```

## Commands For Every Model

Run one block at a time. Stop after each block and inspect the summary before
pulling the next model.

### `easyocr_fa`

```powershell
uv run python scripts/pull_easyocr_fa_model.py
uv run python scripts/benchmark_easyocr_fa.py --gpu
Import-Csv bench_runs/easyocr-fa/summary.csv | Format-Table
```

### `tesseract_fas`

```powershell
uv run python scripts/pull_tesseract_fas_model.py
uv run python scripts/benchmark_tesseract_fas.py
Import-Csv bench_runs/tesseract5-fas/summary.csv | Format-Table
```

### `kraken_fas`

```powershell
uv run python scripts/pull_kraken_fas_model.py
uv run python scripts/benchmark_kraken_fas.py
Import-Csv bench_runs/kraken-fas/summary.csv | Format-Table
```

### `hezarai_crnn_fa_v2`

```powershell
uv run python scripts/pull_hezarai_crnn_fa_v2_model.py
uv run python scripts/benchmark_hezarai_crnn_fa_v2.py
Import-Csv bench_runs/hezarai-crnn-base-fa-v2/summary.csv | Format-Table
```

### `chandra_ocr2`

```powershell
uv run python scripts/pull_chandra_ocr2_model.py
uv run python scripts/benchmark_chandra_ocr2.py
Import-Csv bench_runs/chandra-ocr-2/summary.csv | Format-Table
```

### `deepseek_ocr`

```powershell
uv run python scripts/pull_deepseek_ocr_model.py
uv run python scripts/benchmark_deepseek_ocr.py
Import-Csv bench_runs/DeepSeek-OCR/summary.csv | Format-Table
```

### `deepseek_ocr2`

```powershell
uv run python scripts/pull_deepseek_ocr2_model.py
uv run python scripts/benchmark_deepseek_ocr2.py
Import-Csv bench_runs/DeepSeek-OCR-2/summary.csv | Format-Table
```

### `dots_mocr`

```powershell
uv run python scripts/pull_dots_mocr_model.py
uv run python scripts/benchmark_dots_mocr.py
Import-Csv bench_runs/DotsMOCR/summary.csv | Format-Table
```

### `dots_ocr`

```powershell
uv run python scripts/pull_dots_ocr_model.py
uv run python scripts/benchmark_dots_ocr.py
Import-Csv bench_runs/DotsOCR/summary.csv | Format-Table
```

### `falcon_ocr`

```powershell
uv run python scripts/pull_falcon_ocr_model.py
uv run python scripts/benchmark_falcon_ocr.py
Import-Csv bench_runs/Falcon-OCR/summary.csv | Format-Table
```

### `glm_ocr`

```powershell
uv run python scripts/pull_glm_ocr_model.py
uv run python scripts/benchmark_glm_ocr.py
Import-Csv bench_runs/GLM-OCR/summary.csv | Format-Table
```

### `got_ocr2`

```powershell
uv run python scripts/pull_got_ocr2_model.py
uv run python scripts/benchmark_got_ocr2.py
Import-Csv bench_runs/GOT-OCR-2.0-hf/summary.csv | Format-Table
```

### `hunyuan_ocr`

```powershell
uv run python scripts/pull_hunyuan_ocr_model.py
uv run python scripts/benchmark_hunyuan_ocr.py
Import-Csv bench_runs/HunyuanOCR/summary.csv | Format-Table
```

### `infinity_parser2_flash`

```powershell
uv run python scripts/pull_infinity_parser2_flash_model.py
uv run python scripts/benchmark_infinity_parser2_flash.py
Import-Csv bench_runs/Infinity-Parser2-Flash/summary.csv | Format-Table
```

### `infinity_parser2_pro`

```powershell
uv run python scripts/pull_infinity_parser2_pro_model.py
uv run python scripts/benchmark_infinity_parser2_pro.py
Import-Csv bench_runs/Infinity-Parser2-Pro/summary.csv | Format-Table
```

### `kdl_frontier_parser_nano`

```powershell
uv run python scripts/pull_kdl_frontier_parser_nano_model.py
uv run python scripts/benchmark_kdl_frontier_parser_nano.py
Import-Csv bench_runs/KDL-Frontier-Parser-nano/summary.csv | Format-Table
```

### `khanandeh`

```powershell
uv run python scripts/pull_khanandeh_model.py
uv run python scripts/benchmark_khanandeh.py
Import-Csv bench_runs/Khanandeh-0.1-Persian-OCR-2B-Instruct/summary.csv | Format-Table
```

### `lightonocr2_1b`

```powershell
uv run python scripts/pull_lightonocr2_1b_model.py
uv run python scripts/benchmark_lightonocr2_1b.py
Import-Csv bench_runs/LightOnOCR-2-1B/summary.csv | Format-Table
```

### `mineru25_pro`

```powershell
uv run python scripts/pull_mineru25_pro_model.py
uv run python scripts/benchmark_mineru25_pro.py
Import-Csv bench_runs/MinerU2.5-Pro-2605-1.2B/summary.csv | Format-Table
```

### `nanonets_ocr2_15b`

```powershell
uv run python scripts/pull_nanonets_ocr2_15b_model.py
uv run python scripts/benchmark_nanonets_ocr2_15b.py
Import-Csv bench_runs/Nanonets-OCR2-1.5B-exp/summary.csv | Format-Table
```

### `paddleocr_vl`

```powershell
uv run python scripts/pull_paddleocr_vl_model.py
uv run python scripts/benchmark_paddleocr_vl.py
Import-Csv bench_runs/PaddleOCR-VL-0.9B/summary.csv | Format-Table
```

### `ppocrv5_fa`

```powershell
uv run python scripts/pull_ppocrv5_fa_model.py
uv run python scripts/benchmark_ppocrv5_fa.py
Import-Csv bench_runs/PP-OCRv5-fa/summary.csv | Format-Table
```

### `qwen3_vl_persian_arabic_ocr`

```powershell
uv run python scripts/pull_qwen3_vl_persian_arabic_ocr_model.py
uv run python scripts/benchmark_qwen3_vl_persian_arabic_ocr.py
Import-Csv bench_runs/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0/summary.csv | Format-Table
```

### `surya`

```powershell
uv run python scripts/pull_surya_model.py
uv run python scripts/benchmark_surya.py
Import-Csv bench_runs/surya-ocr-2/summary.csv | Format-Table
```

### `unlimited_ocr_gguf`

```powershell
uv run python scripts/pull_unlimited_ocr_gguf_model.py
uv run python scripts/benchmark_unlimited_ocr_gguf.py
Import-Csv bench_runs/Unlimited-OCR-GGUF-Q4_K_M/summary.csv | Format-Table
```

### `weightedai_persian_ocr`

```powershell
uv run python scripts/pull_weightedai_persian_ocr_model.py
uv run python scripts/benchmark_weightedai_persian_ocr.py
Import-Csv bench_runs/WeightedAI-Persian_OCR/summary.csv | Format-Table
```

## Run Order

Start with small or already-installed engines, then move to heavier VLMs:

1. `easyocr_fa`
2. `tesseract_fas`
3. `kraken_fas`
4. `hezarai_crnn_fa_v2`
5. the VLM rows, one at a time

Keep every model on the same `small_bench/` images. Compare `typed`,
`hand-written`, and `all` rows in `summary.csv`, then inspect the worst
per-image rows in `scores.csv`. CER is the main metric; WER and exact-line
score are sanity checks for readability and layout.

See `BENCHMARK_GUIDE.md` for metric details and benchmark expansion rules.
