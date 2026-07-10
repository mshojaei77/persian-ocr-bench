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

## Available Model Scripts

Use one row at a time. The pull script downloads or warms local assets under
`models/`; the benchmark script writes results under `bench_runs/`.

| model key | pull script | benchmark script |
| --- | --- | --- |
| `chandra_ocr2` | `scripts/pull_chandra_ocr2_model.py` | `scripts/benchmark_chandra_ocr2.py` |
| `deepseek_ocr` | `scripts/pull_deepseek_ocr_model.py` | `scripts/benchmark_deepseek_ocr.py` |
| `deepseek_ocr2` | `scripts/pull_deepseek_ocr2_model.py` | `scripts/benchmark_deepseek_ocr2.py` |
| `dots_mocr` | `scripts/pull_dots_mocr_model.py` | `scripts/benchmark_dots_mocr.py` |
| `dots_ocr` | `scripts/pull_dots_ocr_model.py` | `scripts/benchmark_dots_ocr.py` |
| `easyocr_fa` | `scripts/pull_easyocr_fa_model.py` | `scripts/benchmark_easyocr_fa.py` |
| `falcon_ocr` | `scripts/pull_falcon_ocr_model.py` | `scripts/benchmark_falcon_ocr.py` |
| `glm_ocr` | `scripts/pull_glm_ocr_model.py` | `scripts/benchmark_glm_ocr.py` |
| `got_ocr2` | `scripts/pull_got_ocr2_model.py` | `scripts/benchmark_got_ocr2.py` |
| `hezarai_crnn_fa_v2` | `scripts/pull_hezarai_crnn_fa_v2_model.py` | `scripts/benchmark_hezarai_crnn_fa_v2.py` |
| `hunyuan_ocr` | `scripts/pull_hunyuan_ocr_model.py` | `scripts/benchmark_hunyuan_ocr.py` |
| `infinity_parser2_flash` | `scripts/pull_infinity_parser2_flash_model.py` | `scripts/benchmark_infinity_parser2_flash.py` |
| `infinity_parser2_pro` | `scripts/pull_infinity_parser2_pro_model.py` | `scripts/benchmark_infinity_parser2_pro.py` |
| `kdl_frontier_parser_nano` | `scripts/pull_kdl_frontier_parser_nano_model.py` | `scripts/benchmark_kdl_frontier_parser_nano.py` |
| `khanandeh` | `scripts/pull_khanandeh_model.py` | `scripts/benchmark_khanandeh.py` |
| `kraken_fas` | `scripts/pull_kraken_fas_model.py` | `scripts/benchmark_kraken_fas.py` |
| `lightonocr2_1b` | `scripts/pull_lightonocr2_1b_model.py` | `scripts/benchmark_lightonocr2_1b.py` |
| `mineru25_pro` | `scripts/pull_mineru25_pro_model.py` | `scripts/benchmark_mineru25_pro.py` |
| `nanonets_ocr2_15b` | `scripts/pull_nanonets_ocr2_15b_model.py` | `scripts/benchmark_nanonets_ocr2_15b.py` |
| `paddleocr_vl` | `scripts/pull_paddleocr_vl_model.py` | `scripts/benchmark_paddleocr_vl.py` |
| `ppocrv5_fa` | `scripts/pull_ppocrv5_fa_model.py` | `scripts/benchmark_ppocrv5_fa.py` |
| `qwen3_vl_persian_arabic_ocr` | `scripts/pull_qwen3_vl_persian_arabic_ocr_model.py` | `scripts/benchmark_qwen3_vl_persian_arabic_ocr.py` |
| `surya` | `scripts/pull_surya_model.py` | `scripts/benchmark_surya.py` |
| `tesseract_fas` | `scripts/pull_tesseract_fas_model.py` | `scripts/benchmark_tesseract_fas.py` |
| `unlimited_ocr_gguf` | `scripts/pull_unlimited_ocr_gguf_model.py` | `scripts/benchmark_unlimited_ocr_gguf.py` |
| `weightedai_persian_ocr` | `scripts/pull_weightedai_persian_ocr_model.py` | `scripts/benchmark_weightedai_persian_ocr.py` |

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
