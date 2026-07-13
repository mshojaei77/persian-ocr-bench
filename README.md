# Persian OCR benchmark

This repository screens Persian OCR systems before investing in a larger,
sealed evaluation. The current 20-image `smoke20-v1` corpus is **Phase 1 only**:
it is a viability gate, not a final model ranking or production-quality
leaderboard.

The complete execution plan is tracked in [ROADMAP.md](ROADMAP.md), including
the remaining P0 adapters, human-review gate, sealed large benchmark, and
separate final leaderboards.

The Python package provides one shared contract for dataset validation, Persian
normalization, recognition metrics, preprocessing, run identity, and portable
benchmark artifacts. Model runtimes are optional so catalog and dataset checks
do not install every OCR framework.

## Setup

Install the lightweight core:

```powershell
uv sync
uv run persian-ocr models status
uv run persian-ocr dataset validate
uv run persian-ocr dataset identity
```

Install only the runtime needed for a model:

```powershell
uv sync --extra tesseract
uv sync --extra paddle
uv sync --extra easyocr
uv sync --extra plots
```

Neural adapters use `--device auto` by default: they choose CUDA first (then
Apple MPS where supported) and fall back to CPU. This policy can only use a
GPU when the installed framework is a GPU-enabled build. The locked Paddle
extra currently provides the portable CPU build; for GPU runs install the
Paddle GPU wheel matching the host CUDA version from the [official installer]
before running the adapter. Verify the active runtime with the commands below.

[official installer]: https://www.paddlepaddle.org.cn/install/quick

Use `uv sync --extra all` only when the same environment genuinely needs every
implemented engine and plotting support. Exact resolved versions remain in
`uv.lock`.

## Phase 1 compatibility commands

The existing model adapters remain directly runnable during the package
migration:

```powershell
uv run --extra tesseract python src/tesseract_fas.py --small_bench
uv run --extra paddle python src/ppocrv5_arabic_mobile_rec.py --small_bench
uv run --extra easyocr python src/easyocr_fa.py --small_bench
```

Treat Phase 1 outcomes as `advance`, `hold`, `reject`, or `blocked`. Do not mix
these artifacts with a later large-dataset evaluation. The v2 artifact identity
includes separate protocol, dataset, model, runner, configuration, and runtime
hashes to enforce that boundary.

## Validation without pytest

The project intentionally does not use pytest. Run the real wheel/install/CLI
smoke check instead:

```powershell
uv run python scripts/smoke_package.py
```

For strict human-review gating, run:

```powershell
uv run persian-ocr dataset validate --require-reviewed
```

That command is expected to fail until every reference has documented human
review. The default validation still checks manifest structure, content hashes,
image readability, and dataset identity.

## Workspace paths

Commands discover the checkout from the current directory. When running an
installed command elsewhere, pass `--workspace C:\path\to\persian-ocr` or set
`PERSIAN_OCR_WORKSPACE`. Artifact paths stored in JSON are relative POSIX paths;
machine-specific absolute paths are excluded from run identity.

## End-to-end command reference

Run these commands from the repository root. PowerShell uses the backtick (`` ` ``)
for line continuation; keeping a command on one line is also fine.

### Pull and inspect the catalog

The adapters pull their model files on first use and cache them under `models/`.
There is no separate pull command that bypasses an adapter's pinned model
identity.

```powershell
# Install the lightweight package and lockfile dependencies
uv sync

# Install one runtime before pulling/running that model
uv sync --extra tesseract
uv sync --extra paddle
uv sync --extra easyocr

# Install all implemented runtimes (large environment)
uv sync --extra all

# Inspect catalog status and the next implementation target
uv run persian-ocr models status --all
uv run persian-ocr models status --json
```

### Validate the dataset and identity

```powershell
uv run persian-ocr dataset validate
uv run persian-ocr dataset identity

# Strict gate: expected to fail until every reference has human review
uv run persian-ocr dataset validate --require-reviewed
```

### Pull and benchmark one model

Each adapter downloads missing weights, runs the selected images, and writes a
portable v2 artifact. `--limit 1` is the cheapest initialization smoke check.

```powershell
uv run --extra tesseract python src/tesseract_fas.py --small_bench --limit 1
uv run --extra paddle python src/ppocrv5_arabic_mobile_rec.py --small_bench --limit 1 --device auto
uv run --extra easyocr python src/easyocr_fa.py --small_bench --limit 1 --device auto
```

### Run the complete Phase 1 screen

The orchestrator is sequential and resumable. Successful models are skipped on
later runs; use `--force` when a model must be rerun.

```powershell
# Preview commands without running models
uv run python scripts/run_phase1.py --dry-run

# Run all implemented adapters over smoke20-v1
uv run python scripts/run_phase1.py

# Run one adapter, or rerun it from scratch
uv run python scripts/run_phase1.py --model easyocr_fa
uv run python scripts/run_phase1.py --model easyocr_fa --force

# Cheap one-page smoke run for every adapter
uv run python scripts/run_phase1.py --limit 1 --continue-on-error

# Neural adapters use GPU first (CUDA, then MPS), with CPU fallback.
uv run python scripts/run_phase1.py --device auto

# Verify the installed runtimes expose a GPU (without running OCR)
uv run python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
uv run --extra paddle python -c "import paddle; print('paddle:', paddle.__version__, 'cuda:', paddle.device.is_compiled_with_cuda(), 'gpus:', paddle.device.cuda.device_count())"

# Force CPU when needed
uv run python scripts/run_phase1.py --device cpu
```

Artifacts are written to `bench_runs/smoke20-v1/`; resumable state is in
`bench_runs/smoke20-v1/state.json` and adapter logs are in its `logs/` folder.

### Build the screening leaderboard/report

This is the Phase 1 screening report, not a final winner declaration. It emits
CER, WER, P90 and worst-quartile CER, exact-page rate, Persian yeh/kaf recall,
ZWNJ F1, coverage, and latency metrics. It also refreshes the historical
`leaderboard_*` filenames used by existing notebooks.

```powershell
uv run python scripts/build_screening_report.py `
  --input bench_runs/smoke20-v1 `
  --output bench_runs/leaderboard `
  --strict
```

Current outputs include:

```text
bench_runs/leaderboard/leaderboard.json
bench_runs/leaderboard/leaderboard.csv
bench_runs/leaderboard/leaderboard_by_type.csv
bench_runs/leaderboard/leaderboard_cer.png
bench_runs/leaderboard/leaderboard_accuracy_latency.png
bench_runs/leaderboard/screening_report.md
bench_runs/leaderboard/report_manifest.json
```

### Build the final compatible leaderboard

Final mode excludes incomplete, incompatible, unreviewed, or otherwise invalid
runs. It should only be used after the sealed large benchmark is frozen.

```powershell
uv run python src/leaderboard.py `
  --mode final `
  --input bench_runs/large-v1 `
  --output bench_runs/final-leaderboard `
  --strict
```

### Validate reports and package behavior

```powershell
uv run python scripts/smoke_reporting.py
uv run python scripts/validate_smoke20.py
uv run python scripts/smoke_package.py
uv run python -m compileall -q src scripts
git diff --check
```

Do not use pytest in this repository. Keep generated benchmark artifacts out of
the core install and update `uv.lock` whenever dependencies change.

## Rights and provenance

This repository does not grant an open-source license. Dataset and source
provenance are recorded explicitly where known; unknown third-party ownership
must be resolved before redistribution or public release. See [LICENSE](LICENSE).
