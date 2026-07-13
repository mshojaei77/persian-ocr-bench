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

## Rights and provenance

This repository does not grant an open-source license. Dataset and source
provenance are recorded explicitly where known; unknown third-party ownership
must be resolved before redistribution or public release. See [LICENSE](LICENSE).
