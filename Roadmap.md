# Persian OCR Benchmark Roadmap

Audit date: 2026-07-13  
Scope: the live working tree, `models.yaml`, `small_bench`, committed benchmark artifacts, packaging, and the next model-adapter step.

## Executive judgment

The project has a thoughtful Persian-specific metric foundation and unusually honest model-support notes, but it is not yet a trustworthy model leaderboard or a runnable 26-model benchmark suite. Today it is best described as a 20-page smoke benchmark with two implemented engines and a 26-entry research catalog. The immediate goal should be to make that statement true, reproducible, and safe before expanding the model count.

The two current point estimates do not establish a winner. Tesseract has macro CER `0.5126` with a page-bootstrap 95% interval of `0.3913–0.6279`; PP-OCRv5 has macro CER `0.5655` with `0.4403–0.6889`. The intervals overlap substantially, and both artifacts point to deleted reference files, so the existing rank 1/rank 2 display overstates the evidence.

## Ten critical opinions

| # | Severity | Critical opinion | Evidence and impact | Required fix |
|---:|---|---|---|---|
| 1 | Critical | The package definition points at code that is deleted in the live tree. | `pyproject.toml` builds `src/persian_ocr`, while the active modules are flat under `src/`. There is no stable installed CLI, README metadata, or license metadata. | Choose one canonical package layout, add console entry points and project metadata, then build and smoke-install the wheel. |
| 2 | Critical | The leaderboard's default root escapes the repository. | Flat `src/leaderboard.py` uses `Path(__file__).resolve().parent.parent.parent`, resolving its default `bench_runs` outside this project. | Centralize `REPO_ROOT`, reject accidental path escape, and smoke the default CLI from both repo root and another working directory. |
| 3 | Critical | The only migration script can erase both reference corpora. | `scripts/merge_small_bench.py` overwrites the JSON files from `*.md`, but the repository now contains zero Markdown references. A normal run would write two empty arrays. | Make conversion fail closed on zero inputs, validate image/reference cardinality, write atomically, and add a non-mutating `--check` mode. Decide whether the stale ZIP is canonical or archival. |
| 4 | Critical | Ground-truth provenance is missing and currently misrepresented. | The manifest has only image/track-style metadata; the loader labels migrated JSON text as `reviewed`, while both committed artifacts describe it as `unreviewed_markdown` and cite deleted `.md` paths. | Add source, license, transcription method, annotator/reviewer, review status, and per-sample/reference hashes. Publish a versioned dataset card and do not upgrade quality without recorded review. |
| 5 | Critical | Stale and non-comparable artifacts are ranked together. | The leaderboard does not require matching dataset, scorer, code, prompt/config, model revision, or hardware fingerprints. The current artifacts cannot resolve their own reference paths. | Define a run identity, reject incompatible artifacts, freeze the dataset, and regenerate every baseline against the same protocol. |
| 6 | High | `models.yaml` was a reading list, not an execution registry. | It contains 26 cleanly structured entries, but only two had an adapter/artifact and no runtime loaded the file. Eighteen entries were P0, including ten without explicit Persian evidence. | Keep catalog metadata separate from lifecycle state, validate it automatically, pin components/revisions/licenses, and use a small staged queue. The first registry/status slice is now implemented. |
| 7 | High | The benchmark core is coupled to Tesseract and will not scale to 26 adapters. | The 1,243-line Tesseract module owns dataset IO, normalization, metrics, preprocessing, downloads, reporting, and CLI behavior; PP-OCR and EasyOCR reuse it as their benchmark core. | Extract single-responsibility `dataset`, `normalization`, `metrics`, `artifacts`, and adapter modules after the package layout is settled. Preserve the current output contract during extraction. |
| 8 | Critical | The declared protocol is much larger than the annotated evidence. | `models.yaml` names nine document slices plus RTL/layout/table/formula metrics, but the corpus has 20 pages across four tracks and no boxes, line-order graph, table cells, formulas, or source/license annotations. | Rename the current set `smoke20`; create versioned recognition and document-structure suites with explicit Persian/English, digits, ZWNJ, multi-column RTL, layouts, forms, tables, photos, degradation, and handwriting annotations. |
| 9 | High | The typed/handwritten leaderboard misclassifies pages. | `collect_track_rows` calls every non-`hand` track `typed`, producing 12 typed rows although only 10 images are in the typed split. | Carry explicit split/category fields from the manifest into results and aggregate exact categories without a substring fallback. Validate output counts against the manifest. |
| 10 | High | Ranking logic overstates small-sample results and mishandles perfect CER. | It ranks macro-CER point estimates, ignores the available confidence intervals, and uses `cer or infinity`, which sends valid `0.0` CER to the bottom. | Fix `None` handling, show macro and micro CER with intervals, use paired resampling on identical pages, declare ties/insufficient evidence, and gate ranks on complete comparable runs. |

## Implementation roadmap

### Phase 0 — Protect the current data (issues 3–5)

- Disable destructive behavior in `merge_small_bench.py`: zero-input guard, `--check`, atomic writes, and exact 20-image coverage checks.
- Declare which of `small_bench.zip`, split JSON, and the deleted Markdown files is authoritative.
- Record the current images and references in a versioned manifest with SHA-256 digests.
- Mark all current references unreviewed until a human review record exists.

Exit criteria: running every maintenance command without explicit write flags cannot change reference files; all 20 images resolve to one non-empty reference and provenance record.

### Phase 1 — Repair installability and paths (issues 1–2)

- Restore one canonical `src/persian_ocr/` package and remove flat/package duplication only after preserving the live implementations.
- Add console scripts for model status, each adapter, and the leaderboard.
- Add README, license metadata, supported Python range, and package-data rules for `models.yaml`.
- Centralize path resolution and test commands from the repo root and an unrelated working directory.

Exit criteria: `uv build` succeeds, a clean temporary environment installs the wheel, all console scripts show help, and default outputs remain inside this repository.

### Phase 2 — Freeze benchmark identity (issues 4–5)

- Introduce `persian_ocr_benchmark_v2` with:
  - dataset/manifest/reference digests;
  - catalog version and digest;
  - runner/scorer version and Git commit plus dirty state;
  - full model/component revisions and file hashes;
  - exact preprocessing, prompt, decoding, precision, and device configuration;
  - package versions, hardware, success/failure counts, and raw structured outputs.
- Make the leaderboard reject or separate mismatched run identities.
- Regenerate Tesseract and PP-OCR after the data freeze.

Exit criteria: a result is independently traceable to immutable model, data, code, and configuration inputs; deleted or changed references make validation fail.

### Phase 3 — Extract the benchmark core (issue 7)

- Create focused modules for dataset loading, Persian normalization, scoring, provenance/artifacts, preprocessing, and adapter protocol.
- Keep model adapters responsible only for pull/load/infer and structured model output.
- Preserve raw polygons/confidences before producing any reading-order text view.
- Add smoke scripts, not pytest, for schema compatibility and one-image execution.

Exit criteria: Tesseract, PP-OCR, and EasyOCR use the same core without importing one another; a schema/status smoke command detects adapter drift.

### Phase 4 — Make the model registry executable (issue 6)

- Extend each model entry with lifecycle status, adapter, optional dependency group, component repository IDs, full revisions, license evidence dates, trust-remote-code policy, hardware envelope, prompt revision, supported tracks, and expected artifacts.
- Separate catalog order, Persian-evidence strength, benchmark priority, implementation state, and measured leaderboard rank.
- Reduce P0 to a diverse first wave rather than 18 nominally mandatory models.
- Run `scripts/models_status.py` in every maintenance workflow.

Exit criteria: catalog state agrees with the filesystem/artifacts, every implemented pipeline is fully pinned, and the next action is derived rather than guessed.

### Phase 5 — Correct statistical and category reporting (issues 9–10)

- Preserve exact split and track names end to end.
- Fix zero-CER sorting and require complete comparable results.
- Add paired page-level bootstrap differences and an explicit tie/insufficient-evidence state.
- Report recognition, document structure, and operations in separate leaderboards.

Exit criteria: synthetic 0.0 CER sorts first, counts reconcile to the manifest, mismatched runs are excluded, and overlapping uncertainty is not presented as a decisive rank.

### Phase 6 — Expand from smoke20 to a Persian evaluation suite (issue 8)

- Keep `smoke20` as the fast adapter gate.
- Build a larger held-out recognition set stratified by source, typography, capture, degradation, handwriting, mixed script, numerals, ZWNJ, and document type.
- Build a separate layout set with polygons, RTL reading-order edges, blocks, tables/forms, and formulas.
- Add omission, insertion/hallucination, repeated-output, exact digit/entity, block coverage, and failure-rate metrics alongside CER/WER.
- Document licensing and prevent test-set contamination when fine-tuning.

Exit criteria: each reported metric has an annotated eligible slice, confidence interval, and minimum sample threshold; model selection no longer depends on 20 pages.

## Model execution queue

1. `tesseract_fas` — deterministic CPU floor; regenerate after dataset v1 is frozen.
2. `ppocrv5_arabic_mobile_rec` — compact detector plus recognizer baseline; regenerate under artifact v2.
3. `easyocr_fa` — immediate next adapter; implemented in this change as an aging neural baseline, not a production ceiling.
4. `hezar_crnn_base_fa_v2` — Persian specialist; requires an explicit page detector and separate line-recognition evaluation.
5. `surya_ocr_2` — first compact full-page VLM with explicit Persian evidence.
6. `qwen3_vl_2b_persian_arabic_ocr` versus the exact `qwen3_vl_2b_instruct` base — paired fine-tune attribution.
7. `chandra_ocr_2` — higher-cost Persian ceiling on suitable GPU hardware.

Do not put recognizer-only checkpoints and full-page document VLMs into one aggregate quality rank. Keep raw recognition, reading order/document structure, and operations as separate decisions.

## Current-practice basis

- [EasyOCR official repository](https://github.com/JaidedAI/EasyOCR) and [stable PyPI release](https://pypi.org/project/easyocr/) support `fa`, polygon/text/confidence output, and configurable model storage. The stable release is old enough that it should be treated as a baseline.
- [Hugging Face download guidance](https://huggingface.co/docs/huggingface_hub/en/guides/download) supports full commit revisions; URLs without revisions are not reproducible model identities.
- [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards) provide a useful minimum structure for provenance, licensing, and intended-use documentation.
- [KITAB-Bench](https://github.com/mbzuai-oryx/KITAB-Bench), [OmniDocBench](https://github.com/opendatalab/OmniDocBench), and [olmOCR-bench](https://github.com/jina-ai/olmocr-bench) reinforce multi-domain/task-specific evaluation, separate layout/recognition metrics, reading-order checks, and fidelity checks beyond CER.
- Community reports on [Hacker News](https://news.ycombinator.com/item?id=43174298) and [Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1qiyxl4/we_tested_every_vlm_for_arabic_document/) repeatedly mention omissions, invented substitutions, repetition loops, and RTL handwriting failures. These are anecdotal signals for adversarial test design, not evidence for model ranking.
- [PyPA package layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) supports choosing one intentional import/package layout rather than maintaining two divergent copies.

## Commands for the next checkpoint

```powershell
uv run python scripts/models_status.py --all
uv run python scripts/models_status.py --strict
uv sync --extra easyocr
uv run --extra easyocr python src/easyocr_fa.py --pull --device cpu
uv run --extra easyocr python src/easyocr_fa.py --small_bench --limit 1 --device cpu --output bench_runs/validation/easyocr_fa_smoke.json
uv run --extra easyocr python src/easyocr_fa.py --small_bench --device cpu --output bench_runs/easyocr_fa.json
```

The strict registry command is expected to fail until the stale Tesseract/PP-OCR artifact provenance is regenerated; that failure is a guardrail, not a reason to weaken validation.
