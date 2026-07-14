# Persian OCR smoke dataset

The 20 samples in `data/smoke/` implement Phase 1 of `docs/plan.md`. The canonical labels and SHA-256 checksums are in `data/manifest.jsonl`.

Most samples are deterministically rendered so punctuation, spaces, mixed scripts, digits, and U+200C zero-width non-joiners have exact ground truth. `smoke_006` is cropped from page 10 of Saeed Nafisi's 1953 book *Babak*. `smoke_019` uses another line from that page with a small deterministic bleed-through simulation. The source is retained in `assets/sources/` for auditability.

The handwriting cases are synthetic diagnostic samples rendered with handwriting-like OFL fonts; they are not represented as independent human writers. They should be replaced or supplemented by consented, writer-identified real handwriting in the development and private-test sets.

The labels have passed automated and visual construction QA, but the manifest deliberately marks them `needs_human_review` and `adjudicated: false`. A human must verify each image against `reference_raw` before this corpus is frozen as benchmark data.

## Rebuild

```powershell
uv sync
uv run python scripts/build_smoke_images.py --force
uv run python scripts/validate_smoke_images.py
```

The builder refuses to overwrite existing images unless `--force` is supplied. All stored masters are PNG, including `smoke_007`; that sample undergoes a low-quality JPEG round trip before being saved losslessly as the benchmark master.

## Sources and licenses

- Noto Naskh Arabic and Noto Sans Arabic are distributed under the SIL Open Font License; see `assets/fonts/OFL-Noto.txt`.
- Saeed Nafisi, *Babak* (1953), page 10, from the Indian Digital Library Project via Wikimedia Commons: <https://commons.wikimedia.org/wiki/File:Babak_by_Nafisi.djvu>. Wikimedia Commons identifies the work as public domain in Iran.
