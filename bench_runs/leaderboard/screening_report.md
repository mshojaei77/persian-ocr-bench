# Phase 1 screening report

Dataset: `smoke20-v1` (`9543aff102da6cb4b3df1c4ac0f3c3a7dcf73c302233e8281fe68ad7411c28b7`)

Protocol: `phase1_screening` / `full_page_ocr` (`22b8b6a898198bc4255324e8f07767ab3ea874f718dbc6b410a9abaae4b0fb72`)

This 20-image phase is a viability screen, not a general Persian OCR ranking.

| Decision | Model | Coverage | Norm CER | Raw CER | WER | BoW WER | Order gap | Faith F1 | Exact pages | Yeh recall | Kaf recall | ZWNJ F1 | Mean sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hold | `easyocr_fa` | 1/20 | 0.5524 (-) | 0.604341 | 0.946903 | 1.690265 | 0.0 | 0.068293 | 0.0 | 0.512195 | 0.2 | 0.0 | 10.304 |
| Reject | `ppocrv5_arabic_mobile_rec` | 0/20 | - | - | - | - | - | - | - | - | - | - | - |
| Blocked | `tesseract_fas` | 20/20 | 0.5760 (0.4262-0.7128) | 0.584702 | 0.784347 | 1.086734 | 0.0 | 0.289751 | 0.0 | 0.466206 | 0.435414 | 0.286608 | 0.706 |

## Metric glossary

All rows use the same dataset, protocol, scorer, and capability class. Values are not a final ranking until the human-review and large-benchmark gates pass.

| Metric | Meaning | Direction |
|---|---|---|
| Coverage | Successful pages divided by expected pages. | Higher is better; complete coverage is required. |
| Norm CER / Raw CER | Macro page CER after frozen Persian normalization / exact Unicode CER after NFC only. | Lower is better. |
| P90 CER | 90th-percentile page CER; exposes difficult-page failures. | Lower is better. |
| Worst Q CER | Mean CER of the worst 25% of successful pages. | Lower is better. |
| WER / BoW WER / Order gap | Sequential WER, order-insensitive token error rate, and their positive difference. The gap isolates likely reading-order failures. | Lower is better. |
| Faith F1 | Token-level precision/recall F1; omissions and insertions remain visible in CSV/JSON. | Higher is better. |
| Exact pages | Fraction of pages with zero canonical CER. | Higher is better. |
| Yeh recall | Recall of Persian yeh characters in the orthographic slice. | Higher is better. |
| Kaf recall | Recall of Persian kaf characters in the orthographic slice. | Higher is better. |
| ZWNJ F1 | F1 score for zero-width non-joiner placement. | Higher is better. |
| Mean/P95 sec | Average and 95th-percentile end-to-end seconds per page. | Lower is better. |

Interpretation: use CER/WER for general transcription quality, P90 and Worst Q for reliability on hard pages, orthographic metrics for Persian-specific errors, and latency metrics for operational trade-offs. Do not select a winner from one metric alone.

## Validation

{"artifacts_compatible": 2, "artifacts_complete": 0, "artifacts_seen": 3, "artifacts_unique": 2, "artifacts_valid": 2, "duplicate_runs": 0, "errors": 3, "warnings": 1}
