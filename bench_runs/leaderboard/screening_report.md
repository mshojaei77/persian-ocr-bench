# Phase 1 screening report

Dataset: `smoke20-v1` (`9543aff102da6cb4b3df1c4ac0f3c3a7dcf73c302233e8281fe68ad7411c28b7`)

Protocol: `phase1_screening` / `full_page_ocr` (`22b8b6a898198bc4255324e8f07767ab3ea874f718dbc6b410a9abaae4b0fb72`)

This 20-image phase is a viability screen, not a general Persian OCR ranking.

| Decision | Model | Attempts | CER eligible | Norm CER | Raw CER | WER | BoW WER | Order gap | Faith F1 | Exact pages | Mean sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hold | `easyocr_fa` | 20/20 | 17/20 | 0.4598 (0.3563-0.5598) | 0.49881 | 0.834228 | 1.402408 | 0.0 | 0.241238 | 0.0 | 11.639 |
| Hold | `ppocrv5_arabic_mobile_rec` | 20/20 | 17/20 | 0.5888 (0.4422-0.7306) | 0.618809 | 0.841573 | 1.160315 | 0.0 | 0.227477 | 0.0 | 9.374 |
| Hold | `tesseract_fas` | 20/20 | 17/20 | 0.5760 (0.4262-0.7128) | 0.584702 | 0.784347 | 1.086734 | 0.0 | 0.289751 | 0.0 | 0.693 |

## Metric glossary

All rows use the same dataset, protocol, scorer, and capability class. Values are not a final ranking until the human-review and large-benchmark gates pass.

| Metric | Meaning | Direction |
|---|---|---|
| Attempts / CER eligible | All successful attempts / pages eligible for official CER/WER. Metric exclusions are never silently treated as perfect scores. | Higher is better; complete attempts and disclosed eligibility are required. |
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

{"artifacts_compatible": 3, "artifacts_complete": 3, "artifacts_seen": 3, "artifacts_unique": 3, "artifacts_valid": 3, "duplicate_runs": 0, "errors": 0, "warnings": 0}
