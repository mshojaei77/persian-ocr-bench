# Phase 1 screening report

Dataset: `smoke20-v1` (`8abf9b32bb2746d85b75c07049f35cebd245d42dc656c1c955a219cfd19a7dbe`)

Protocol: `phase1_screening` / `full_page_ocr` (`ce683321a491e5a64f2e6c54b8418820767279dee30b743aff7a14b30494bfc8`)

This 20-image phase is a viability screen, not a general Persian OCR ranking.

| Decision | Model | Coverage | Macro CER (95% CI) | P90 CER | Worst-quartile CER | Exact pages | Mean sec/image |
|---|---|---:|---:|---:|---:|---:|---:|
| Hold | `easyocr_fa` | 20/20 | 0.4534 (0.3596-0.5459) | 0.667674 | 0.703858 | - | 14.156 |
| Hold | `ppocrv5_arabic_mobile_rec` | 20/20 | 0.5546 (0.4252-0.6807) | 0.901278 | 0.89608 | - | 14.555 |
| Hold | `tesseract_fas` | 20/20 | 0.5521 (0.4147-0.6835) | 0.87631 | 0.906049 | - | 0.758 |

## Validation

{"artifacts_compatible": 3, "artifacts_complete": 3, "artifacts_seen": 3, "artifacts_unique": 3, "artifacts_valid": 3, "duplicate_runs": 0, "errors": 0, "warnings": 0}
