# Phase 1 screening report

Dataset: `smoke20-v1` (`8abf9b32bb2746d85b75c07049f35cebd245d42dc656c1c955a219cfd19a7dbe`)

Protocol: `phase1_screening` / `full_page_ocr` (`ce683321a491e5a64f2e6c54b8418820767279dee30b743aff7a14b30494bfc8`)

This 20-image phase is a viability screen, not a general Persian OCR ranking.

| Decision | Model | Class | Coverage | Macro CER (95% CI) | Mean sec/image |
|---|---|---|---:|---:|---:|
| Hold | `easyocr_fa` | full_page_detector_recognizer_pipeline | 20/20 | 0.4534 (0.3596-0.5459) | 14.156 |
| Hold | `ppocrv5_arabic_mobile_rec` | full_page_detector_recognizer_pipeline | 20/20 | 0.5546 (0.4252-0.6807) | 14.555 |
| Hold | `tesseract_fas` | full_page_detector_recognizer_pipeline | 20/20 | 0.5521 (0.4147-0.6835) | 0.758 |

## Validation

{"artifacts_compatible": 3, "artifacts_complete": 3, "artifacts_seen": 3, "artifacts_unique": 3, "artifacts_valid": 3, "duplicate_runs": 0, "errors": 0, "warnings": 0}
