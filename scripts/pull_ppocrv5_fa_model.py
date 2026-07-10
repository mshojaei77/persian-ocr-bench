from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up PP-OCRv5 multilingual Persian models.")
    parser.add_argument("--lang", default="fa")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--image", default="small_bench/typed/1.jpg")
    args = parser.parse_args()

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    # PaddleOCR downloads/caches the selected detection + fa recognition models on first use.
    result = ocr.predict(args.image)
    print(f"Pulled PP-OCRv5 lang={args.lang}; warmup pages={len(result)}; device={args.device}")


if __name__ == "__main__":
    main()
