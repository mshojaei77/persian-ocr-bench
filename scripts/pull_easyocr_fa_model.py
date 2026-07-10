from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/warm EasyOCR Persian models.")
    parser.add_argument("--model-dir", default="models/easyocr")
    parser.add_argument("--languages", nargs="+", default=["fa", "en"])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    import easyocr

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        args.languages,
        gpu=args.gpu,
        model_storage_directory=str(model_dir),
        download_enabled=True,
    )
    print(f"Loaded EasyOCR languages={args.languages}; model_dir={model_dir}; gpu={args.gpu}")
    print(f"Recognizer: {type(reader.recognizer).__name__}")


if __name__ == "__main__":
    main()
