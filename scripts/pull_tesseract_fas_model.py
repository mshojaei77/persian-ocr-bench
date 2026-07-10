from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://github.com/tesseract-ocr/tessdata/raw/main/fas.traineddata"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Tesseract Persian fas.traineddata.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--tessdata-dir", default="models/tessdata")
    args = parser.parse_args()

    out_dir = Path(args.tessdata_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fas.traineddata"
    urlretrieve(args.url, out_path)
    print(f"Downloaded to: {out_path}")


if __name__ == "__main__":
    main()
