from __future__ import annotations

import argparse
import subprocess


def run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="List or download Kraken Persian/Arabic-script OCR models.")
    parser.add_argument("--doi", help="Kraken repository DOI to download, for example 10.5281/zenodo.xxxxxxxx")
    parser.add_argument("--language", default="fas")
    parser.add_argument("--script", default="Arab")
    parser.add_argument("--kraken-bin", default="kraken")
    args = parser.parse_args()

    if args.doi:
        run([args.kraken_bin, "get", args.doi])
        return

    print("Persian language candidates:")
    run([args.kraken_bin, "show", "--recognition", "--language", args.language])
    print("\nArabic-script candidates:")
    run([args.kraken_bin, "show", "--recognition", "--script", args.script])
    print("\nDownload one with:")
    print(f"  {args.kraken_bin} get <DOI>")
    print("or:")
    print("  uv run python scripts\\pull_kraken_fas_model.py --doi <DOI>")


if __name__ == "__main__":
    main()
