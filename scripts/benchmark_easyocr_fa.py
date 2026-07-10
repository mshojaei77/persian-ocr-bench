from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any


def clean_lines(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"[\u200b\u200d\ufeff]", "", text)
    return clean_lines(text)


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: list[str] | str, prediction: list[str] | str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return edit_distance(reference, prediction) / len(reference)


def line_exact(reference: str, prediction: str) -> float:
    ref_lines = [line for line in reference.split("\n") if line.strip()]
    pred_lines = {line for line in prediction.split("\n") if line.strip()}
    if not ref_lines:
        return 1.0 if not pred_lines else 0.0
    return sum(1 for line in ref_lines if line in pred_lines) / len(ref_lines)


def bench_items(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.jpg"), key=lambda path: (path.parent.name, int(path.stem)))


def box_y(result: tuple[Any, str, float]) -> float:
    box = result[0]
    try:
        return min(float(point[1]) for point in box)
    except Exception:
        return 0.0


def box_x(result: tuple[Any, str, float]) -> float:
    box = result[0]
    try:
        return min(float(point[0]) for point in box)
    except Exception:
        return 0.0


def easyocr_text(results: list[tuple[Any, str, float]], rtl: bool) -> str:
    # ponytail: simple y/x ordering; replace with line clustering if EasyOCR output is visibly shuffled.
    ordered = sorted(results, key=lambda item: (box_y(item), -box_x(item) if rtl else box_x(item)))
    return clean_lines("\n".join(text for _, text, _ in ordered))


def to_plain(results: list[tuple[Any, str, float]]) -> list[dict[str, Any]]:
    rows = []
    for box, text, confidence in results:
        if hasattr(box, "tolist"):
            box = box.tolist()
        rows.append({"box": box, "text": text, "confidence": float(confidence)})
    return rows


def load_reader(languages: list[str], model_dir: Path, gpu: bool) -> Any:
    import easyocr

    return easyocr.Reader(
        languages,
        gpu=gpu,
        model_storage_directory=str(model_dir),
        download_enabled=True,
    )


def run_model(
    bench_root: Path,
    output_root: Path,
    languages: list[str],
    model_dir: Path,
    gpu: bool,
    paragraph: bool,
    rtl: bool,
) -> None:
    reader = load_reader(languages, model_dir, gpu)
    for image_path in bench_items(bench_root):
        split = image_path.parent.name
        pred_dir = output_root / split
        raw_dir = output_root / "_raw" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        results = reader.readtext(str(image_path), detail=1, paragraph=paragraph)
        elapsed = time.perf_counter() - start
        prediction = easyocr_text(results, rtl=rtl)

        (pred_dir / f"{image_path.stem}.md").write_text(prediction, encoding="utf-8")
        (raw_dir / f"{image_path.stem}.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "elapsed_seconds": elapsed,
                    "languages": languages,
                    "gpu": gpu,
                    "paragraph": paragraph,
                    "prediction": prediction,
                    "results": to_plain(results),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{split}/{image_path.name}: {elapsed:.1f}s")


def score_predictions(bench_root: Path, output_root: Path, model_name: str) -> None:
    rows: list[dict[str, str | float | int]] = []
    for image_path in bench_items(bench_root):
        split = image_path.parent.name
        reference = image_path.with_suffix(".md").read_text(encoding="utf-8")
        pred_path = output_root / split / f"{image_path.stem}.md"
        prediction = pred_path.read_text(encoding="utf-8") if pred_path.exists() else ""
        ref_norm = normalize_text(reference)
        pred_norm = normalize_text(prediction)
        rows.append(
            {
                "model": model_name,
                "split": split,
                "item": image_path.stem,
                "cer_norm": round(error_rate(ref_norm, pred_norm), 6),
                "wer_norm": round(error_rate(ref_norm.split(), pred_norm.split()), 6),
                "line_exact_norm": round(line_exact(ref_norm, pred_norm), 6),
                "ref_chars": len(ref_norm),
                "pred_chars": len(pred_norm),
            }
        )

    with (output_root / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for split in ["typed", "hand-written", "all"]:
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not selected:
            continue
        summary_rows.append(
            {
                "model": model_name,
                "split": split,
                "mean_cer_norm": round(sum(float(row["cer_norm"]) for row in selected) / len(selected), 6),
                "mean_wer_norm": round(sum(float(row["wer_norm"]) for row in selected) / len(selected), 6),
                "mean_line_exact_norm": round(
                    sum(float(row["line_exact_norm"]) for row in selected) / len(selected), 6
                ),
                "items": len(selected),
            }
        )

    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and score EasyOCR Persian on small_bench.")
    parser.add_argument("--bench-root", default="small_bench")
    parser.add_argument("--output-root", default="bench_runs/easyocr-fa")
    parser.add_argument("--model-name", default="easyocr-fa")
    parser.add_argument("--model-dir", default="models/easyocr")
    parser.add_argument("--languages", nargs="+", default=["fa", "en"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--paragraph", action="store_true")
    parser.add_argument("--ltr", action="store_true", help="Sort boxes left-to-right instead of Persian right-to-left.")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        run_model(
            bench_root=bench_root,
            output_root=output_root,
            languages=args.languages,
            model_dir=Path(args.model_dir),
            gpu=args.gpu,
            paragraph=args.paragraph,
            rtl=not args.ltr,
        )
        (output_root / "run_info.json").write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "model_dir": args.model_dir,
                    "languages": args.languages,
                    "gpu": args.gpu,
                    "paragraph": args.paragraph,
                    "rtl_sort": not args.ltr,
                    "source": "jaidedai/easyocr",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    score_predictions(bench_root, output_root, args.model_name)
    print(output_root / "summary.csv")


if __name__ == "__main__":
    main()
