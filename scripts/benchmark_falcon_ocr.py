from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image


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


def load_model(model_path: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model


def layout_text(results: list[dict[str, Any]]) -> str:
    return clean_lines("\n".join(str(item.get("text", "")) for item in results if item.get("text")))


def generate_text(model: Any, image_path: Path, category: str, layout: bool, max_new_tokens: int) -> tuple[str, Any]:
    image = Image.open(image_path).convert("RGB")
    if layout:
        raw = model.generate_with_layout(image, max_new_tokens=max_new_tokens)
        page = raw[0] if raw and isinstance(raw[0], list) else raw
        return layout_text(page), raw
    raw = model.generate(image, category=category, max_new_tokens=max_new_tokens)
    text = raw[0] if isinstance(raw, list) else raw
    return clean_lines(str(text)), raw


def run_model(
    bench_root: Path,
    output_root: Path,
    model_path: str,
    category: str,
    layout: bool,
    max_new_tokens: int,
) -> None:
    model = load_model(model_path)
    for image_path in bench_items(bench_root):
        split = image_path.parent.name
        pred_dir = output_root / split
        raw_dir = output_root / "_raw" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        prediction, raw = generate_text(model, image_path, category, layout, max_new_tokens)
        elapsed = time.perf_counter() - start

        (pred_dir / f"{image_path.stem}.md").write_text(prediction, encoding="utf-8")
        (raw_dir / f"{image_path.stem}.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "elapsed_seconds": elapsed,
                    "model_path": model_path,
                    "category": category,
                    "layout": layout,
                    "max_new_tokens": max_new_tokens,
                    "raw": raw,
                    "prediction": prediction,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
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
    parser = argparse.ArgumentParser(description="Run and score Falcon-OCR on small_bench.")
    parser.add_argument("--bench-root", default="small_bench")
    parser.add_argument("--output-root", default="bench_runs/Falcon-OCR")
    parser.add_argument("--model-path", default="models/Falcon-OCR")
    parser.add_argument("--model-name", default="Falcon-OCR")
    parser.add_argument("--category", default="plain")
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        run_model(
            bench_root=bench_root,
            output_root=output_root,
            model_path=args.model_path,
            category=args.category,
            layout=args.layout,
            max_new_tokens=args.max_new_tokens,
        )
        (output_root / "run_info.json").write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "model_path": args.model_path,
                    "category": args.category,
                    "layout": args.layout,
                    "max_new_tokens": args.max_new_tokens,
                    "note": "Falcon-OCR exposes custom model.generate(image, category=...) and optional generate_with_layout.",
                    "source": "tiiuae/Falcon-OCR",
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
