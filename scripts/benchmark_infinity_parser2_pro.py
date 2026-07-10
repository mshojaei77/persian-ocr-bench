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


DEFAULT_PROMPT = """
- Extract layout information from the provided document image.
- For each layout element, output its bbox, category, and the text content within the bbox.
- Bbox format: [x1, y1, x2, y2].
- Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula', 'figure_caption', 'table_caption', 'formula_caption', 'figure_footnote', 'table_footnote', 'page_footnote', 'footer'].
- Text extraction and formatting:
  1) For 'figure', the text field must be an empty string.
  2) For 'formula', format text as LaTeX.
  3) For 'table', format text as HTML.
  4) For all other categories, format text as Markdown.
- The output text must be exactly the original text from the image, with no translation or rewriting.
- Sort all layout elements in human reading order.
- Final output must be a single JSON object.
"""


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


def extract_json_text(raw_text: str) -> str:
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return clean_lines(raw_text)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return clean_lines(raw_text)

    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                texts.append(value["text"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    return clean_lines("\n".join(texts)) or clean_lines(raw_text)


def load_model(model_path: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return processor, model


def generate_text(
    processor: Any,
    model: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[str, str]:
    import torch
    from qwen_vl_utils import process_vision_info

    pil_image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image_inputs, _ = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=text, images=image_inputs, do_resize=False, padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return clean_lines(raw_text), extract_json_text(raw_text)


def run_model(
    bench_root: Path,
    output_root: Path,
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    min_pixels: int,
    max_pixels: int,
) -> None:
    processor, model = load_model(model_path)
    for image_path in bench_items(bench_root):
        split = image_path.parent.name
        pred_dir = output_root / split
        raw_dir = output_root / "_raw" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        raw_text, prediction = generate_text(processor, model, image_path, prompt, max_new_tokens, min_pixels, max_pixels)
        elapsed = time.perf_counter() - start

        (pred_dir / f"{image_path.stem}.md").write_text(prediction, encoding="utf-8")
        (raw_dir / f"{image_path.stem}.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "elapsed_seconds": elapsed,
                    "model_path": model_path,
                    "max_new_tokens": max_new_tokens,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "raw_text": raw_text,
                    "prediction": prediction,
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
    parser = argparse.ArgumentParser(description="Run and score Infinity-Parser2-Pro on small_bench.")
    parser.add_argument("--bench-root", default="small_bench")
    parser.add_argument("--output-root", default="bench_runs/Infinity-Parser2-Pro")
    parser.add_argument("--model-path", default="models/Infinity-Parser2-Pro")
    parser.add_argument("--model-name", default="Infinity-Parser2-Pro")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--min-pixels", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=16777216)
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
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        (output_root / "run_info.json").write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "model_path": args.model_path,
                    "backend": "native transformers",
                    "max_new_tokens": args.max_new_tokens,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "note": "Infinity-Parser2 is a document parser; this runner extracts text fields from generated JSON for CER/WER scoring.",
                    "source": "infly/Infinity-Parser2-Pro",
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
