from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_PROMPT = (
    "Below is the image of one page of a document. Return only the plain text "
    "transcription as if reading it naturally. Preserve Persian text, numbers, "
    "punctuation, line breaks, and any Latin text. Do not translate, summarize, "
    "explain, or hallucinate."
)


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


def load_model(base_model: str, adapter_model: str) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_model)
    model.eval()
    processor = AutoProcessor.from_pretrained(adapter_model, trust_remote_code=True)
    return processor, model


def generate_text(processor: Any, model: Any, image_path: Path, prompt: str, max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_image = Path(tmp_dir) / "image.png"
        Image.open(image_path).convert("RGB").save(tmp_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{tmp_image.as_posix()}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return clean_lines(
            processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )


def run_model(
    bench_root: Path,
    output_root: Path,
    base_model: str,
    adapter_model: str,
    prompt: str,
    max_new_tokens: int,
) -> None:
    processor, model = load_model(base_model, adapter_model)
    for image_path in bench_items(bench_root):
        split = image_path.parent.name
        pred_dir = output_root / split
        raw_dir = output_root / "_raw" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        prediction = generate_text(processor, model, image_path, prompt, max_new_tokens)
        elapsed = time.perf_counter() - start

        (pred_dir / f"{image_path.stem}.md").write_text(prediction, encoding="utf-8")
        (raw_dir / f"{image_path.stem}.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "elapsed_seconds": elapsed,
                    "prompt": prompt,
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
    parser = argparse.ArgumentParser(description="Run and score Khanandeh Persian OCR on small_bench.")
    parser.add_argument("--bench-root", default="small_bench")
    parser.add_argument("--output-root", default="bench_runs/Khanandeh-0.1-Persian-OCR-2B-Instruct")
    parser.add_argument("--base-model", default="models/qwen2-vl-2b-instruct-unsloth-bnb-4bit")
    parser.add_argument("--adapter-model", default="models/Khanandeh-0.1-Persian-OCR-2B-Instruct")
    parser.add_argument("--model-name", default="Khanandeh-0.1-Persian-OCR-2B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        run_model(
            bench_root=bench_root,
            output_root=output_root,
            base_model=args.base_model,
            adapter_model=args.adapter_model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        (output_root / "run_info.json").write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "base_model": args.base_model,
                    "adapter_model": args.adapter_model,
                    "max_new_tokens": args.max_new_tokens,
                    "prompt": args.prompt,
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
