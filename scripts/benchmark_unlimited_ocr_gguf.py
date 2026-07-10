from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import unicodedata
from pathlib import Path


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


def strip_grounding(text: str) -> str:
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|/?[a-zA-Z_]+\|>", "", text)
    return clean_lines(text)


def llama_text(
    image_path: Path,
    llama_bin: str,
    model: Path,
    mmproj: Path,
    prompt: str,
    max_tokens: int,
    ctx_size: int,
    gpu_layers: int,
    repeat_penalty: float,
) -> tuple[str, str, float]:
    command = [
        llama_bin,
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--image",
        str(image_path),
        "-p",
        prompt,
        "--temp",
        "0",
        "-n",
        str(max_tokens),
        "-c",
        str(ctx_size),
        "--repeat-penalty",
        str(repeat_penalty),
    ]
    if gpu_layers >= 0:
        command.extend(["-ngl", str(gpu_layers)])

    start = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"llama-mtmd-cli failed for {image_path}: {result.stderr.strip()}")
    return strip_grounding(result.stdout), result.stderr.strip(), elapsed


def run_model(
    bench_root: Path,
    output_root: Path,
    llama_bin: str,
    model: Path,
    mmproj: Path,
    prompt: str,
    max_tokens: int,
    ctx_size: int,
    gpu_layers: int,
    repeat_penalty: float,
) -> None:
    for image_path in bench_items(bench_root):
        split = image_path.parent.name
        pred_dir = output_root / split
        raw_dir = output_root / "_raw" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        prediction, stderr, elapsed = llama_text(
            image_path=image_path,
            llama_bin=llama_bin,
            model=model,
            mmproj=mmproj,
            prompt=prompt,
            max_tokens=max_tokens,
            ctx_size=ctx_size,
            gpu_layers=gpu_layers,
            repeat_penalty=repeat_penalty,
        )
        (pred_dir / f"{image_path.stem}.md").write_text(prediction, encoding="utf-8")
        (raw_dir / f"{image_path.stem}.json").write_text(
            json.dumps(
                {
                    "image": str(image_path),
                    "elapsed_seconds": elapsed,
                    "llama_bin": llama_bin,
                    "model": str(model),
                    "mmproj": str(mmproj),
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "ctx_size": ctx_size,
                    "gpu_layers": gpu_layers,
                    "repeat_penalty": repeat_penalty,
                    "stderr": stderr,
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
    parser = argparse.ArgumentParser(description="Run and score Unlimited-OCR GGUF on small_bench.")
    parser.add_argument("--bench-root", default="small_bench")
    parser.add_argument("--output-root", default="bench_runs/Unlimited-OCR-GGUF-Q4_K_M")
    parser.add_argument("--model-name", default="Unlimited-OCR-GGUF-Q4_K_M")
    parser.add_argument("--llama-bin", default="llama-mtmd-cli")
    parser.add_argument("--model", default="models/Unlimited-OCR-GGUF/Unlimited-OCR-Q4_K_M.gguf")
    parser.add_argument("--mmproj", default="models/Unlimited-OCR-GGUF/mmproj-Unlimited-OCR-F16.gguf")
    parser.add_argument("--prompt", default="Free OCR.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Pass -ngl only when >= 0.")
    parser.add_argument("--repeat-penalty", type=float, default=1.05)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    bench_root = Path(args.bench_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.score_only:
        run_model(
            bench_root=bench_root,
            output_root=output_root,
            llama_bin=args.llama_bin,
            model=Path(args.model),
            mmproj=Path(args.mmproj),
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            ctx_size=args.ctx_size,
            gpu_layers=args.gpu_layers,
            repeat_penalty=args.repeat_penalty,
        )
        (output_root / "run_info.json").write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "llama_bin": args.llama_bin,
                    "gguf": args.model,
                    "mmproj": args.mmproj,
                    "prompt": args.prompt,
                    "max_tokens": args.max_tokens,
                    "ctx_size": args.ctx_size,
                    "gpu_layers": args.gpu_layers,
                    "repeat_penalty": args.repeat_penalty,
                    "note": "Requires a DeepSeek-OCR-aware llama.cpp build with llama-mtmd-cli support.",
                    "source": "sahilchachra/Unlimited-OCR-GGUF",
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
