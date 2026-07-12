"""Build a comparable leaderboard from benchmark JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "bench_runs"
DEFAULT_OUTPUT = DEFAULT_INPUT / "leaderboard"


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def metric(summary: dict[str, Any], *names: str) -> float | None:
    primary = summary.get("primary_results", {})
    for name in names:
        value = number(primary.get(name))
        if value is not None:
            return value
    return None


def load_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    benchmark = summary.get("benchmark", {})
    model = summary.get("model", {})
    config = summary.get("config", {})
    operations = summary.get("operations", {})
    primary_ops = operations.get("primary_configuration", operations)
    model_id = model.get("id") or path.stem
    return {
        "rank": None,
        "model_id": model_id,
        "artifact": path.name,
        "schema": benchmark.get("schema"),
        "benchmark": benchmark.get("name"),
        "scope": benchmark.get("scope"),
        "n_images": summary.get("n_images", summary.get("n_runs")),
        "n_ok": summary.get("n_ok"),
        "n_err": summary.get("n_err", summary.get("n_skipped")),
        "cer": metric(summary, "macro_page_CER_canonical", "macro_CER_canonical"),
        "wer": metric(summary, "mean_WER_canonical", "mean_WER"),
        "micro_cer": metric(summary, "micro_corpus_CER_canonical"),
        "mean_seconds": number(
            primary_ops.get("mean_seconds_per_run")
            or operations.get("mean_seconds_per_run")
        ),
        "median_seconds": number(
            primary_ops.get("median_seconds_per_run")
            or operations.get("median_seconds_per_run")
        ),
        "source": str(path),
        "config": config.get("variant") or config.get("primary_lang"),
    }


def collect_track_rows(input_dir: Path) -> list[dict[str, Any]]:
    """Aggregate per-image results into typed and hand-written views."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_id = payload.get("summary", {}).get("model", {}).get("id") or path.stem
        for result in payload.get("results", []):
            if result.get("error"):
                continue
            track = str(result.get("track", "")).lower()
            category = "hand-written" if "hand" in track else "typed"
            grouped.setdefault((model_id, category), []).append(result)
    rows = []
    for (model_id, category), results in sorted(grouped.items()):
        cers = [number(item.get("cer_grapheme_canonical")) for item in results]
        wers = [number(item.get("wer_canonical")) for item in results]
        cers = [value for value in cers if value is not None]
        wers = [value for value in wers if value is not None]
        rows.append({
            "model_id": model_id,
            "category": category,
            "n_images": len(results),
            "cer": sum(cers) / len(cers) if cers else None,
            "wer": sum(wers) / len(wers) if wers else None,
            "mean_seconds": sum(float(item["seconds"]) for item in results) / len(results),
        })
    return rows


def collect(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            rows.append(load_row(path))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"[WARN] skipped {path.name}: {exc}")
    rows.sort(key=lambda row: (row["cer"] is None, row["cer"] or float("inf"), row["model_id"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def write_artifacts(rows: list[dict[str, Any]], track_rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["rank", "model_id", "cer"]
    with (output_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "leaderboard.json").write_text(
        json.dumps({"schema": "persian_ocr_leaderboard_v1", "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "leaderboard_by_type.json").write_text(
        json.dumps({"schema": "persian_ocr_leaderboard_v1", "rows": track_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if track_rows:
        with (output_dir / "leaderboard_by_type.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(track_rows[0]))
            writer.writeheader()
            writer.writerows(track_rows)


def write_charts(rows: list[dict[str, Any]], track_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["cer"] is not None]
    if not valid:
        return
    labels = [row["model_id"] for row in valid][::-1]
    cer = [row["cer"] for row in valid][::-1]
    fig, ax = plt.subplots(figsize=(10, max(3.5, len(valid) * 0.55)))
    ax.barh(labels, cer, color="#2f6f9f")
    ax.set(title="Benchmark leaderboard", xlabel="Macro page CER (lower is better)", xlim=(0, max(1.0, max(cer) * 1.12)))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "leaderboard_cer.png", dpi=160)
    plt.close(fig)

    timed = [row for row in valid if row["mean_seconds"] is not None]
    if timed:
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(timed) * 0.55)))
        timed = timed[::-1]
        ax.barh([row["model_id"] for row in timed], [row["mean_seconds"] for row in timed], color="#c58a32")
        ax.set(title="Benchmark latency", xlabel="Mean seconds per image (lower is better)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "leaderboard_latency.png", dpi=160)
        plt.close(fig)

    for category in ("typed", "hand-written"):
        category_rows = [row for row in track_rows if row["category"] == category and row["cer"] is not None]
        if not category_rows:
            continue
        category_rows.sort(key=lambda row: row["cer"], reverse=True)
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(category_rows) * 0.55)))
        ax.barh([row["model_id"] for row in category_rows], [row["cer"] for row in category_rows], color="#2f6f9f")
        ax.set(title=f"{category.title()} benchmark leaderboard", xlabel="Mean page CER (lower is better)", xlim=(0, max(1.0, max(row["cer"] for row in category_rows) * 1.12)))
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"leaderboard_{category.replace('-', '_')}.png", dpi=160)
        plt.close(fig)

    if len(valid) >= 2 and timed:
        fig, ax = plt.subplots(figsize=(7, 5))
        for row in valid:
            if row["mean_seconds"] is not None:
                ax.scatter(row["mean_seconds"], row["cer"], color="#2f6f9f")
                ax.annotate(row["model_id"], (row["mean_seconds"], row["cer"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set(title="Accuracy versus latency", xlabel="Mean seconds per image", ylabel="Macro page CER (lower is better)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "leaderboard_accuracy_latency.png", dpi=160)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = collect(args.input)
    track_rows = collect_track_rows(args.input)
    write_artifacts(rows, track_rows, args.output)
    write_charts(rows, track_rows, args.output)
    print(f"Loaded {len(rows)} benchmark(s) from {args.input}")
    print(f"Artifacts: {args.output}")
    print("\nRank  Model                              CER       WER       sec/image  OK/total")
    for row in rows:
        cer = "-" if row["cer"] is None else f"{row['cer']:.4f}"
        wer = "-" if row["wer"] is None else f"{row['wer']:.4f}"
        seconds = "-" if row["mean_seconds"] is None else f"{row['mean_seconds']:.2f}"
        print(f"{row['rank']:>4}  {row['model_id'][:34]:<34} {cer:>8} {wer:>8} {seconds:>10}  {row['n_ok']}/{row['n_images']}")
    print("\nBy input type")
    for row in track_rows:
        print(f"  {row['category']:<12} {row['model_id'][:30]:<30} CER={row['cer']:.4f}  n={row['n_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
