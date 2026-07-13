"""Deterministic no-pytest regression smoke for reporting eligibility rules."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from persian_ocr.reporting import (  # noqa: E402
    BENCHMARK_SCHEMA,
    PROTOCOL_IDENTITY_SCHEMA,
    RUN_IDENTITY_SCHEMA,
    build_report,
    canonical_sha256,
    validate_artifact,
    write_report,
)


def make_artifact(
    model_id: str,
    cer: float,
    *,
    phase: str = "screening",
    dataset_id: str = "smoke20-v1",
    error_pages: int = 0,
    schema: str = BENCHMARK_SCHEMA,
) -> dict[str, object]:
    protocol = {
        "schema": PROTOCOL_IDENTITY_SCHEMA,
        "id": "smoke20-v1" if phase == "screening" else "persian-heldout-v1",
        "phase": phase,
        "track": "full_page_recognition",
        "version": "1.0.0",
    }
    protocol["digest"] = canonical_sha256(protocol)
    image_hashes = {
        f"small_bench/{'typed' if index < 10 else 'hand-written'}/{index % 10 + 1}.jpg": canonical_sha256(
            {"dataset": dataset_id, "image": index}
        )
        for index in range(20)
    }
    sample_hashes = [
        {
            "sample_id": f"sample-{index + 1:03d}",
            "image_sha256": image_hash,
            "reference_sha256": canonical_sha256(
                {"dataset": dataset_id, "reference": index}
            ),
        }
        for index, image_hash in enumerate(image_hashes.values())
    ]
    digest_lines = [f"{dataset_id}\n1.0.0"] + [
        f"{sample['sample_id']}|{sample['image_sha256']}|{sample['reference_sha256']}"
        for sample in sample_hashes
    ]
    dataset_digest = hashlib.sha256(
        ("\n".join(digest_lines) + "\n").encode("utf-8")
    ).hexdigest()
    dataset = {
        "schema": "persian_ocr_dataset_identity_v2",
        "id": dataset_id,
        "version": "1.0.0",
        "digest": dataset_digest,
        "dataset_sha256": dataset_digest,
        "manifest": "small_bench/manifest.jsonl",
        "manifest_sha256": canonical_sha256(
            {"dataset": dataset_id, "kind": "manifest"}
        ),
        "reference_corpora_sha256": {
            "small_bench/typed.json": canonical_sha256(
                {"dataset": dataset_id, "corpus": "typed"}
            ),
            "small_bench/hand-written.json": canonical_sha256(
                {"dataset": dataset_id, "corpus": "handwritten"}
            ),
        },
        "images_sha256": image_hashes,
        "n_samples": 20,
        "splits": {"phase1_screening": 20},
        "content_types": {"printed": 10, "handwritten": 10},
        "samples": sample_hashes,
    }
    results: list[dict[str, object]] = []
    for index, image in enumerate(image_hashes):
        failed = index >= 20 - error_pages
        result: dict[str, object] = {
            "sample_id": f"sample-{index + 1:03d}",
            "image": image,
            "split": "phase1_screening",
            # Track is intentionally useless for content classification.  The
            # report must use explicit content_type instead.
            "track": "social_media_graphics",
            "content_type": "printed" if index < 10 else "handwritten",
            "status": "error" if failed else "ok",
            "seconds": round(1.0 + cer + index / 1000, 4),
        }
        if failed:
            result["error"] = "SyntheticError: deterministic smoke"
        else:
            result.update(
                {
                    "reference": "متن مرجع",
                    "prediction": "متن پیش‌بینی",
                    "metrics": {"cer_canonical": cer, "wer_canonical": cer},
                }
            )
        results.append(result)
    run_identity = {
        "schema": RUN_IDENTITY_SCHEMA,
        "protocol": protocol,
        "dataset": dataset,
        "model": {
            "id": model_id,
            "class": "full_page_pipeline",
            "checkpoint_type": "synthetic",
            "identity": {"revision": "smoke"},
        },
        "runner": {"package": "persian-ocr", "version": "smoke"},
        "config": {"temperature": 0},
        "environment": {"python": "smoke", "platform": "smoke", "packages": {}},
    }
    run_identity["digest"] = canonical_sha256(run_identity)
    ok = 20 - error_pages
    return {
        "schema": schema,
        "run_id": run_identity["digest"],
        "run_identity": run_identity,
        "summary": {
            "model": {"id": model_id, "class": "full_page_pipeline"},
            "protocol": {
                "id": protocol["id"],
                "phase": phase,
                "track": protocol["track"],
                "version": protocol["version"],
                "digest": protocol["digest"],
            },
            "dataset": {"id": dataset_id, "digest": dataset_digest, "n_samples": 20},
            "counts": {"total": 20, "ok": ok, "error": error_pages, "skipped": 0},
            "metrics": {
                "macro_page_cer_canonical": cer if ok else None,
                "mean_wer_canonical": cer if ok else None,
                "micro_corpus_cer_canonical": cer if ok else None,
            },
            "operations": {"mean_seconds_per_image": 1.0 + cer},
        },
        "results": results,
    }


def write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def row(report: dict[str, object], model_id: str) -> dict[str, object]:
    return next(item for item in report["rows"] if item["model_id"] == model_id)  # type: ignore[index]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="persian-ocr-reporting-") as directory:
        root = Path(directory)
        paths = [
            write(root / "01-perfect.json", make_artifact("perfect", 0.0)),
            write(root / "02-imperfect.json", make_artifact("imperfect", 0.1)),
            write(
                root / "03-partial.json", make_artifact("partial", 0.05, error_pages=1)
            ),
            write(
                root / "04-incompatible-schema.json",
                make_artifact("old-schema", 0.0, schema="persian_ocr_benchmark_v1"),
            ),
            write(
                root / "05-incompatible-dataset.json",
                make_artifact("other-dataset", 0.0, dataset_id="other-v1"),
            ),
        ]
        screen = build_report(paths, mode="screening")

        assert row(screen, "perfect")["macro_page_cer_canonical"] == 0.0
        assert row(screen, "perfect")["decision"] == "Advance"
        assert row(screen, "imperfect")["decision"] == "Hold"
        assert row(screen, "partial")["decision"] == "Hold"
        assert row(screen, "partial")["coverage"] == 0.95
        assert row(screen, "old-schema")["decision"] == "Blocked"
        assert row(screen, "other-dataset")["decision"] == "Blocked"
        encoded = json.dumps(screen, ensure_ascii=False)
        assert '"rank"' not in encoded
        assert str(root) not in encoded

        perfect_slices = {
            item["content_type"]: item["expected"]
            for item in screen["slices"]
            if item["model_id"] == "perfect"
        }
        assert perfect_slices == {"printed": 10, "handwritten": 10}

        output = root / "report"
        output.mkdir()
        stale_chart = output / "leaderboard_cer.png"
        stale_chart.write_bytes(b"stale")
        generated = write_report(screen, output)
        assert not stale_chart.exists()
        manifest = json.loads(
            (output / "report_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["generated_files"] == sorted(
            name for name in generated if name != "report_manifest.json"
        )
        assert all((output / name).is_file() for name in generated)

        invalid = validate_artifact(paths[3])
        assert not invalid.valid
        assert any(
            issue.code == "artifact_schema_incompatible" for issue in invalid.issues
        )

        final_paths = [
            write(
                root / "11-final-perfect.json",
                make_artifact("final-perfect", 0.0, phase="final"),
            ),
            write(
                root / "12-final-imperfect.json",
                make_artifact("final-imperfect", 0.1, phase="final"),
            ),
            write(
                root / "13-final-partial.json",
                make_artifact("final-partial", 0.05, phase="final", error_pages=1),
            ),
        ]
        final = build_report(final_paths, mode="final")
        assert [item["model_id"] for item in final["rows"]] == [
            "final-perfect",
            "final-imperfect",
        ]
        assert [item["rank"] for item in final["rows"]] == [1, 2]
        assert final["rows"][1]["paired_comparison"]["conclusion"] == "worse"
        assert final["rows"][1]["paired_comparison"]["ci95"] == [0.1, 0.1]
        assert any(item["model_id"] == "final-partial" for item in final["excluded"])

    print("PASS: incompatible schemas and datasets are blocked")
    print("PASS: partial 19/20 coverage cannot advance or enter the final leaderboard")
    print("PASS: perfect 0.0 CER remains first in final mode")
    print(
        "PASS: explicit content_type produces exactly 10 printed and 10 handwritten pages"
    )
    print("PASS: Phase 1 output contains decisions and no numeric rank")
    print(
        "PASS: generated-file manifest is deterministic and stale legacy charts are removed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
