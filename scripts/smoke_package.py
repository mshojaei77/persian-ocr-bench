"""Build, install, import, and invoke the package without pytest."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODULES = {
    "persian_ocr/__init__.py",
    "persian_ocr/__main__.py",
    "persian_ocr/artifacts.py",
    "persian_ocr/catalog.py",
    "persian_ocr/dataset.py",
    "persian_ocr/metrics.py",
    "persian_ocr/normalization.py",
    "persian_ocr/paths.py",
    "persian_ocr/preprocessing.py",
}


def run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print(">", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep temporary artifacts.")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("ERROR: uv is required")

    temporary = Path(tempfile.mkdtemp(prefix="persian-ocr-smoke-"))
    cleanup = not args.keep
    try:
        run([sys.executable, "-m", "compileall", "-q", "src/persian_ocr", "scripts/smoke_package.py"])
        distribution = temporary / "dist"
        run([uv, "build", "--no-sources", "--out-dir", str(distribution)])
        wheels = list(distribution.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one wheel, found: {wheels}")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
        missing = sorted(EXPECTED_MODULES - names)
        if missing:
            raise RuntimeError(f"Wheel is missing package modules: {missing}")

        environment = temporary / "venv"
        run([uv, "venv", str(environment)])
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cli = environment / ("Scripts/persian-ocr.exe" if sys.platform == "win32" else "bin/persian-ocr")
        run([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)])
        import_probe = (
            "import sys, persian_ocr; "
            "assert persian_ocr.__version__; "
            "forbidden={'cv2','numpy','paddleocr','pytesseract','easyocr','matplotlib'}; "
            "loaded=forbidden.intersection(sys.modules); "
            "assert not loaded, loaded; "
            "print(persian_ocr.__version__)"
        )
        run([str(python), "-c", import_probe], cwd=temporary)
        run([str(python), "-m", "persian_ocr", "--help"], cwd=temporary)
        run([str(cli), "dataset", "validate", "--help"], cwd=temporary)
        print(f"PASS: {wheel.name} contains importable package code and a working CLI")
        if args.keep:
            print(f"Artifacts kept at: {temporary}")
        return 0
    finally:
        if cleanup and temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
