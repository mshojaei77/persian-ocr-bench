"""Workspace and logical-path handling shared by every benchmark command."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import tomllib
from typing import Iterable


WORKSPACE_ENV = "PERSIAN_OCR_WORKSPACE"
PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent


class WorkspaceNotFoundError(RuntimeError):
    """Raised when repository-backed data is requested outside a workspace."""


def _parents_inclusive(path: Path) -> Iterable[Path]:
    current = path.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def is_workspace(path: Path) -> bool:
    """Return whether *path* is a Persian OCR checkout/workspace root."""
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return payload.get("project", {}).get("name") == "persian-ocr"


def discover_workspace(
    explicit: str | Path | None = None,
    *,
    start: str | Path | None = None,
    required: bool = True,
) -> Path | None:
    """Find the workspace without assuming package files live in the checkout."""
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if is_workspace(candidate):
            return candidate
        raise WorkspaceNotFoundError(f"Not a persian-ocr workspace: {candidate}")

    configured = os.environ.get(WORKSPACE_ENV, "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if is_workspace(candidate):
            return candidate
        raise WorkspaceNotFoundError(
            f"{WORKSPACE_ENV} does not point to a persian-ocr workspace: {candidate}"
        )

    search_roots = [Path(start) if start is not None else Path.cwd(), PACKAGE_ROOT]
    seen: set[Path] = set()
    for search_root in search_roots:
        for candidate in _parents_inclusive(search_root):
            if candidate in seen:
                continue
            seen.add(candidate)
            if is_workspace(candidate):
                return candidate

    if required:
        raise WorkspaceNotFoundError(
            "Could not find the persian-ocr workspace. Run inside the checkout, "
            f"pass --workspace, or set {WORKSPACE_ENV}."
        )
    return None


@dataclass(frozen=True)
class WorkspacePaths:
    """Canonical mutable paths for a repository-backed benchmark workspace."""

    root: Path

    @classmethod
    def discover(cls, explicit: str | Path | None = None) -> "WorkspacePaths":
        root = discover_workspace(explicit, required=True)
        assert root is not None
        return cls(root)

    @property
    def catalog(self) -> Path:
        return self.root / "models.yaml"

    @property
    def small_bench(self) -> Path:
        return self.root / "small_bench"

    @property
    def manifest(self) -> Path:
        return self.small_bench / "manifest.jsonl"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def bench_runs(self) -> Path:
        return self.root / "bench_runs"


def workspace_paths(explicit: str | Path | None = None) -> WorkspacePaths:
    return WorkspacePaths.discover(explicit)


def resolve_path(value: str | Path, *, base: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def logical_path(value: str | Path, *, base: str | Path) -> str:
    """Return a portable relative POSIX path, rejecting paths outside *base*."""
    path = resolve_path(value, base=base)
    root = Path(base).expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside logical root {root}: {path}") from exc
    return relative.as_posix()


def ensure_logical_path(value: str) -> str:
    """Validate and normalize a relative artifact path."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Logical path must be a non-empty string")
    if "\\" in value:
        raise ValueError(f"Logical path must use POSIX separators: {value!r}")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.drive
        or ".." in pure.parts
        or (pure.parts and ":" in pure.parts[0])
    ):
        raise ValueError(f"Logical path must be relative and contained: {value!r}")
    normalized = pure.as_posix()
    if normalized in {"", "."}:
        raise ValueError("Logical path must identify a file or directory")
    return normalized


# Compatibility for code migrating from flat checkout-bound modules. New code
# should call workspace_paths() at execution time instead of freezing this value.
REPO_ROOT = discover_workspace(required=False) or Path.cwd().resolve()


__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "SOURCE_ROOT",
    "WORKSPACE_ENV",
    "WorkspaceNotFoundError",
    "WorkspacePaths",
    "discover_workspace",
    "ensure_logical_path",
    "is_workspace",
    "logical_path",
    "resolve_path",
    "workspace_paths",
]
