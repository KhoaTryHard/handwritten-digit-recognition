# Module nay chua helper tao duong dan tu thu muc goc cua project.
"""Path utilities for project-relative files and directories."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_file(*parts: str) -> Path:
    """Return a path inside the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def project_path(*parts: str) -> str:
    """Return a string path inside the project root."""
    return str(project_file(*parts))


def ensure_directory(directory: str | Path) -> Path:
    """Create a directory if needed and return it."""
    resolved_directory = Path(directory)
    resolved_directory.mkdir(parents=True, exist_ok=True)
    return resolved_directory


def ensure_parent_directory(path: str | Path) -> Path:
    """Create the parent directory for a target file and return the file path."""
    resolved_path = Path(path)
    ensure_directory(resolved_path.parent)
    return resolved_path
