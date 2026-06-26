# Module nay chua helper tao duong dan tu thu muc goc cua project.
"""Tien ich duong dan cho file va thu muc tinh tu project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_file(*parts: str) -> Path:
    """Tra ve duong dan ben trong project root."""
    return PROJECT_ROOT.joinpath(*parts)


def project_path(*parts: str) -> str:
    """Tra ve duong dan dang chuoi ben trong project root."""
    return str(project_file(*parts))


def ensure_directory(directory: str | Path) -> Path:
    """Tao thu muc neu can va tra ve thu muc do."""
    resolved_directory = Path(directory)
    resolved_directory.mkdir(parents=True, exist_ok=True)
    return resolved_directory


def ensure_parent_directory(path: str | Path) -> Path:
    """Tao thu muc cha cho file dich va tra ve duong dan file."""
    resolved_path = Path(path)
    ensure_directory(resolved_path.parent)
    return resolved_path
