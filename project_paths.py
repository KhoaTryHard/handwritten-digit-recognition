# Module nay giu tuong thich nguoc cho helper tao duong dan theo project root.
"""Backward-compatible project path helpers."""

from __future__ import annotations

from pathlib import Path

from digit_pipeline.config import project_file as _project_file
from digit_pipeline.config import project_path as _project_path


PROJECT_ROOT = _project_file()


def project_file(*parts: str) -> Path:
    """Return a project-relative path."""
    return _project_file(*parts)


def project_path(*parts: str) -> str:
    """Return a project-relative path as a string."""
    return _project_path(*parts)
