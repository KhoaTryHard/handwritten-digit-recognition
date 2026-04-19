# Module nay chua helper liet ke anh va tao ten file dich khong bi trung.
"""Small file-system helpers reused across scripts."""

from __future__ import annotations

from pathlib import Path

from digit_pipeline.config.settings import IMAGE_EXTENSIONS


def list_image_files(directory: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return image files inside a directory."""
    resolved_directory = Path(directory)
    iterator = resolved_directory.rglob("*") if recursive else resolved_directory.iterdir()

    return sorted(
        file_path
        for file_path in iterator
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_unique_destination(destination: str | Path, *, label: str = "val") -> Path:
    """Return a non-conflicting destination path."""
    candidate = Path(destination)
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        updated_candidate = candidate.with_name(
            f"{candidate.stem}_{label}{suffix}{candidate.suffix}"
        )
        if not updated_candidate.exists():
            return updated_candidate
        suffix += 1
