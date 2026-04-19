# Module nay tach mot phan anh train sang validation de dung lai o nhieu script.
"""Helpers for splitting directory-based handwritten datasets."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from digit_pipeline.config.settings import DEFAULT_SEED, DIGIT_LABELS
from digit_pipeline.utils import build_unique_destination, list_image_files


@dataclass(frozen=True, kw_only=True)
class DataSplitConfig:
    """Configuration for moving train images into a validation directory."""

    train_dir: Path
    val_dir: Path
    val_ratio: float = 0.2
    seed: int = DEFAULT_SEED


@dataclass(frozen=True)
class SplitSummary:
    """Summary of a train/validation split operation."""

    moved_per_class: dict[str, int]
    total_moved: int


def split_personal_dataset(config: DataSplitConfig) -> SplitSummary:
    """Move a portion of each digit folder from train to validation."""
    if not config.train_dir.is_dir():
        raise FileNotFoundError(f"Missing training directory: {config.train_dir}")

    randomizer = random.Random(config.seed)
    config.val_dir.mkdir(parents=True, exist_ok=True)
    moved_per_class: dict[str, int] = {}
    total_moved = 0

    # Duyet tung lop chu so de giu ti le tach on dinh theo moi thu muc.
    for digit_label in DIGIT_LABELS:
        source_dir = config.train_dir / digit_label
        destination_dir = config.val_dir / digit_label

        if not source_dir.is_dir():
            moved_per_class[digit_label] = 0
            continue

        source_files = list_image_files(source_dir)
        if not source_files:
            moved_per_class[digit_label] = 0
            continue

        randomizer.shuffle(source_files)
        if len(source_files) >= 5:
            move_count = max(1, int(round(len(source_files) * config.val_ratio)))
        else:
            move_count = min(1, len(source_files))

        destination_dir.mkdir(parents=True, exist_ok=True)
        for file_path in source_files[:move_count]:
            destination_path = build_unique_destination(destination_dir / file_path.name)
            shutil.move(str(file_path), str(destination_path))

        moved_per_class[digit_label] = move_count
        total_moved += move_count

    return SplitSummary(
        moved_per_class=moved_per_class,
        total_moved=total_moved,
    )
