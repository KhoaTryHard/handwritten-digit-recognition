# Module nay tach mot phan du lieu viet tay goc tu train sang validation.
"""Split the raw personal dataset into train and validation folders."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.data_loading import DataSplitConfig, split_personal_dataset


CONFIG = DataSplitConfig(
    train_dir=project_file("my_digits_new", "train"),
    val_dir=project_file("my_digits_new", "val"),
    val_ratio=0.2,
    seed=42,
)


def main() -> None:
    """Run the train/validation split for personal images."""
    summary = split_personal_dataset(CONFIG)

    # In thong ke theo tung lop de de kiem tra bo du lieu sau khi tach.
    for digit_label, moved_count in summary.moved_per_class.items():
        print(f"[{digit_label}] moved {moved_count} image(s) -> val")

    print(f"\nDone. Total moved: {summary.total_moved}")
    print(f"Train: {CONFIG.train_dir}")
    print(f"Val  : {CONFIG.val_dir}")


if __name__ == "__main__":
    main()
