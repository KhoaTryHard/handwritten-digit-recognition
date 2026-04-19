# Module nay xuat anh cua mot cap nham lan cu the de quan sat truc quan.
"""Export images for one chosen confusion pair."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.evaluation import (
    collect_directory_predictions,
    export_confusion_pair_images,
)


VAL_DIR = project_file("my_digits_28", "val")
MODEL_PATH = project_file("models", "stage_03_final.keras")
OUTPUT_DIR = project_file("reports", "confusion_pairs")

BATCH_SIZE = 64
MAX_SAVE = 50
TARGET_TRUE = 9
TARGET_PRED = 3


def main() -> None:
    """Export image files for one confusion pair."""
    predictions = collect_directory_predictions(
        VAL_DIR,
        MODEL_PATH,
        BATCH_SIZE,
        keep_images=True,
    )
    saved_paths = export_confusion_pair_images(
        predictions,
        OUTPUT_DIR,
        true_label=TARGET_TRUE,
        predicted_label=TARGET_PRED,
        max_save=MAX_SAVE,
    )
    print(f"Count saved {TARGET_TRUE}->{TARGET_PRED}: {len(saved_paths)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
