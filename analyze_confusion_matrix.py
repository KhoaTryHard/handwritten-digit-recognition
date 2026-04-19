# Module nay tong hop confusion matrix va cac chi so loi tren tap validation.
"""Analyze the confusion matrix on the validation dataset."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.evaluation import (
    build_confusion_summary,
    collect_directory_predictions,
)


VAL_DIR = project_file("my_digits_28", "val")
MODEL_PATH = project_file("models", "stage_03_final.keras")
BATCH_SIZE = 64


def main() -> None:
    """Print confusion matrix statistics for the validation set."""
    predictions = collect_directory_predictions(
        VAL_DIR,
        MODEL_PATH,
        BATCH_SIZE,
    )
    summary = build_confusion_summary(predictions)

    print(f"Classes: {list(predictions.class_names)}")
    print("\nConfusion matrix (rows=true, cols=pred):\n", summary.confusion_matrix)

    print("\nPer-class accuracy:")
    for class_result in summary.per_class_accuracy:
        print(
            f"  class {class_result.label}: "
            f"{class_result.accuracy:.4f} "
            f"({class_result.correct}/{class_result.total})"
        )

    print(f"\nNum wrong: {len(summary.wrong_indices)}/{len(predictions.y_true)}")
    print(
        "Some wrong (true -> pred):",
        list(
            zip(
                predictions.y_true[summary.wrong_indices[:20]],
                predictions.y_pred[summary.wrong_indices[:20]],
            )
        ),
    )

    print("Top confusions:")
    for rank, confusion_pair in enumerate(summary.top_confusions, start=1):
        print(
            f"{rank:02d}. true {confusion_pair.true_label} -> "
            f"pred {confusion_pair.predicted_label}: {confusion_pair.count}"
        )


if __name__ == "__main__":
    main()
