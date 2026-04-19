# Module nay xuat cac mau du doan sai ra CSV de phan tich chi tiet.
"""Export misclassified validation samples to CSV."""

from __future__ import annotations

from digit_pipeline.config import project_file
from digit_pipeline.evaluation import (
    collect_directory_predictions,
    export_misclassified_predictions,
)


VAL_DIR = project_file("my_digits_28", "val")
MODEL_PATH = project_file("models", "stage_03_final.keras")
OUTPUT_CSV = project_file("reports", "misclassified_validation.csv")
BATCH_SIZE = 64


def main() -> None:
    """Export all misclassified validation samples to CSV."""
    predictions = collect_directory_predictions(
        VAL_DIR,
        MODEL_PATH,
        BATCH_SIZE,
    )
    dataframe = export_misclassified_predictions(predictions, OUTPUT_CSV)
    print(f"Total wrong: {len(dataframe)}/{len(predictions.file_paths)}")
    print(dataframe.head(1000).to_string(index=False))
    print(f"Saved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
